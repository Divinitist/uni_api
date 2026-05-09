from openai import OpenAI
import os, datetime

def client(api_key_prefix: str, base_url: str):
    api_key = os.getenv(f'{api_key_prefix}_API_KEY')
    if api_key is None:
        raise ValueError(f'{api_key_prefix}_API_KEY 未配置')
    return OpenAI(api_key=api_key, base_url=base_url)

_QWEN_THINKING_BUDGET      = {0: 0,      1: 1024, 2: 8192,  3: 38912}
_DEEPSEEK_REASONING_EFFORT = {1: 'high', 2: 'high', 3: 'max'}
_GEMINI_REASONING_EFFORT   = {0: "none", 1: "low", 2: "medium", 3: "high"}


def _get_free_devices(min_free_gb: float = 10.0) -> list[int]:
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split('\n'):
        idx, free = line.split(', ')
        gpus.append((int(idx), int(free)))
    gpus.sort(key=lambda x: -x[1])
    return [g[0] for g in gpus if g[1] > min_free_gb * 1024]


def _call_transformers(model_name: str, messages: list[dict], max_output_tokens: int = None, temperature: float = None) -> dict:
    import torch
    
    if not hasattr(_call_transformers, '_cache'):
        _call_transformers._cache = {}

    if model_name not in _call_transformers._cache:
        free_gpus = _get_free_devices(5)
        if not free_gpus:
            raise RuntimeError("没有足够空闲显存的 GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in free_gpus)

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="cuda", dtype="bfloat16")
        except (ValueError, ImportError):
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", dtype="bfloat16")

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        _call_transformers._cache[model_name] = (model, processor)

    model, processor = _call_transformers._cache[model_name]

    has_image = any(
        isinstance(msg.get('content'), list) and
        any(b.get('type') == 'image' for b in msg['content'])
        for msg in messages
    )

    if has_image:
        from qwen_vl_utils import process_vision_info

        qwen_messages = []
        for msg in messages:
            if not isinstance(msg.get('content'), list):
                qwen_messages.append(msg)
                continue
            new_content = []
            for block in msg['content']:
                if block['type'] == 'text':
                    new_content.append({'type': 'text', 'text': block['content']})
                elif block['type'] == 'image':
                    new_content.append({'type': 'image', 'image': block['url']})
            qwen_messages.append({**msg, 'content': new_content})

        text = processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)
    else:
        qwen_messages = [
            {**msg, 'content': [
                {'type': 'text', 'text': block['content']}
                for block in msg['content']
            ] if isinstance(msg.get('content'), list) else msg['content']}
            for msg in messages
        ]
        text = processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

    gen_kwargs = {}
    if max_output_tokens:
        gen_kwargs['max_new_tokens'] = max_output_tokens
    if temperature:
        gen_kwargs['temperature'] = temperature
        gen_kwargs['do_sample'] = True

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    answer = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    prompt_tokens = inputs.input_ids.shape[1]
    completion_tokens = outputs.shape[1] - prompt_tokens

    return {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model':     model_name,
        'answer':    answer,
        'thinking':  None,
        'logprobs':  None,
        'usage': {
            'prompt_tokens':     prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens':      prompt_tokens + completion_tokens,
        }
    }


from PIL import Image
import base64, io

def _encode_local_image(path: str) -> str:
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}"


def _build_content(content: str | list) -> str | list:
    """自定义格式 → OpenAI 格式"""
    if isinstance(content, str):
        return content
    result = []
    for block in content:
        if block['type'] == 'text':
            result.append({'type': 'text', 'text': block['content']})
        elif block['type'] == 'image':
            if 'base64' in block:
                mime = block.get('mime', 'image/jpeg')
                url = f"data:{mime};base64,{block['base64']}"
            elif 'url' in block:
                raw_url = block['url']
                url = raw_url if raw_url.startswith(('http://', 'https://')) else _encode_local_image(raw_url)
            else:
                raise ValueError("image block 需要提供 'url' 或 'base64' 字段")
            result.append({'type': 'image_url', 'image_url': {'url': url}})
        else:
            raise ValueError(f"不支持的 content block 类型：{block['type']}")
    return result


def _normalize_messages(messages: list[dict] | list[str]) -> list[dict]:
    """仅处理 role 简写，保持原始自定义格式不动"""
    if isinstance(messages[0], str):
        return [
            {'role': 'system', 'content': messages[0]},
            {'role': 'user',   'content': messages[1]},
        ]
    return messages


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """自定义格式 → OpenAI 格式，仅在 API 路径调用"""
    return [{**msg, 'content': _build_content(msg['content'])} for msg in messages]


def call(
    model_name: str,
    messages: list[dict] | list[str] = None,
    enable_thinking: bool = False,
    thinking_level: int = 0,
    enable_search: bool = False,
    tools: list[dict] = None,
    output_format: str = None,
    max_output_tokens: int = None,
    temperature: float = None,
    stream: bool = False,
    logprobs: bool = False,
) -> dict:
    messages = _normalize_messages(messages)

    # HuggingFace 本地模型，直接用原始格式
    if model_name.count('/') > 0 and any(c.isupper() for c in model_name):
        return _call_transformers(model_name, messages, max_output_tokens, temperature)

    # API 路径，转 OpenAI 格式
    messages = _to_openai_messages(messages)

    common = dict(model=model_name, messages=messages)
    if stream:
        common['stream'] = True
    if max_output_tokens is not None:
        common['max_tokens'] = max_output_tokens
    if temperature is not None:
        common['temperature'] = temperature
    if logprobs:
        common['logprobs'] = True
        common['top_logprobs'] = 3
    if output_format == 'json':
        common['response_format'] = {'type': 'json_object'}

    if model_name.count('/') > 0:
        # ── OpenRouter ────────────────────────────────────────────────────
        extra = {}
        if enable_thinking:
            budgets = {1: 1024, 2: 8192, 3: 38912}
            extra['thinking'] = {'type': 'enabled', 'budget_tokens': budgets[thinking_level]}
        if enable_search:
            extra['plugins'] = [{'id': 'web'}]
        response = client('OPENROUTER', 'https://openrouter.ai/api/v1').chat.completions.create(
            **common, tools=tools, extra_body=extra if extra else None,
        )

    elif model_name.startswith('qwen'):
        # ── 阿里云百炼 ───────────────────────────────────────────────────
        extra = {}
        if enable_thinking:
            extra['enable_thinking'] = True
            extra['thinking_budget'] = _QWEN_THINKING_BUDGET[thinking_level]
        if enable_search:
            extra['enable_search'] = True
        response = client('DASHSCOPE', 'https://dashscope.aliyuncs.com/compatible-mode/v1').chat.completions.create(
            **common, tools=tools, extra_body=extra if extra else None,
        )

    elif model_name.startswith('doubao'):
        # ── 火山方舟 ─────────────────────────────────────────────────────
        if enable_search:
            raise ValueError('火山方舟 Chat API 不支持联网搜索功能')
        extra = {}
        if enable_thinking:
            thinking_types = {0: 'auto', 1: 'auto', 2: 'enabled', 3: 'enabled'}
            extra['thinking'] = {'type': thinking_types[thinking_level]}
        else:
            extra['thinking'] = {'type': 'disabled'}
        response = client('ARK', 'https://ark.cn-beijing.volces.com/api/v3').chat.completions.create(
            **common, tools=tools, extra_body=extra,
        )

    elif model_name.startswith('deepseek'):
        # ── DeepSeek ─────────────────────────────────────────────────────
        for msg in messages:
            if isinstance(msg.get('content'), list):
                raise ValueError("DeepSeek 不支持多模态输入，请使用纯文本消息")
        if enable_thinking:
            extra_kwargs = {'reasoning_effort': _DEEPSEEK_REASONING_EFFORT[thinking_level]}
        else:
            extra_kwargs = {'extra_body': {'thinking': {'type': 'disabled'}}}
        response = client('DEEPSEEK', 'https://api.deepseek.com').chat.completions.create(
            **common, **extra_kwargs, tools=tools,
        )

    elif model_name.startswith('gemini'):
        # ── Gemini ───────────────────────────────────────────────────────
        merged_tools = list(tools) if tools else []
        if enable_search:
            merged_tools.append({"type": "google_search"})
        response = client('GEMINI', 'https://generativelanguage.googleapis.com/v1beta/openai/').chat.completions.create(
            **common,
            reasoning_effort=_GEMINI_REASONING_EFFORT[thinking_level] if enable_thinking else 'none',
            tools=merged_tools if merged_tools else None,
        )

    else:
        raise ValueError(f"不支持的模型前缀：{model_name}")

    choice = response.choices[0]
    logprobs_data = None
    if logprobs and choice.logprobs:
        logprobs_data = [
            {'token': t.token, 'logprob': t.logprob}
            for t in (choice.logprobs.content or [])
        ]

    return {
        'timestamp':  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model':      model_name,
        'answer':     choice.message.content,
        'thinking':   getattr(choice.message, 'reasoning_content', None),
        'logprobs':   logprobs_data,
        'usage': {
            'prompt_tokens':     response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens':      response.usage.total_tokens,
        } if response.usage else None,
    }


if __name__ == '__main__':
    call()