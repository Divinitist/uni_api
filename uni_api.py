from openai import OpenAI
import os, datetime

def client(api_key_prefix: str, base_url: str):
    api_key = os.getenv(f'{api_key_prefix}_API_KEY')
    if api_key is None:
        raise ValueError(f'{api_key_prefix}_API_KEY 未配置')
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

_QWEN_THINKING_BUDGET    = {0: 0,      1: 1024, 2: 8192,  3: 38912}
_DEEPSEEK_REASONING_EFFORT = {1: 'high', 2: 'high', 3: 'max'}
_GEMINI_REASONING_EFFORT = {0: "none", 1: "low", 2: "medium", 3: "high"}

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
        free_gpus = _get_free_devices()
        if not free_gpus:
            raise RuntimeError("没有足够空闲显存的 GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in free_gpus)

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", dtype="bfloat16")
        except (ValueError, ImportError):
            from transformers import AutoModelForCausalLM, AutoProcessor
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="bfloat16")

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        _call_transformers._cache[model_name] = (model, processor)

    model, processor = _call_transformers._cache[model_name]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    has_image = any(
        isinstance(msg.get('content'), list) and
        any(b.get('type') == 'image_url' for b in msg['content'])
        for msg in messages
    )

    if has_image:
        from PIL import Image
        import base64, re, io, requests
        images = []
        for msg in messages:
            if not isinstance(msg.get('content'), list):
                continue
            for block in msg['content']:
                if block.get('type') != 'image_url':
                    continue
                url = block['image_url']['url']
                if url.startswith('data:'):
                    b64 = re.sub(r'^data:[^;]+;base64,', '', url)
                    images.append(Image.open(io.BytesIO(base64.b64decode(b64))))
                else:
                    images.append(Image.open(io.BytesIO(requests.get(url).content)))
        inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    else:
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
        'model': model_name,
        'answer': answer,
        'thinking': None,
        'logprobs': None,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        }
    }

from PIL import Image
import base64, mimetypes, io

# def _encode_local_image(path: str, max_size: int = 1024, quality: int = 85) -> str:
#     img = Image.open(path)
#     img.thumbnail((max_size, max_size))
#     buf = io.BytesIO()
#     img.save(buf, format='JPEG', quality=quality)
#     return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
def _encode_local_image(path: str) -> str:
    from PIL import Image
    import base64, io
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)  # 只转格式，不缩尺寸，质量保持高
    data = base64.b64encode(buf.getvalue()).decode()
    print(f"转换后大小: {len(data) / 1024 / 1024:.2f} MB")
    return f"data:image/jpeg;base64,{data}"

def _build_content(content: str | list) -> str | list:
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
                if raw_url.startswith(('http://', 'https://')):
                    url = raw_url
                else:
                    url = _encode_local_image(raw_url)
            else:
                raise ValueError("image block 需要提供 'url' 或 'base64' 字段")
            result.append({'type': 'image_url', 'image_url': {'url': url}})
        else:
            raise ValueError(f"不支持的 content block 类型：{block['type']}")
    return result


def _normalize_messages(messages: list[dict] | list[str]) -> list[dict]:
    """
    统一消息格式：
      - list[str]：[sys_str, user_str] 简写形式
      - list[dict]：标准消息列表，content 字段可能是 str 或自定义多模态 list
    """
    if isinstance(messages[0], str):
        return [
            {'role': 'system', 'content': messages[0]},
            {'role': 'user',   'content': messages[1]},
        ]

    return [
        {**msg, 'content': _build_content(msg['content'])}
        for msg in messages
    ]

def call(
    model_name: str,
    messages: list[dict] | list[str] = None,
    enable_thinking: bool = False,
    thinking_level: int = 0,        # 0, 1, 2, 3
    enable_search: bool = False,
    tools: list[dict] = None,       # 用户提供的工具列表（标准 OpenAI tools 格式）
    output_format: str = None,      # "json" 或 None（text）
    max_output_tokens: int = None,
    temperature: float = None,
    stream: bool = False,
    logprobs: bool = False,
) -> dict:
    """
    设计迭代：
    1. 通过读取各个 api 文档来得知可以配置哪些内容
    2. 拉一个列表，把我所关心的配置写出来
    3. 把这个列表中的配置对各个 api 做适配【对于每家 api，逐项适配我的要求】
    给定（模型，消息，是否思考，思考强度，是否允许搜索，工具列表，返回格式，最大输出长度，温度，是否流式，是否返回 logits）
    返回和打log（时间戳，模型，思考内容，回答内容，logits，token花费）
    """
    
    messages = _normalize_messages(messages)

    # ── 公共参数（各家 OpenAI 兼容接口均支持）──────────────────────────────
    common = dict(
        model=model_name,
        messages=messages,
        # stream=stream,
    )
    if stream is True:
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

    # ════════════════════════════════════════════════════════════════════════
    #  各家差异化参数适配
    # ════════════════════════════════════════════════════════════════════════
    if model_name.count('/') > 0 and any(c.isupper() for c in model_name):
        # HuggingFace 本地模型
        return _call_transformers(model_name, messages, max_output_tokens, temperature)

    elif model_name.count('/') > 0:
        # ── OpenRouter ─────────────────────────────────────────────────────────
        # OpenRouter 通过标准 OpenAI 兼容接口代理多家模型
        # thinking      → extra_body={"thinking": {"type": "enabled", "budget_tokens": N}}
        # enable_search → extra_body={"plugins": [{"id": "web"}]}
        # tools         → 标准顶层参数
        extra = {}

        if enable_thinking:
            budgets = {1: 1024, 2: 8192, 3: 38912}
            extra['thinking'] = {
                'type': 'enabled',
                'budget_tokens': budgets[thinking_level],
            }

        if enable_search:
            extra['plugins'] = [{'id': 'web'}]

        response = client(
            'OPENROUTER',
            'https://openrouter.ai/api/v1'
        ).chat.completions.create(
            **common,
            tools=tools,
            extra_body=extra if extra else None,
        )
    elif model_name.startswith('qwen'):
        # ── Qwen（阿里云百炼）──────────────────────────────────────────────
        # enable_thinking / thinking_budget → extra_body（非 OpenAI 标准）
        # enable_search                     → extra_body（非 OpenAI 标准）
        # tools                             → 标准顶层参数
        extra = {}
        if enable_thinking:
            extra['enable_thinking'] = True
            extra['thinking_budget'] = _QWEN_THINKING_BUDGET[thinking_level]
        if enable_search:
            extra['enable_search'] = True

        response = client(
            'DASHSCOPE',
            'https://dashscope.aliyuncs.com/compatible-mode/v1'
        ).chat.completions.create(
            **common,
            tools=tools,
            extra_body=extra if extra else None,
            # stream_options={"include_usage": True} if stream else None,
        )

    elif model_name.startswith('doubao'):
        # ── Seed / 豆包（火山方舟）────────────────────────────────────────
        # thinking → extra_body={"thinking": {"type": "enabled"/"disabled"/"auto"}}
        # tools    → 标准顶层参数
        # enable_search: 火山方舟 Chat API 不支持联网，需走 bot 应用，此处忽略
        extra = {}
        if enable_search:
            raise ValueError('火山方舟 Chat API 不支持联网搜索功能')
        if enable_thinking:
            thinking_types = {0: 'auto', 1: 'auto', 2: 'enabled', 3: 'enabled'}
            extra['thinking'] = {'type': thinking_types[thinking_level]}
        else:
            extra['thinking'] = {'type': 'disabled'}

        response = client(
            'ARK',
            'https://ark.cn-beijing.volces.com/api/v3'
        ).chat.completions.create(
            **common,
            tools=tools,
            extra_body=extra,
        )

    elif model_name.startswith('deepseek'):
        # DeepSeek 不支持多模态输入，提前检查
        for msg in messages:
            if isinstance(msg.get('content'), list):
                raise ValueError("DeepSeek 不支持多模态输入，请使用纯文本消息")
        # ── DeepSeek ───────────────────────────────────────────────────────
        # thinking_level=0  → thinking={"type": "disabled"}，关闭思考
        # thinking_level=1,2 → reasoning_effort="high"
        # thinking_level=3   → reasoning_effort="max"
        # 注意：reasoning_effort 与 thinking type 是同一旋钮的不同写法，不能混传
        if enable_thinking:
            extra_kwargs = {
                'reasoning_effort': _DEEPSEEK_REASONING_EFFORT[thinking_level]
            }
        else:
            extra_kwargs = {
                'extra_body': {
                    'thinking': {'type': 'disabled'}
                }
            }

        response = client(
            'DEEPSEEK',
            'https://api.deepseek.com'
        ).chat.completions.create(
            **common,
            **extra_kwargs,
            tools=tools,
        )

    elif model_name.startswith('gemini'):
        # ── Gemini（Google）───────────────────────────────────────────────
        # reasoning_effort → 顶层参数 "none"/"low"/"medium"/"high"
        # enable_search    → 合并进 tools 列表，{"type": "google_search"}
        # tools（用户自定义）→ 标准顶层参数，与 google_search 合并后一起传入
        merged_tools = list(tools) if tools else []
        if enable_search:
            merged_tools.append({"type": "google_search"})

        response = client(
            'GEMINI',
            'https://generativelanguage.googleapis.com/v1beta/openai/'
        ).chat.completions.create(
            **common,
            reasoning_effort=_GEMINI_REASONING_EFFORT[thinking_level] if enable_thinking else 'none',
            tools=merged_tools if merged_tools else None,
        )

    else:
        raise ValueError(f"不支持的模型前缀：{model_name}")

    # ════════════════════════════════════════════════════════════════════════
    #  统一提取返回值
    # ════════════════════════════════════════════════════════════════════════
    choice = response.choices[0]

    logprobs_data = None
    if logprobs and choice.logprobs:
        logprobs_data = [
            {'token': t.token, 'logprob': t.logprob}
            for t in (choice.logprobs.content or [])
        ]

    retval = {
        'timestamp':   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model':       model_name,
        'answer':      choice.message.content,
        'thinking':    getattr(choice.message, 'reasoning_content', None),
        'logprobs':    logprobs_data,
        'usage': {
            'prompt_tokens':     response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens':      response.usage.total_tokens,
        } if response.usage else None,
    }

    return retval

if __name__ == '__main__':
    call()