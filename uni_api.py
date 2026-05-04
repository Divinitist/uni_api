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

def _build_content(content: str | list) -> str | list:
    """
    将自定义多模态格式转换为 OpenAI 标准 content 格式。
    输入：
      - str：纯文本，直接透传
      - list：多模态块列表，格式为
          [
            {'type': 'text',  'content': 'xxx'},
            {'type': 'image', 'url': 'xxx'},        # 网络图片 URL
            {'type': 'image', 'base64': 'xxx', 'mime': 'image/jpeg'},  # base64
          ]
    输出：OpenAI 标准 content（str 或 list of dict）
    """
    if isinstance(content, str):
        return content

    result = []
    for block in content:
        if block['type'] == 'text':
            result.append({
                'type': 'text',
                'text': block['content'],
            })
        elif block['type'] == 'image':
            if 'url' in block:
                image_url = block['url']
            elif 'base64' in block:
                mime = block.get('mime', 'image/jpeg')
                image_url = f"data:{mime};base64,{block['base64']}"
            else:
                raise ValueError("image block 需要提供 'url' 或 'base64' 字段")
            result.append({
                'type': 'image_url',
                'image_url': {'url': image_url},
            })
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
    temperature: int = None,
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
        stream=stream,
    )
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

    if model_name.startswith('qwen'):
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
            stream_options={"include_usage": True} if stream else None,
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
    print('不要执行这个文件')