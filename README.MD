#### Usage

这些模型有很多乱七八糟的限制，有的能用 json format | thinking | search 有的不能用，看着办吧至少把形式统一了，用不了再调整模型和参数配置。

```python
from uni_api import call

res = call(
    # model_name='doubao-seed-1-6-flash-250615',
    # model_name='deepseek-v4-flash',
    model_name='qwen-flash',
    messages=[
        '你是一个秘书',
        '搜索并告诉我新加坡今天的天气，并以json形式输出'
    ],
    output_format='json',
    # enable_thinking=True,
    # enable_search=True,
)
print(res)
```