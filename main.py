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