from uni_api import call

res = call(
    model_name='seed-1.6-flash',
    messages=[
        '你是一个秘书',
        '告诉我新加坡今天的天气'
    ],
    enable_thinking=False,
    enable_search=True,
    logprobs=True
)
print(res)