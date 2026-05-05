from uni_api import call

res = call(
    # model_name='doubao-seed-1-6-flash-250615',
    # model_name='deepseek-v4-flash',
    model_name='qwen-flash',
    messages=[
        '你是一个随机数生成器',
        '以1/2概率输出A，以1/2概率输出B，总共输出10个A或B，不要输出其他任何东西'
    ],
    # output_format='json',
    # enable_thinking=True,
    # enable_search=True,
    logprobs = True,
)
print(res)