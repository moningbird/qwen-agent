"""A TIR(tool-integrated reasoning) math agent
```bash
python tir_math.py
```
"""
import os
import json
from pprint import pprint

from qwen_agent.agents import TIRMathAgent
from qwen_agent.gui import WebUI

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# We use the following two systems to distinguish between COT mode and TIR mode
TIR_SYSTEM = """f
    请根据以下数学题和对应的解析，调用代码生成工具来为题目生成代码解析过程，并执行代码给出最终的结果。

    数学题:
    {instruction}
    解析：
    {output}

    要求:
    1. 调用代码并解释解题过程。
    2. 执行代码并给出最终的答案。"""
COT_SYSTEM = """Please reason step by step, and put your final answer within \\boxed{}."""


def init_agent_service():
    # Use this to access the qwen2.5-math model deployed on dashscope
    llm_cfg = {#'model': 'qwen2.5-math-72b-instruct',
               #'model_type': 'qwen_dashscope',
               #'generate_cfg': {'top_k': 1}
               # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
                'model': 'qwen2.5:72b',
                'model_server': 'https://api.adamchatbot.chat/v1',  # base_url，也称为 api_base
                'api_key': 'sk-LUv33qaHFRd1EJSm64C0999d46C746A5A437E8F9610d31E4',
               }
    bot = TIRMathAgent(llm=llm_cfg, name='qwen2.5:72b', system_message=TIR_SYSTEM)
    return bot


def test(query: str = '你好'):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = [{'role': 'user', 'content': query}]
    # print(bot.run(messages))
    # for response in bot.run(messages):
    #     pprint(response, indent=2)


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        # Query example: 斐波那契数列前10个数字
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [
            r'曲线 $y=2 \\ln (x+1)$ 在点 $(0,0)$ 处的切线方程为 $( )$.',
            '斐波那契数列前10个数字',
            'A digital display shows the current date as an $8$-digit integer consisting of a $4$-digit year, '
            'followed by a $2$-digit month, followed by a $2$-digit date within the month. '
            'For example, Arbor Day this year is displayed as 20230428. '
            'For how many dates in $2023$ will each digit appear an even number of times '
            'in the 8-digital display for that date?'
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()

import json

def process_dataset(dataset_path: str, output_path: str):
    """
    处理包含多个数学题的 JSON 数据集，并将结果整理成指定的 JSON 格式并保存到文件中。

    :param dataset_path: examples/resource/yingyongti_test.json
    :param output_path: examples/resource/tir_math.json
    """
    bot = init_agent_service()

    # 读取数据集文件
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    results = []

    # 处理每个问题
    for idx, item in enumerate(dataset):
        question = item.get("instruction", "").strip()
        if not question:
            continue

        print(f"Processing question {idx + 1}: {question}")
        messages = [{'role': 'user', 'content': question}]

        # 初始化模型输出变量
        model_output = ""

        # 获取模型的完整回答
        response_list = []
        for response in bot.run(messages):
            pprint(response, indent=2)
            response_list.append(response)
        # print(response_list)  # 打印完整的响应

        # 提取最后一次输出的 content
        if isinstance(response_list, list) and response_list:
            last_response = response_list[-1]

            if last_response:
                model_output = last_response[0].get('content', '')
                print(model_output)

        # 整理成指定的 JSON 格式
        result = {
            "instruction": question,
            "input": "",
            "output": model_output
        }
        results.append(result)
        # 保存结果到文件
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)


    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    # 处理数据集并输出结果
     dataset_path = os.path.join(ROOT_RESOURCE, 'yingyongti_math.json')
     output_path = os.path.join(ROOT_RESOURCE, 'tir_math.json')
     process_dataset(dataset_path, output_path)

    # test()
    # app_tui()
    # app_gui()
