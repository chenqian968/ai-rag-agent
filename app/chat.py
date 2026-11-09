"""
命令行聊天机器人(第 1 周:最小验证)

功能:
1. 通过 OpenAI 兼容接口调用 DeepSeek 大模型
2. 支持流式(streaming)输出,逐字打印
3. 从 .env 读取 API Key,避免硬编码
4. 简单的异常处理与重试

运行方式(在项目根目录):
    python app/chat.py

依赖见 requirements.txt。首次运行前请:
    1. 复制 .env.example 为 .env
    2. 填入你的 DEEPSEEK_API_KEY
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# 系统提示词,定义助手的角色
SYSTEM_PROMPT = "你是一个乐于助人的 AI 助手,用简洁、准确的中文回答问题。"


def check_config() -> None:
    """检查配置是否就绪,缺少 Key 时给出明确提示。"""
    if not API_KEY or API_KEY == "sk-your-api-key-here":
        print("=" * 50)
        print("❌ 尚未配置 API Key!")
        print("请按以下步骤操作:")
        print("  1. 复制 .env.example 为 .env")
        print("     (Windows: copy .env.example .env)")
        print("  2. 打开 .env,把 DEEPSEEK_API_KEY 改成你的真实 Key")
        print("  3. Key 申请地址:https://platform.deepseek.com/")
        print("=" * 50)
        sys.exit(1)


def stream_chat(client: OpenAI, messages: list) -> str:
    """
    以流式方式请求模型,逐字打印回复,并返回完整文本。

    参数:
        client: OpenAI 客户端实例
        messages: 对话历史(含 role 和 content 的字典列表)
    返回:
        模型完整回复文本
    """
    # 将系统提示词放在最前面
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    response = client.chat.completions.create(
        model=MODEL,
        messages=full_messages,
        temperature=TEMPERATURE,
        stream=True,  # 开启流式
    )

    answer_parts = []
    print("助手: ", end="", flush=True)
    for chunk in response:
        # 流式响应中每个 chunk 的 delta.content 是增量文本
        delta = chunk.choices[0].delta.content
        if delta:
            answer_parts.append(delta)
            print(delta, end="", flush=True)
    print()  # 换行
    return "".join(answer_parts)


def main() -> None:
    check_config()

    # 初始化客户端(OpenAI 兼容接口,指向 DeepSeek)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 对话历史:保存多轮上下文(不含 system,由 stream_chat 自动加上)
    history: list = []

    print("🤖 命令行聊天机器人已启动(输入 exit / quit 退出)")
    print("-" * 50)

    while True:
        try:
            user_input = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见!")
            break

        # 把用户消息加入历史
        history.append({"role": "user", "content": user_input})

        # 带重试的调用
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                answer = stream_chat(client, history)
                # 把助手回复也加入历史,保持多轮记忆
                history.append({"role": "assistant", "content": answer})
                break
            except Exception as e:  # noqa: BLE001
                print(f"\n⚠️ 调用失败(第 {attempt} 次):{e}")
                if attempt == max_retries:
                    print("❌ 重试次数用尽,请检查网络或 API Key。")
                    # 回滚本次用户消息,避免污染历史
                    history.pop()
                else:
                    time.sleep(2)  # 简单退避,避免立即重试

        print("-" * 50)


if __name__ == "__main__":
    main()
