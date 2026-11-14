"""
Agent 交互式命令行入口(第 3 周:Agent 化交付物)

整合 ReAct Agent + 工具调用 + 多轮记忆,提供交互式对话。
相比第 2 周的 rag.py(固定"检索→生成"),这里的 Agent 会自主决策。

运行方式(在项目根目录):
    python -m app.agent_cli
"""

import os
import sys

from dotenv import load_dotenv

# 优先 IPv4,规避部分网络环境下的 DNS/IPv6 连接问题
import app.network  # noqa: F401

from langchain_core.messages import HumanMessage, AIMessage

from app.agent import run_agent

load_dotenv()


def main() -> None:
    """交互式 Agent 对话,支持多轮记忆。"""
    print("正在初始化 Agent...")
    print("✅ ReAct Agent 已启动(具备知识库检索 + 计算能力,输入 exit 退出)")
    print("-" * 50)

    # 维护多轮对话历史:每轮把 (用户问题, 助手回答) 追加进去
    history = []

    while True:
        try:
            query = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "退出"):
            print("再见!")
            break

        # 运行 Agent(带上历史,实现多轮记忆)
        answer = run_agent(query, history=history)
        print("助手:", answer)

        # 更新历史
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))

        print("-" * 50)


if __name__ == "__main__":
    main()
