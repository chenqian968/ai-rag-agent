"""
RAG 知识库问答命令行(第 2 周:核心交付物)

流程:
1. 导入文档:把本地文档切块、向量化、存入 Chroma
2. 问答:提问 -> 检索相关块 -> 拼进 Prompt -> 大模型基于检索内容作答(带引用)

运行方式(在项目根目录):
    # 第一次:先导入文档建立知识库
    python -m app.rag ingest docs/你的文档.pdf

    # 然后问答
    python -m app.rag chat
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.loader import load_documents, load_documents_from_dir, split_documents
from app.vectorstore import create_vector_store, load_vector_store, search

load_dotenv()

# LLM 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

COLLECTION_NAME = "knowledge_base"

# 检索相关块的数量
TOP_K = 4


def get_llm_client() -> OpenAI:
    """创建 DeepSeek 客户端。"""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-api-key-here":
        raise RuntimeError("未配置 DEEPSEEK_API_KEY!请在 .env 中填入")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def build_prompt(query: str, docs) -> str:
    """
    构建带上下文的 Prompt。
    核心思想:把检索到的相关片段作为"参考资料"拼给模型,并要求模型只依据资料作答,
    从而抑制幻觉、并让回答可溯源。
    """
    # 拼接检索结果,带上来源
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        context_parts.append(f"[资料 {i} | 来源: {source}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt = f"""你是一个专业的知识库问答助手。请根据下面提供的资料回答用户问题。

【规则】
1. 只依据【参考资料】回答,不要使用资料之外的知识编造内容。
2. 如果资料中没有相关信息,请明确回答"根据现有资料无法回答"。
3. 回答要准确、简洁,并在句末用 [资料1]、[资料2] 标注引用来源。

【参考资料】
{context}

【用户问题】
{query}

【回答】"""

    return prompt


def answer(client: OpenAI, query: str, docs) -> str:
    """基于检索结果调用大模型生成回答(流式)。"""
    prompt = build_prompt(query, docs)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # 知识问答用较低温度,更稳定
        stream=True,
    )
    parts = []
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            print(delta, end="", flush=True)
    print()
    return "".join(parts)


def cmd_ingest(path: str) -> None:
    """导入文档,建立知识库。"""
    print(f"正在导入文档: {path} ...")

    # 加载 + 分块
    p = Path(path)
    docs = load_documents(path) if p.is_file() else load_documents_from_dir(path)
    chunks = split_documents(docs)
    print(f"已加载 {len(docs)} 个文档片段,切分为 {len(chunks)} 个文本块")

    # 向量化入库
    print("正在向量化并写入向量库(首次会调用 embedding 接口,可能需要一点时间)...")
    create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print("✅ 知识库建立完成!")


def cmd_chat() -> None:
    """进入交互式问答。"""
    client = get_llm_client()
    print("正在加载向量库...")
    vector_store = load_vector_store(COLLECTION_NAME)
    print("✅ 知识库问答已启动(输入 exit 退出)")
    print("-" * 50)

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

        # 检索相关块
        docs = search(vector_store, query, top_k=TOP_K)
        if not docs:
            print("助手: 未检索到相关内容。")
        else:
            print("助手: ", end="", flush=True)
            answer(client, query, docs)

        print("-" * 50)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法:")
        print("  python -m app.rag ingest <文件或目录路径>   # 导入文档建立知识库")
        print("  python -m app.rag chat                     # 交互式问答")
        sys.exit(1)

    command = sys.argv[1]
    if command == "ingest":
        if len(sys.argv) < 3:
            print("请提供要导入的文件或目录路径")
            sys.exit(1)
        cmd_ingest(sys.argv[2])
    elif command == "chat":
        cmd_chat()
    else:
        print(f"未知命令: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
