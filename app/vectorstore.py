"""
向量库与检索模块(第 2 周:RAG)

功能:
1. 用 embedding 模型把文本块转成向量
2. 存入 Chroma 向量库(本地持久化)
3. 按语义相似度检索最相关的文本块

设计要点(面试可讲):
- Chroma 是轻量级本地向量库,适合开发/演示,生产可换 Milvus 等
- Embedding 用硅基流动的 BAAI/bge-m3,中文语义效果好
- 检索本质是"向量相似度计算"(默认余弦相似度),返回最相近的 top-k 块
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# 向量库持久化目录
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vector_store")

# embedding 相关配置
EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBEDDING_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def get_embeddings() -> OpenAIEmbeddings:
    """创建 embedding 模型实例(OpenAI 兼容接口,指向硅基流动)。"""
    if not EMBEDDING_API_KEY or EMBEDDING_API_KEY == "sk-your-api-key-here":
        raise RuntimeError(
            "未配置 SILICONFLOW_API_KEY!请在 .env 中填入硅基流动的 API Key"
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )


def create_vector_store(documents: List[Document], collection_name: str = "knowledge_base") -> Chroma:
    """
    将文档块向量化并存入 Chroma 向量库。

    参数:
        documents: 已分块的文档列表
        collection_name: 集合名(可理解为"一个知识库")
    返回:
        Chroma 向量库实例
    """
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=VECTOR_STORE_DIR,
    )
    return vector_store


def load_vector_store(collection_name: str = "knowledge_base") -> Chroma:
    """加载已持久化的向量库(只读检索)。"""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )


def search(
    vector_store: Chroma,
    query: str,
    top_k: int = 4,
) -> List[Document]:
    """
    按语义相似度检索与问题最相关的文本块。

    参数:
        vector_store: 向量库实例
        query: 用户问题
        top_k: 返回最相关的块数量
    返回:
        相关文档块列表(按相似度从高到低)
    """
    return vector_store.similarity_search(query, k=top_k)
