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
import shutil
from typing import List

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 优先 IPv4,规避部分网络环境下的 DNS/IPv6 连接问题
import app.network  # noqa: F401

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
    混合检索:语义相似度检索 + 关键词字面匹配兜底。

    背景:纯向量检索(语义)对"专业术语/缩写"类查询(如 ROE、PE、抄底)召回率偏低,
    因为这些词在向量空间里容易被同领域的泛化词稀释。这里做关键词兜底:
    提取查询中的"硬关键词"(英文缩写 + 中文专业词),把字面含这些词的块补充进结果。

    参数:
        vector_store: 向量库实例
        query: 用户问题
        top_k: 返回最相关的块数量
    返回:
        相关文档块列表(语义结果优先,关键词命中块补位)
    """
    import re

    # 1. 语义检索,先取稍多一些候选
    docs = vector_store.similarity_search(query, k=top_k)

    # 2. 提取硬关键词
    stop_chars = set("怎么什么如何是了吗呢的都在能可不有个哪些多少为什么含义意思代表指解释理解")
    hard_keywords = re.findall(r"[A-Za-z][A-Za-z0-9+./\-]{1,}", query)  # 英文缩写/术语
    # 中文关键词:去掉停用字,提取剩余连续中文串
    zh = re.sub(r"[A-Za-z0-9+./\-%:：,，。？?、\s]+", "|", query)
    for seg in zh.split("|"):
        cleaned = "".join(c for c in seg if c not in stop_chars)
        if len(cleaned) >= 2:
            hard_keywords.append(cleaned)

    # 只保留"有意义"的关键词:长度>=2,且不是纯停用词
    hard_keywords = [k for k in dict.fromkeys(hard_keywords) if len(k) >= 2]
    if not hard_keywords:
        return docs

    # 3. 全量取回(演示数据量小),把"字面含关键词"的块**优先插到前面**
    all_docs = vector_store.similarity_search(query, k=500)
    existing = [d.page_content for d in docs]
    keyword_hits = []
    for d in all_docs:
        if any(k.upper() in d.page_content.upper() for k in hard_keywords):
            if d.page_content not in existing:
                keyword_hits.append(d)

    # 关键词命中的块放最前面(它们是最可能含答案的),语义结果排后
    docs = keyword_hits + docs

    return docs[:top_k]


def reset_vector_store() -> None:
    """
    清空向量库(删除持久化目录)。

    用于"换文档"场景:先 reset,再 ingest 新文档,避免旧内容混入干扰回答。
    """
    dir_path = os.path.join(VECTOR_STORE_DIR)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"✅ 已清空向量库目录: {dir_path}")
    else:
        print("向量库目录不存在,无需清空。")
