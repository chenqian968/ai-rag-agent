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


def _extract_keywords(query: str) -> List[str]:
    """
    提取查询中的"硬关键词":英文缩写/术语 + 中文实义词。

    中文没有空格分词,不能简单"去停用字后拼串"(会把一句话粘成一整串,
    如"小个子显高的搭配思路"变成无意义的"小子显高搭配路",导致字面匹配失效)。
    这里用 n-gram 切词:中文词多为 2~4 字,用二元组/三元组在连续汉字上滑动切分,
    切出"显高""搭配""小个""个子"等片段,其中的实义词就能字面命中对应文档。
    """
    import re

    stop_chars = set("怎么什么如何是了吗呢的都在能可不有个哪些多少为什么含义意思代表指解释理解")
    hard_keywords = re.findall(r"[A-Za-z][A-Za-z0-9+./\-]{1,}", query)  # 英文缩写/术语

    # 提取连续中文片段,对每段做 bigram + trigram 切词
    zh_segments = re.findall(r"[\u4e00-\u9fff]+", query)
    for seg in zh_segments:
        # 先去掉纯停用字,再对剩余连续串做 n-gram 切分
        core = "".join(c for c in seg if c not in stop_chars)
        for n in (2, 3):
            for i in range(len(core) - n + 1):
                gram = core[i:i + n]
                if any(c in stop_chars for c in gram):
                    continue
                hard_keywords.append(gram)

    return [k for k in dict.fromkeys(hard_keywords) if len(k) >= 2]


def _recall(vector_store: Chroma, query: str, recall_k: int) -> List[Document]:
    """
    召回阶段:混合检索(语义 + 关键词兜底),返回 recall_k 个候选。

    先用关键词命中数做一次粗排,确保主题词明确的查询能命中正确领域块
    (而非被多领域混库时的向量噪音稀释),再交由 rerank 精排。
    """
    hard_keywords = _extract_keywords(query)

    # 全量取回(演示数据量小,可控),按关键词命中数粗排
    all_docs = vector_store.similarity_search(query, k=1000)

    def _hit_score(doc) -> int:
        return sum(1 for k in hard_keywords if k in doc.page_content)

    ranked = sorted(all_docs, key=_hit_score, reverse=True)
    return ranked[:recall_k]


def search(
    vector_store: Chroma,
    query: str,
    top_k: int = 4,
    recall_k: int = 20,
    use_rerank: bool = True,
) -> List[Document]:
    """
    两阶段检索:召回(recall) + 精排(rerank)。

    阶段一 召回:混合检索取 recall_k 个候选(语义相似度 + 关键词兜底)。
    阶段二 精排:调用 rerank 模型对候选做交叉编码打分,取 top_k 个最相关块。

    参数:
        vector_store: 向量库实例
        query: 用户问题(或改写后的查询)
        top_k: 精排后最终返回的块数量
        recall_k: 召回阶段返回的候选数量
        use_rerank: 是否启用 rerank(关闭则退回"关键词粗排直接截断")
    返回:
        相关文档块列表(按相关性降序)
    """
    # 阶段一:召回
    candidates = _recall(vector_store, query, recall_k)

    # 阶段二:精排
    if use_rerank and len(candidates) > top_k:
        try:
            from app.reranker import rerank

            texts = [d.page_content for d in candidates]
            indices = rerank(query, texts, top_n=top_k)
            return [candidates[i] for i in indices]
        except Exception:
            # rerank 失败(如未配置 key / 网络异常)时降级为关键词粗排截断
            pass

    return candidates[:top_k]


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
