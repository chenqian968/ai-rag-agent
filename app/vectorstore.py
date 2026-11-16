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

    # 1. 提取"硬关键词":英文缩写/术语 + 中文实义词
    #   中文没有空格分词,不能简单"去停用字后拼串"(会把一句话粘成一整串,
    #   如"小个子显高的搭配思路"变成无意义的"小子显高搭配路",导致字面匹配失效)。
    #   这里用 n-gram 切词:中文词多为 2~4 字,用二元组/三元组在连续汉字上滑动切分,
    #   切出"显高""搭配""小个""个子"等片段,其中的实义词就能字面命中对应文档。
    stop_chars = set("怎么什么如何是了吗呢的都在能可不有个哪些多少为什么含义意思代表指解释理解")
    hard_keywords = re.findall(r"[A-Za-z][A-Za-z0-9+./\-]{1,}", query)  # 英文缩写/术语

    # 提取连续中文片段,对每段做 bigram + trigram 切词
    zh_segments = re.findall(r"[\u4e00-\u9fff]+", query)
    for seg in zh_segments:
        # 先去掉纯停用字,再对剩余连续串做 n-gram 切分
        core = "".join(c for c in seg if c not in stop_chars)
        # bigram(二元组)与 trigram(三元组)切词
        for n in (2, 3):
            for i in range(len(core) - n + 1):
                gram = core[i:i + n]
                # 过滤掉仍含停用字的片段,避免"的了""的吗"这类噪音
                if any(c in stop_chars for c in gram):
                    continue
                hard_keywords.append(gram)

    # 去重、去空、保留长度>=2 的关键词
    hard_keywords = [k for k in dict.fromkeys(hard_keywords) if len(k) >= 2]

    # 2. 全量取回所有块(演示数据量小,可控),按"关键词命中数"重新排序:
    #    字面含关键词越多的块越靠前。这样对"穿搭/小个子/显高"这类主题词明确的
    #    查询,能确保命中正确的领域块,而不是被纯向量相似度里"领域间噪音"稀释掉。
    all_docs = vector_store.similarity_search(query, k=1000)

    def _hit_score(doc) -> int:
        """统计块内容里命中的关键词数量(去重)。"""
        content = doc.page_content
        return sum(1 for k in hard_keywords if k in content)

    # 命中数多的排前;命中数相同则保持语义排序(相似度降序)
    ranked = sorted(all_docs, key=_hit_score, reverse=True)

    # 3. 取前 top_k 个;若存在命中数>0 的块,优先返回它们(相关块置顶)
    return ranked[:top_k]


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
