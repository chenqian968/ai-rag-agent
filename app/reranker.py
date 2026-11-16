"""
Rerank 精排模块(迭代升级:两阶段检索的"精排"阶段)

背景:
RAG 检索本质是"漏斗"——召回(recall)负责"别漏",精排(rank)负责"排对"。
向量相似度检索是"双塔"模型,速度快但精度有限;Rerank 用"交叉编码"(Cross-Encoder)
把 query 和每个候选拼接后逐对打分,精度更高,但计算成本大,因此只对召回后的少量候选精排。

经典链路:
    向量粗排 Top 50 → Rerank 精排 → Top 6 → 生成

本模块调用硅基流动的 bge-reranker-v2-m3 在线 API(与 embedding 共用同一 API Key),
避免本地部署 torch + sentence_transformers 的巨大开销。

接口规格(POST https://api.siliconflow.cn/v1/rerank):
    请求: {"model": "BAAI/bge-reranker-v2-m3", "query": "...", "documents": [...], "top_n": N}
    响应: {"results": [{"index": 原数组下标, "relevance_score": 0~1 分数}, ...]},已按分数降序
"""

import os
from typing import List

import requests

from dotenv import load_dotenv

# 优先 IPv4,规避部分网络环境下的 DNS/IPv6 连接问题
import app.network  # noqa: F401

load_dotenv()

RERANK_API_KEY = os.getenv("SILICONFLOW_API_KEY")
RERANK_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def rerank(
    query: str,
    documents: List[str],
    top_n: int = 6,
) -> List[int]:
    """
    对候选文档做交叉编码精排,返回按相关性降序的原始下标列表。

    参数:
        query: 用户问题(或改写后的查询)
        documents: 候选文本块列表(来自召回阶段)
        top_n: 精排后返回前 N 个
    返回:
        按 relevance_score 降序的原始下标列表(长度 <= top_n)
    """
    if not RERANK_API_KEY or RERANK_API_KEY == "sk-your-api-key-here":
        raise RuntimeError("未配置 SILICONFLOW_API_KEY,无法调用 rerank 接口")

    if not documents:
        return []

    resp = requests.post(
        f"{RERANK_BASE_URL}/rerank",
        headers={"Authorization": f"Bearer {RERANK_API_KEY}"},
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    return [r["index"] for r in results]
