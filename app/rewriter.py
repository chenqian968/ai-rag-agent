"""
查询改写模块(迭代升级:解决"问题本身没写清楚"的检索短板)

背景:
检索系统默认"用户问题写得很清楚",但真实提问五花八门,直接影响召回质量:
  1. 多轮对话里的"半句话/指代"——"那其他城市呢?",直接拿去检索天知道查什么;
  2. 口语化/模糊表达——"这玩意咋弄的?",缺少利于检索的关键词;
  3. 单一查询角度——一个 query 可能覆盖不全,多角度检索更全。

本模块用 DeepSeek 把口语化/指代性问题改写成"独立、完整、无指代"的检索查询,
并可生成多个角度的查询(Multi-Query)分别检索后合并,提升召回率。

设计要点(面试可讲):
- 查询改写是"检索和生成之间的一层 query 理解",是高级 RAG 的标志;
- 多轮对话里指代消解(如"它"指代上文)是查询改写的核心场景;
- Multi-Query 通过"多个检索角度取并集"提升召回,以少量 LLM 调用换召回率。
"""

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# 优先 IPv4,规避部分网络环境下的 DNS/IPv6 连接问题
import app.network  # noqa: F401

load_dotenv()

REWRITE_API_KEY = os.getenv("DEEPSEEK_API_KEY")
REWRITE_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
REWRITE_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")


def _get_client() -> OpenAI:
    if not REWRITE_API_KEY or REWRITE_API_KEY == "sk-your-api-key-here":
        raise RuntimeError("未配置 DEEPSEEK_API_KEY,无法调用查询改写")
    return OpenAI(api_key=REWRITE_API_KEY, base_url=REWRITE_BASE_URL)


def rewrite_query(query: str, history: List[str] = None) -> str:
    """
    结合对话历史,把用户问题改写成独立、完整、无指代的检索查询。

    参数:
        query: 用户当前问题
        history: 历史对话文本列表(如 ["用户: 一线城市住宿标准?", "助手: 每晚不超过500元"])
    返回:
        改写后的检索查询
    """
    client = _get_client()

    history_text = "\n".join(history) if history else "（无历史对话）"

    prompt = f"""你是查询改写助手。请结合对话历史,把用户的最后一个问题改写成
一个独立、完整、无指代的检索查询,便于在知识库中检索。

【规则】
1. 把指代词(如"它""这""那""其他")还原成具体对象;
2. 补全省略的上下文,使查询脱离历史也能被理解;
3. 保留原问题的核心意图和专业术语,不要添加无关信息;
4. 只输出改写后的查询本身,不要任何解释、引号或前缀。

【对话历史】
{history_text}

【用户当前问题】
{query}

【改写后的检索查询】"""

    resp = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 改写需稳定,低温度
    )
    return resp.choices[0].message.content.strip()


def multi_query(query: str, num_queries: int = 3) -> List[str]:
    """
    生成多个角度的检索查询(Multi-Query),用于提升召回。

    参数:
        query: 用户问题(建议先经过 rewrite_query 处理)
        num_queries: 生成的查询数量(含原问题)
    返回:
        查询列表(原问题 + 生成的多个角度查询)
    """
    client = _get_client()

    prompt = f"""你是检索查询生成助手。请为下面的问题生成 {num_queries - 1} 个
不同角度的检索查询,用于多路检索提升召回率。

【规则】
1. 每个查询从不同角度表达同一意图(如同义词替换、换表述、聚焦不同侧面);
2. 每个查询都应独立、完整、可检索;
3. 每行输出一个查询,不要编号、不要引号、不要其他解释。

【原问题】
{query}

【多个角度的检索查询】"""

    resp = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,  # 多角度需一定多样性
    )

    lines = [l.strip(" -•0123456789.") for l in resp.choices[0].message.content.strip().splitlines()]
    queries = [q for q in lines if q]
    # 原问题放最前,保证至少有一个查询
    return [query] + queries[: num_queries - 1]
