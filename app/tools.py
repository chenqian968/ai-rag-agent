"""
Agent 工具定义(第 3 周:Agent 化)

给 Agent 提供一组"工具",让它通过 Function Calling 自主决定调用哪个。
每个工具都是一个函数 + 一段清晰的描述,大模型根据描述判断何时调用。

设计要点(面试可讲):
- 工具即"能力边界":Agent 只能做我们允许它做的事
- 检索工具(search_knowledge)让 Agent 能查知识库
- 计算器(calculate)演示"非检索类工具",体现 Agent 的多工具调度能力
"""

from typing import List

from langchain_core.tools import tool

from app.vectorstore import load_vector_store, search
from app.rag import COLLECTION_NAME, TOP_K


# 全局向量库实例(懒加载,避免每次调用都重新初始化)
_vector_store = None


def _get_vector_store():
    """懒加载向量库实例(单例)。"""
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store(COLLECTION_NAME)
    return _vector_store


@tool
def search_knowledge(query: str) -> str:
    """
    检索知识库,获取与问题相关的文本内容。
    当你需要回答关于知识库内容的问题时(比如股票、投资、财务指标等),调用此工具。
    参数 query 应该是用户问题的核心关键词或完整的检索语句。

    参数:
        query: 要检索的问题或关键词
    返回:
        检索到的相关文本块(带来源标注)
    """
    vector_store = _get_vector_store()
    docs = search(vector_store, query, top_k=TOP_K)
    if not docs:
        return "未检索到相关内容。"

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[资料 {i} | 来源: {source}]\n{doc.page_content}")

    return "\n\n".join(parts)


@tool
def calculate(expression: str) -> str:
    """
    执行数学计算。当用户问涉及具体数值计算的问题时(比如"100万本金年化5%一年多少利息"),
    调用此工具进行精确计算。

    参数:
        expression: 数学表达式,如 "1000000 * 0.05"
    返回:
        计算结果
    """
    try:
        # 用受限的 eval 做简单四则运算(演示用;生产环境应使用更安全的解析器)
        # 只允许数字、运算符、括号和空格,杜绝任意代码执行
        import re
        if not re.fullmatch(r"[0-9+\-*/().%\s]+", expression):
            return "表达式包含非法字符,只支持数字和 + - * / ( ) % 运算。"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


# 所有可用工具的列表(Agent 会拿到这个列表,根据描述选择调用)
ALL_TOOLS: List = [search_knowledge, calculate]
