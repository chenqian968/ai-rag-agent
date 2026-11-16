"""
Agent 工具定义(第 3 周:Agent 化 + 迭代升级)

给 Agent 提供一组"工具",让它通过 Function Calling 自主决定调用哪个。
每个工具都是一个函数 + 一段清晰的描述,大模型根据描述判断何时调用。

设计要点(面试可讲):
- 工具即"能力边界":Agent 只能做我们允许它做的事
- 检索工具(search_knowledge)让 Agent 能查知识库
- 计算器(calculate)演示"非检索类工具",体现 Agent 的多工具调度能力

迭代升级(检索链路增强):
- search_knowledge 内置"查询改写 + 多查询检索 + rerank 精排"三段式检索:
  1. 结合多轮历史把口语化/指代性问题改写成完整查询;
  2. Multi-Query 生成多角度查询,分别检索后合并去重;
  3. rerank 对合并候选精排,取 top_k。
"""

from typing import List, Optional

from langchain_core.tools import tool

from app.vectorstore import load_vector_store, search
from app.rag import COLLECTION_NAME, TOP_K


# 全局向量库实例(懒加载,避免每次调用都重新初始化)
_vector_store = None

# 模块级"多轮历史上下文":Agent 运行前由 run_agent 注入,
# 供 search_knowledge 做查询改写时还原指代(如"它"指代上文)。
# 用模块级变量是因为 LangChain @tool 只能接收 LLM 传入的参数,拿不到对话历史。
_current_history: Optional[List[str]] = None


def set_history(history: Optional[List[str]]) -> None:
    """设置当前对话历史(由 agent.run_agent 在调用前注入)。"""
    global _current_history
    _current_history = history


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
    当你需要回答关于知识库内容的问题时,调用此工具。
    注意:知识库的主题是动态变化的(以用户当前实际导入的文档为准,可能是大数据、
    编程、金融等任何领域),不要预设固定主题。只要用户的问题像是在询问"某份资料/
    某个知识点"的内容,就应该先调用本工具检索,而不是凭记忆直接回答。
    参数 query 应该是用户问题的核心关键词或完整的检索语句。

    参数:
        query: 要检索的问题或关键词
    返回:
        检索到的相关文本块(带来源标注)
    """
    vector_store = _get_vector_store()

    # 1. 查询改写(结合多轮历史,还原指代)
    queries = [query]
    try:
        from app.rewriter import rewrite_query, multi_query
        rewritten = rewrite_query(query, history=_current_history)
        # 2. 多查询扩展(原问题 + 多角度查询)
        queries = multi_query(rewritten)
    except Exception:
        # 改写失败(如未配置 key)时降级为直接用原问题检索
        queries = [query]

    # 3. 多查询检索 + 合并去重
    merged: dict = {}  # 用 page_content 去重
    for q in queries:
        for doc in search(vector_store, q, top_k=TOP_K):
            if doc.page_content not in merged:
                merged[doc.page_content] = doc

    # 4. 对合并候选做 rerank 精排(以改写后的主查询为基准)
    docs = list(merged.values())
    try:
        from app.reranker import rerank
        texts = [d.page_content for d in docs]
        indices = rerank(queries[0], texts, top_n=TOP_K)
        docs = [docs[i] for i in indices]
    except Exception:
        docs = docs[:TOP_K]

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
