"""
ReAct Agent(第 3 周:Agent 化核心)

用 LangGraph 实现 ReAct(Reasoning + Acting)循环,让 Agent 能自主决策:

    ┌──────────────┐    有工具调用    ┌──────────────┐
    │  agent 思考   │ ──────────────→ │  执行工具     │
    │ (调用 LLM)    │                 │ (观察结果)    │
    └──────────────┘                 └──────┬───────┘
           ↑                                │
           │        没有工具调用            │
           └────────────────────────────────┘
                          ↓
                    输出最终回答

核心思想:Agent 不再"固定地检索→生成",而是自己决定"该不该查知识库、查几次、
要不要算个数、还是直接回答"。这就是从"流水线"到"智能体"的升级。

设计要点(面试可讲):
- LangGraph 用"图"来编排 Agent 的决策流程,节点=动作,边=流转条件
- 相比手写 while 循环,LangGraph 的好处是可观测、可扩展、支持状态持久化
- 这里用的是 DeepSeek 的 Function Calling(工具调用)能力
"""

from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.rag import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY
from app.tools import ALL_TOOLS


# 用 ChatOpenAI 封装 DeepSeek(OpenAI 兼容),并绑定工具。
# bind_tools 会自动把 LangChain 工具转成 Function Calling 的 JSON Schema,
# 并处理消息的序列化与工具调用结果回传,避免手写 OpenAI dict 带来的兼容性问题。
_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
).bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# 1. 定义 Agent 的状态(State):贯穿整个循环的共享数据
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """Agent 的共享状态。

    messages: 对话消息列表(含系统提示、用户输入、AI 思考、工具调用与结果)。
              用 add_messages 这个 reducer 自动做消息追加与合并。
    """
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 2. 系统提示词:定义 Agent 的"人设"和行为准则
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个专业的智能助手,具备以下能力:
1. 检索知识库(search_knowledge 工具):回答与知识库内容相关的问题
2. 数学计算(calculate 工具):执行精确的数值计算

你的工作方式(ReAct):
- 先判断:用户的问题是否需要调用工具?
  * 涉及知识库内容的问题 → 调用 search_knowledge(知识库主题不固定,以当前实际导入的文档为准,不要预设某个领域)
  * 涉及数值计算 → 调用 calculate
  * 简单闲聊或不需要工具的问题 → 直接回答
- 如果需要工具,调用它,拿到结果后再组织回答
- 回答要准确、简洁;基于工具返回的资料作答时,标注 [资料1]、[资料2] 引用来源
- 如果工具返回"未检索到相关内容",诚实告知用户"根据现有资料无法回答",不要编造

注意:你可以根据情况多次调用工具,也可以不调用工具直接回答。"""


# ---------------------------------------------------------------------------
# 3. 核心节点:调用 LLM,让它决定"下一步做什么"
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """调用大模型(已绑定工具),让模型自主决定:调用工具 or 直接回答。

    这是 ReAct 的 "Reasoning"(推理)环节——模型输出一个决定。
    ChatOpenAI + bind_tools 会自动处理 Function Calling 的序列化,
    模型返回的 AIMessage 要么带 tool_calls(要调工具),要么是普通文本(直接回答)。
    """
    # 组装消息:系统提示 + 历史对话 + 用户输入
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # 调用绑定了工具的 LLM,让模型自主决策
    ai_message = _llm.invoke(messages)

    return {"messages": [ai_message]}


# ---------------------------------------------------------------------------
# 4. 工具执行节点:真正调用工具,拿到结果(ReAct 的 "Acting" 环节)
# ---------------------------------------------------------------------------
def tool_node(state: AgentState) -> dict:
    """执行模型请求的工具调用,并把结果作为 ToolMessage 返回。

    这是 ReAct 的 "Acting"(行动)环节,也是 "Observation"(观察)——
    工具执行结果会作为新的消息喂回给模型,让它继续推理。
    """
    # 取最后一条 AI 消息里的工具调用请求
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    results = []
    tool_map = {t.name: t for t in ALL_TOOLS}

    for tc in tool_calls:
        # tool_calls 的每一项可能是 dict,也可能是对象,统一用 get 兼容
        if isinstance(tc, dict):
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id")
        else:
            tool_name = getattr(tc, "name", None)
            tool_args = getattr(tc, "args", {})
            tool_id = getattr(tc, "id", None)

        # 找到对应的工具并执行
        if tool_name in tool_map:
            try:
                # LangChain 的 tool 可以直接调用(invoke)
                output = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                output = f"工具执行出错: {e}"
        else:
            output = f"未知工具: {tool_name}"

        results.append(ToolMessage(content=str(output), tool_call_id=tool_id, name=tool_name))

    return {"messages": results}


# ---------------------------------------------------------------------------
# 5. 路由函数:判断下一步是"继续调工具"还是"结束"
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """根据最后一条消息决定流转方向。

    - 如果 AI 消息里有 tool_calls → 去执行工具(循环继续)
    - 否则 → 结束,输出最终回答
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# 6. 组装图:把节点和边连起来
# ---------------------------------------------------------------------------
def build_agent_graph():
    """构建 ReAct Agent 的状态图。"""
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # 设置入口
    graph.set_entry_point("agent")

    # 边:agent 节点结束后,根据条件路由
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",   # 需要调工具 → 去 tools 节点
            "end": END,         # 直接回答 → 结束
        },
    )

    # tools 执行完 → 回到 agent 节点(让模型基于工具结果继续推理)
    graph.add_edge("tools", "agent")

    return graph.compile()


# 编译好的图(供外部调用)
AGENT_GRAPH = build_agent_graph()


# ---------------------------------------------------------------------------
# 7. 对外接口:跑一轮 Agent(带多轮记忆)
# ---------------------------------------------------------------------------
def run_agent(user_input: str, history: List[BaseMessage] = None) -> str:
    """运行 Agent,处理一次用户输入,返回最终回答。

    参数:
        user_input: 用户本轮输入
        history: 历史消息列表(用于多轮记忆,让 Agent 记住上文)
    返回:
        Agent 的最终回答文本
    """
    messages = list(history or []) + [HumanMessage(content=user_input)]

    # 执行图,直到结束
    final_state = AGENT_GRAPH.invoke({"messages": messages})

    # 最后一条消息即最终回答
    last_message = final_state["messages"][-1]
    return last_message.content
