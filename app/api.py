"""
FastAPI 后端(第 4 周:产品化)

把前 3 周的能力封装成 HTTP API,供前端调用:
- POST /api/chat       Agent 对话(带多轮历史)
- POST /api/ingest     上传文档建库
- POST /api/reset      清空知识库
- GET  /               健康检查

设计要点(面试可讲):
- FastAPI 天然支持 Pydantic 数据校验 + 自动生成 OpenAPI 文档(/docs)
- 用"会话级历史"实现多轮记忆:前端把历史消息随请求一起传回
- 无状态 API:每次请求独立,便于横向扩展
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 优先 IPv4,规避部分网络环境下的 DNS/IPv6 连接问题
import app.network  # noqa: F401

from app.agent import run_agent
from app.loader import load_documents, split_documents
from app.vectorstore import create_vector_store, reset_vector_store

load_dotenv()

app = FastAPI(
    title="AI 知识库问答 Agent",
    description="基于 RAG + ReAct Agent 的智能问答服务",
    version="1.0.0",
)

# 允许跨域(前端本地开发时端口不同)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    """单条对话消息(用于多轮记忆的历史传递)。"""
    role: str          # "user" 或 "assistant"
    content: str


class ChatRequest(BaseModel):
    """对话请求体。"""
    message: str                       # 本轮用户输入
    history: List[ChatMessage] = []    # 历史对话(可选)


class ChatResponse(BaseModel):
    """对话响应体。"""
    answer: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """健康检查。"""
    return {"status": "ok", "service": "ai-rag-agent"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Agent 对话接口(带多轮历史)。"""
    try:
        # 把前端传来的历史转成 LangChain 消息格式
        from langchain_core.messages import HumanMessage, AIMessage

        history = []
        for m in req.history:
            if m.role == "user":
                history.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                history.append(AIMessage(content=m.content))

        answer = run_agent(req.message, history=history)
        return ChatResponse(answer=answer)
    except Exception as e:
        return ChatResponse(answer="", error=str(e))


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    """上传文档并导入知识库。"""
    # 保存上传的文件到临时目录
    suffix = Path(file.filename).suffix or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 加载 + 分块 + 向量化入库
        documents = load_documents(tmp_path)
        chunks = split_documents(documents)
        # 把 source 元数据从"临时文件名"修正为用户上传的真实文件名,便于溯源展示
        real_name = file.filename
        for c in chunks:
            c.metadata["source"] = real_name
        create_vector_store(chunks)
        return {
            "status": "ok",
            "filename": real_name,
            "chunks": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"导入失败: {e}")
    finally:
        # 清理临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/api/reset")
def reset():
    """清空知识库。"""
    try:
        reset_vector_store()
        return {"status": "ok", "message": "知识库已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 前端静态资源(第 4 周:网页界面)
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    # 挂载静态文件(css/js)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

    @app.get("/")
    def index():
        """返回前端页面。"""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
