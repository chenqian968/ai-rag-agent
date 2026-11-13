"""
图形界面(Streamlit):文件上传 + 知识库问答

功能:
1. 网页端拖拽/选择上传 PDF/Word/Markdown/TXT 文档
2. 一键导入知识库(自动分块 + 向量化)
3. 网页端对话问答(带引用)
4. 清空知识库

运行方式(在项目根目录):
    streamlit run app/ui.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from app.loader import load_documents, split_documents
from app.vectorstore import (
    create_vector_store,
    load_vector_store,
    search,
    reset_vector_store,
)
from app.rag import get_llm_client, build_prompt, COLLECTION_NAME, TOP_K

# 页面基础配置
st.set_page_config(page_title="AI 知识库问答助手", page_icon="📚", layout="wide")

# 上传文件的临时保存目录
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 允许上传的文件类型
ALLOWED_TYPES = [".pdf", ".docx", ".md", ".txt"]

# 会话状态:保存对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []


def ingest_uploaded_file(uploaded_file) -> None:
    """处理上传的文件:保存到本地 -> 分块 -> 向量化入库。"""
    # 检查文件类型
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_TYPES:
        st.error(f"不支持的文件类型 '{suffix}',请上传 PDF/Word/Markdown/TXT")
        return

    # 保存上传的文件到本地
    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 加载 + 分块
    with st.spinner(f"正在解析并分块 '{uploaded_file.name}'..."):
        docs = load_documents(str(save_path))
        chunks = split_documents(docs)

    # 向量化入库
    with st.spinner(f"正在向量化 {len(chunks)} 个文本块(调用 embedding 接口)..."):
        create_vector_store(chunks, collection_name=COLLECTION_NAME)

    st.success(f"✅ 已导入 '{uploaded_file.name}',共 {len(chunks)} 个文本块")


def answer_question(query: str, vector_store) -> str:
    """检索 + 生成,返回完整回答文本。"""
    docs = search(vector_store, query, top_k=TOP_K)
    if not docs:
        return "未检索到相关内容,请先在左侧导入文档。"
    client = get_llm_client()
    prompt = build_prompt(query, docs)
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=False,
    )
    return resp.choices[0].message.content


# ============ 侧边栏:文档管理 ============
with st.sidebar:
    st.header("📁 文档管理")

    # 上传文件
    uploaded_files = st.file_uploader(
        "上传文档(可多选)",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        help="支持 PDF / Word / Markdown / TXT",
    )

    if uploaded_files:
        if st.button("导入到知识库", type="primary", use_container_width=True):
            for f in uploaded_files:
                ingest_uploaded_file(f)

    st.divider()

    # 清空知识库
    if st.button("清空知识库", use_container_width=True):
        reset_vector_store()
        st.success("知识库已清空")
        st.rerun()

    st.caption("提示:清空后需重新导入文档才能问答")


# ============ 主界面:对话 ============
st.title("📚 AI 知识库问答助手")
st.caption("基于 RAG 的垂直领域智能问答 —— 上传文档,即可就文档内容提问")

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的问题,例如:什么是市盈率?"):
    # 记录并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                vector_store = load_vector_store(COLLECTION_NAME)
                answer = answer_question(prompt, vector_store)
            except Exception as e:  # noqa: BLE001
                answer = f"⚠️ 出错了:{e}\n\n请确认已导入文档,且 .env 配置正确。"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
