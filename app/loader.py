"""
文档加载与分块模块(第 2 周:RAG)

功能:
1. 加载本地文档(PDF / Word / Markdown / TXT)
2. 将长文档切成合适大小的文本块(chunk),便于向量检索

设计要点(面试可讲):
- 用递归字符分块器,优先按段落/句子等自然边界切分,而非简单按固定字数
- chunk_size 与 chunk_overlap 的权衡:块太大检索粒度粗,块太小上下文不完整;
  overlap 用于让相邻块之间保留重叠,避免关键信息被切断
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 支持的文件后缀 -> 对应的加载器
# 加载器返回 List[Document],统一了后续处理逻辑
# 注意:.md 用 TextLoader 加载(Markdown 本质是纯文本,避免引入重型 unstructured 依赖)
LOADER_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}

# 分块参数(可根据效果调优)
CHUNK_SIZE = 500      # 每个文本块的最大字符数
CHUNK_OVERLAP = 50    # 相邻块之间的重叠字符数


def load_documents(file_path: str | Path) -> List[Document]:
    """
    加载单个文档文件,返回 LangChain Document 列表。

    参数:
        file_path: 文件路径
    返回:
        List[Document]
    抛出:
        ValueError: 不支持的文件类型
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)
    if loader_cls is None:
        supported = ", ".join(LOADER_MAP.keys())
        raise ValueError(f"不支持的文件类型 '{suffix}',当前支持: {supported}")

    loader = loader_cls(str(path))
    documents = loader.load()

    # 给每个文档片段补充来源信息,方便后续引用溯源
    for doc in documents:
        doc.metadata["source"] = path.name

    return documents


def load_documents_from_dir(dir_path: str | Path) -> List[Document]:
    """
    加载目录下所有支持的文档文件。

    参数:
        dir_path: 目录路径
    返回:
        List[Document]
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在: {root}")

    all_docs: List[Document] = []
    for suffix in LOADER_MAP.keys():
        for file in root.rglob(f"*{suffix}"):
            all_docs.extend(load_documents(file))

    if not all_docs:
        raise ValueError(f"目录 '{root}' 下没有找到支持的文档文件")
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    将文档列表切分成文本块。

    参数:
        documents: 待切分的文档
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数
    返回:
        切分后的 Document 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 切分优先级:优先按段落、句子等自然边界
        separators=["\n\n", "\n", "。", "!", "!", "?", "?", ";", ";", " ", ""],
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    # 简单自测:加载一个测试文档并打印分块结果
    import sys

    if len(sys.argv) < 2:
        print("用法: python app/loader.py <文件或目录路径>")
        sys.exit(1)

    target = sys.argv[1]
    p = Path(target)
    docs = load_documents(target) if p.is_file() else load_documents_from_dir(target)
    chunks = split_documents(docs)
    print(f"共加载 {len(docs)} 个文档片段,切分为 {len(chunks)} 个文本块")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- 块 {i}(来源: {c.metadata.get('source')}) ---")
        print(c.page_content[:200])
