# AI 知识库问答 Agent(求职项目)

基于 **RAG + AI Agent** 的垂直领域智能问答助手。目标岗位:大模型应用 / AI Agent 开发工程师。

> 📌 当前进度:**第 2 周** — RAG 知识库已实现(文档加载 → 分块 → Embedding → Chroma 检索 → 带引用问答)。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| 大模型 | DeepSeek(OpenAI 兼容接口) |
| Embedding | 硅基流动 BAAI/bge-m3 |
| 向量库 | Chroma |
| RAG 框架 | LangChain |
| Agent 框架 | LangGraph(第 3 周加入) |
| 后端 | FastAPI(第 4 周加入) |
| 前端 | Streamlit(第 4 周加入) |

## 快速开始

### 1. 准备 Python 环境

```bash
# 要求 Python 3.11
python -m venv .venv

# Windows 激活
.venv\Scripts\activate
# macOS / Linux 激活
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制示例配置
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

编辑 `.env`,填入你的 Key:

```
DEEPSEEK_API_KEY=sk-xxxxxxxx          # 对话模型(deepseek)
SILICONFLOW_API_KEY=sk-xxxxxxxx       # embedding 模型(硅基流动)
```

> Key 申请地址:
> - 对话模型:https://platform.deepseek.com/
> - Embedding 模型:https://siliconflow.cn/ (注册后获取 API Key)

### 3. 运行命令行聊天

```bash
python app/chat.py
```

### 4. RAG 知识库问答(第 2 周)

先导入文档建立知识库(支持 PDF / Word / Markdown / TXT):

```bash
# 导入单个文件
python -m app.rag ingest docs/ai-agent-guide.md

# 或导入整个目录
python -m app.rag ingest docs/
```

然后进入交互式问答:

```bash
python -m app.rag chat
```

提问后,助手会检索知识库中的相关内容并**带引用作答**([资料1]、[资料2] 等)。

### 5. 换一份自己的文档

把你要问答的 PDF / Word / Markdown / TXT 文件放到 `docs/` 目录(或任意位置),然后重新导入:

```bash
# 1. (可选)想彻底换新文档时,先清空旧知识库,避免旧内容混入
python -m app.rag reset

# 2. 导入你自己的文档(替换成实际路径)
python -m app.rag ingest docs/你的文档.pdf

# 3. 问答
python -m app.rag chat
```

> `reset` 会删除向量库数据(不影响你的源文档文件),导入时若想"追加"多个文档,跳过 reset 直接 ingest 即可。

## 项目结构

```
ai-rag-agent/
├── app/            # 应用代码
│   ├── chat.py     # 命令行聊天(第 1 周)
│   ├── loader.py   # 文档加载与分块(第 2 周)
│   ├── vectorstore.py  # 向量库与检索(第 2 周)
│   └── rag.py      # RAG 问答命令行(第 2 周)
├── data/           # 数据(向量库、上传文档等,已 gitignore)
├── docs/           # 文档、测试语料、架构图、技术笔记
├── .env.example    # 环境变量模板
└── requirements.txt
```

## 路线图(四周)

- [x] 第 1 周:LLM 基础调用 + 环境搭建
- [x] 第 2 周:RAG 知识库(文档分块 + Embedding + Chroma 检索)
- [ ] 第 3 周:Agent 化(工具调用 + 多轮记忆)
- [ ] 第 4 周:FastAPI + Streamlit + Docker 上线

## License

MIT
