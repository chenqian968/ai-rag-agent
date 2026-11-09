# AI 知识库问答 Agent(求职项目)

基于 **RAG + AI Agent** 的垂直领域智能问答助手。目标岗位:大模型应用 / AI Agent 开发工程师。

> 📌 当前进度:**第 1 周** — LLM 基础调用(命令行聊天)已验证环境可跑通。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| 大模型 | DeepSeek(OpenAI 兼容接口) |
| Agent 框架 | LangChain + LangGraph(第 3 周加入) |
| 后端 | FastAPI(第 4 周加入) |
| 向量库 | Chroma(第 2 周加入) |
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
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

> Key 申请地址:https://platform.deepseek.com/ (注册后充值小额额度即可)

### 3. 运行命令行聊天

```bash
python app/chat.py
```

## 项目结构

```
ai-rag-agent/
├── app/            # 应用代码
│   └── chat.py     # 命令行聊天(第 1 周)
├── data/           # 数据(向量库、上传文档等,已 gitignore)
├── docs/           # 文档、架构图、技术笔记
├── .env.example    # 环境变量模板
└── requirements.txt
```

## 路线图(四周)

- [x] 第 1 周:LLM 基础调用 + 环境搭建
- [ ] 第 2 周:RAG 知识库(文档分块 + Embedding + Chroma 检索)
- [ ] 第 3 周:Agent 化(工具调用 + 多轮记忆)
- [ ] 第 4 周:FastAPI + Streamlit + Docker 上线

## License

MIT
