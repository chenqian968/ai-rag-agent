# AI 知识库问答 Agent(求职项目)

基于 **RAG + AI Agent** 的垂直领域智能问答助手。目标岗位:大模型应用 / AI Agent 开发工程师。

> 📌 当前进度:**第 4 周(完成)** — 已产品化:FastAPI 后端 + 网页界面 + Docker 部署。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| 大模型 | DeepSeek(OpenAI 兼容接口) |
| Embedding | 硅基流动 BAAI/bge-m3 |
| 向量库 | Chroma |
| RAG 框架 | LangChain |
| Agent 框架 | LangGraph(ReAct 循环 + Function Calling) |
| 后端 | FastAPI + Uvicorn |
| 前端 | 原生 HTML + CSS + JS(轻量,无需构建) |
| 部署 | Docker |

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

### 6. ReAct Agent 对话(第 3 周)

第 2 周是"固定检索→生成"的流水线;第 3 周升级为**能自主决策的 Agent**:

```bash
python -m app.agent_cli
```

Agent 具备两个工具,会自己判断该用哪个:

| 工具 | 作用 | 示例问题 |
|---|---|---|
| `search_knowledge` | 检索知识库 | "什么是市盈率PE?" |
| `calculate` | 数值计算 | "10万本金年化5%一年多少利息?" |

同时支持**多轮记忆**,能追问、能指代(比如先问市盈率,再问"那它和市净率有什么区别?"),Agent 能理解"它"指代上文。

### 7. 网页版(第 4 周,推荐)

启动后端服务,打开网页对话界面:

```bash
# 启动后端(默认 http://127.0.0.1:8000)
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

然后在浏览器打开 http://127.0.0.1:8000/ ,即可:
- 在网页上**上传文档**建立知识库
- 与 Agent **对话问答**(支持多轮记忆)

接口文档(自动生成):http://127.0.0.1:8000/docs

### 8. Docker 一键部署(第 4 周)

```bash
# 方式一:docker compose(推荐)
docker compose up -d

# 方式二:直接构建运行
docker build -t ai-rag-agent .
docker run -p 8000:8000 --env-file .env ai-rag-agent
```

## 项目结构

```
ai-rag-agent/
├── app/            # 应用代码
│   ├── chat.py     # 命令行聊天(第 1 周)
│   ├── loader.py   # 文档加载与分块(第 2 周)
│   ├── vectorstore.py  # 向量库与检索(第 2 周)
│   ├── rag.py      # RAG 问答命令行(第 2 周)
│   ├── tools.py    # Agent 工具定义(第 3 周)
│   ├── agent.py    # ReAct Agent(LangGraph 编排,第 3 周)
│   ├── agent_cli.py # Agent 交互命令行(第 3 周)
│   └── api.py      # FastAPI 后端(第 4 周)
├── frontend/       # 前端界面(第 4 周)
│   ├── index.html
│   └── static/     # css / js
├── data/           # 数据(向量库、上传文档等,已 gitignore)
├── docs/           # 文档、测试语料、架构图、技术笔记
├── Dockerfile      # 容器镜像(第 4 周)
├── docker-compose.yml
├── .env.example    # 环境变量模板
└── requirements.txt
```

## 路线图(四周)

- [x] 第 1 周:LLM 基础调用 + 环境搭建
- [x] 第 2 周:RAG 知识库(文档分块 + Embedding + Chroma 检索)
- [x] 第 3 周:Agent 化(工具调用 + 多轮记忆)
- [x] 第 4 周:FastAPI + 网页前端 + Docker 上线

## License

MIT
