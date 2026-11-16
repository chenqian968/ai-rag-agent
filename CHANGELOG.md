# 更新日志（CHANGELOG）

本项目的所有重要变更记录，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 规划中
- 引入 rerank 模型对检索结果精排
- 支持查询改写（query rewriting）
- 对话记忆管理（历史压缩/摘要）
- 流式响应与更完整的前端体验

## [1.0.0] - 2026-11-16

### 新增
- 软件工程标准文档四件套：需求规格说明书、概要设计说明书、详细设计说明书、测试用例文档
- 项目答辩讲稿（含高频面试问答）
- 系统架构图与时序图（SVG + PNG）
- CHANGELOG 与 MIT License

### 修复
- 修复更换知识库后 Agent 不检索新文档的问题（系统提示/工具描述移除写死的领域）
- 修复中文查询检索不到目标文档的问题（n-gram 切词替代去停用字拼串）
- 修复 ingest 入库时 source 元数据记为临时文件名的问题

### 新增
- 新增大数据、穿搭、花语三份测试知识库文档

## [0.4.0] - 2026-11-14

### 新增（第 4 周：产品化）
- FastAPI 后端（/api/chat、/api/ingest、/api/reset、/api/health）
- 原生网页前端（文档上传 + 对话界面）
- Dockerfile 与 docker-compose.yml
- 面试提纲

## [0.3.0] - 2026-11-13

### 新增（第 3 周：Agent 化）
- LangGraph ReAct 循环（agent.py）
- Agent 工具：search_knowledge、calculate
- 多轮记忆（agent_cli.py）
- Function Calling 改用 bind_tools（修复手写序列化失败）

## [0.2.0] - 2026-11-12

### 新增（第 2 周：RAG 知识库）
- 文档加载与递归分块（loader.py）
- 向量库与混合检索（vectorstore.py）
- RAG 问答命令行（rag.py）
- 技术博客《RAG 检索不准？我的 4 个调优实践》

### 修复
- 优化 RAG 检索：移除问题清单污染、按条目分块、混合检索关键词兜底
- 修复 IPv6 优先导致的 DNS 连接失败（network.py 强制 IPv4）

## [0.1.0] - 2026-11-09

### 新增（第 1 周：LLM 基础调用）
- 命令行流式聊天机器人（chat.py）
- 环境配置（.env、.env.example）
- 项目初始化与 Git 仓库
