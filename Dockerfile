# AI 知识库问答 Agent - Docker 镜像
# 构建:docker build -t ai-rag-agent .
# 运行:docker run -p 8000:8000 --env-file .env ai-rag-agent

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖清单,利用 Docker 层缓存(依赖不变时无需重装)
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY .env.example ./.env.example

# 数据目录(向量库持久化,可挂载卷)
RUN mkdir -p /app/data

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
