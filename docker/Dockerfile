# ===========================================
# EXO Cluster Manager Docker 镜像
# ===========================================
# 构建命令 (从项目根目录): docker build -f docker/Dockerfile -t exo-manager:latest .
# 运行命令: docker run -p 8080:8080 exo-manager:latest
# 或使用: cd docker && docker compose up -d
# ===========================================

FROM python:3.11-slim

LABEL maintainer="EXO Team"
LABEL description="EXO Distributed AI Inference Cluster Manager"
LABEL version="1.0.0"

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EXO_ENV=production \
    EXO_PORT=18080

# 安装系统依赖 (gRPC 和加密库需要, tzdata 用于时区支持)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 设置默认时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 创建工作目录
WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 将默认模型配置模板复制到独立路径（不被 volume mount 遮盖）
RUN mkdir -p /app/templates && cp data/custom_models.json /app/templates/

# 创建数据目录 (用于存储数据库、日志等)
RUN mkdir -p /app/data /app/logs

# 暴露端口
EXPOSE 18080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:18080/api/cluster/status')" || exit 1

# 启动命令 (使用 start_manager.py 支持完整功能)
CMD ["python", "start_manager.py", "--port", "18080"]
