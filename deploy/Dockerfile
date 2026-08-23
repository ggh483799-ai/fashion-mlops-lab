# ---------- Stage 1: builder（装依赖） ----------
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ---------- Stage 2: runtime（最小运行镜像） ----------
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 只拷贝依赖产物（builder 的 site-packages），不拷贝构建中间层
# 启动用 `python -m uvicorn`（模块方式），不依赖 /usr/local/bin 下的 entry point 文件
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 应用代码 + 模型权重（不入库 .venv / data，最小化镜像体积）
COPY deploy/app.py deploy/app.py
COPY eval/eval.py eval/eval.py
COPY models/baseline.pt models/baseline.pt

# 非 root 运行（安全最佳实践）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查：探活 /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
