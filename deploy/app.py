"""
app.py — Fashion-MNIST 服装分类模型服务化（FastAPI）

三端点：
    POST /predict  接收图片(UploadFile) → 返回 {label, class_name, probability}
    GET  /healthz  存活探针 → {"status": "ok"}
    GET  /metrics  Prometheus 指标（QPS / latency 直方图 / 错误率）

启动：uvicorn deploy.app:app --host 0.0.0.0 --port 8000
"""
import io
import os
import sys
import time

import torch
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from torchvision import transforms

# 让脚本从任何 cwd 启动都能找到 eval.eval（复用网络结构定义）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval import FashionCNN, MEAN, STD  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "baseline.pt")

app = FastAPI(title="fashion-mlops-lab", version="0.1.0")

# ---- Prometheus 指标 ----
PREDICT_QPS = Counter("model_predict_requests_total", "Total /predict requests")
PREDICT_ERRORS = Counter("model_predict_errors_total", "Total /predict errors")
PREDICT_LATENCY = Histogram(
    "model_predict_latency_seconds",
    "Latency of /predict (seconds)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---- 模型加载（启动时一次） ----
model = FashionCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
# Fashion-MNIST 固定 10 类（硬编码，避免运行时依赖数据集目录；容器部署无 data/ 也能启动）
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    PREDICT_QPS.inc()
    t0 = time.time()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("L").resize((28, 28))
        x = _transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax())
        return {
            "label": idx,
            "class_name": CLASS_NAMES[idx],
            "probability": round(float(prob[idx]), 4),
            "top3": [
                {"class_name": CLASS_NAMES[i], "probability": round(float(prob[i]), 4)}
                for i in prob.argsort(descending=True)[:3]
            ],
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 输入解析失败 → 400
        PREDICT_ERRORS.inc()
        raise HTTPException(status_code=400, detail=f"bad image: {e}") from e
    finally:
        PREDICT_LATENCY.observe(time.time() - t0)


@app.get("/metrics")
def metrics():
    """Prometheus 抓取端点：必须返回纯文本（Response），不能用 JSONResponse 否则会被 JSON 转义"""
    return Response(content=generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)
