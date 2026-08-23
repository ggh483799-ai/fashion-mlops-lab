# API 测试记录（第 2 幕服务化验证）

> 日期：2026-08-23
> 服务：deploy/app.py（FastAPI），uvicorn 127.0.0.1:8000
> 测试样本：test_img.png（Fashion-MNIST 测试集第 1 张，真实类别 Ankle boot）

## 三端点验证结果

### 1. GET /healthz —— 存活探针
```
$ curl -s http://127.0.0.1:8000/healthz
{"status":"ok","model":"baseline.pt"}
```

### 2. POST /predict —— 图片分类（上传 test_img.png）
```
$ curl -s -F "file=@test_img.png" http://127.0.0.1:8000/predict
{"label":9,"class_name":"Ankle boot","probability":1.0,
 "top3":[{"class_name":"Ankle boot","probability":1.0},
         {"class_name":"Sneaker","probability":0.0},
         {"class_name":"Sandal","probability":0.0}]}
```
✅ 预测正确（真实类别 Ankle boot，模型给出 100% 置信度）

### 3. GET /metrics —— Prometheus 指标（纯文本）
```
# HELP model_predict_requests_total Total /predict requests
# TYPE model_predict_requests_total counter
model_predict_requests_total 1.0
```
✅ Prometheus 可解析格式（`# HELP`/`# TYPE` + 样本行）

## 踩坑记录

1. **python-multipart 缺失**：FastAPI `UploadFile` 需要 `python-multipart`，未装则启动即报 RuntimeError → `pip install python-multipart` 并写入 requirements.txt。
2. **/metrics 误用 JSONResponse**：指标文本被 JSON 转义（带引号），Prometheus 解析会失败 → 改用 `Response(content=..., media_type=CONTENT_TYPE_LATEST)`。
3. **CLASS_NAMES 依赖数据集**：原实现从 `datasets.FashionMNIST(...).classes` 加载，容器内无 data/ 会启动失败 → 硬编码标准 10 类。
4. Windows 杀端口进程：Git Bash 里 `taskkill //F` 参数会被转义，用 PowerShell `Stop-Process` 最稳。

## 复现命令

```bash
cd fashion-mlops-lab
.venv/Scripts/python.exe -m uvicorn deploy.app:app --host 127.0.0.1 --port 8000
# 另开终端：
curl -s http://127.0.0.1:8000/healthz
curl -s -F "file=@test_img.png" http://127.0.0.1:8000/predict
curl -s http://127.0.0.1:8000/metrics | head
```
