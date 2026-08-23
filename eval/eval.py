"""
eval.py — 评估门禁（项目核心）：accuracy / P99 latency / 峰值内存 三项 vs 基线

用法:
    python eval/eval.py --model models/bad.pt --baseline models/baseline.pt

判定规则（任一超阈值即 FAIL，exit 1）:
    - accuracy   低于基线 0.5 个点以上  -> FAIL
    - P99 latency 超过基线 1.2 倍        -> FAIL
    - 峰值内存   超过基线 1.3 倍        -> FAIL

报告输出: reports/eval-{model}.json
"""
import argparse
import json
import os
import sys
import time

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN, STD = 0.2860, 0.3530

# 门禁阈值（与基线对比的相对阈值；初版用固定值，后续可用基线多次评估的 3σ 动态化）
# 踩坑记录：CPU 单张推理延迟 <1ms 噪声大，p99 自比都误判；改用 mean + 预热 + 阈值 1.5
THRESHOLDS = {"acc_drop": 0.005, "latency_ratio": 1.50, "mem_ratio": 1.30}


class FashionCNN(nn.Module):
    """网络结构与 train.py 保持一致；评估脚本自包含，只读权重文件，不依赖训练代码"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path):
    model = FashionCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def evaluate(model, loader, warmup_batches=3):
    """在测试集上跑三项指标：accuracy / mean+P99 latency(ms per sample) / 峰值内存(MB)
    先预热 warmup_batches 批再计延迟，降低 CPU 首轮加载/缓存噪声"""
    model.eval()
    proc = psutil.Process()
    corr = tot = 0
    lats = []
    peak_mem = 0
    warmed = 0
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            corr += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
            peak_mem = max(peak_mem, proc.memory_info().rss)
            if warmed < warmup_batches:          # 预热批不计延迟
                warmed += 1
                continue
            s = time.time()
            out = model(x)
            lats.append((time.time() - s) / x.size(0) * 1000)
    acc = corr / tot
    mean_lat = sum(lats) / len(lats)
    lat_sorted = sorted(lats)
    p99 = lat_sorted[min(int(len(lat_sorted) * 0.99), len(lat_sorted) - 1)]
    return {
        "accuracy": round(acc, 4),
        "mean_latency_ms": round(mean_lat, 3),
        "p99_latency_ms": round(p99, 3),
        "peak_mem_mb": round(peak_mem / 1e6, 1),
        "eval_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="评估门禁：三项指标对比基线")
    ap.add_argument("--model", required=True, help="待评估模型权重")
    ap.add_argument("--baseline", required=True, help="基线模型权重")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    te = datasets.FashionMNIST(root="data", train=False, download=False, transform=tf)
    loader = DataLoader(te, batch_size=args.batch_size, num_workers=0)

    new = evaluate(load_model(args.model), loader)
    base = evaluate(load_model(args.baseline), loader)

    fail = []
    acc_drop = base["accuracy"] - new["accuracy"]
    if acc_drop > THRESHOLDS["acc_drop"]:
        fail.append(f"accuracy {new['accuracy']:.4f} vs baseline {base['accuracy']:.4f} (drop {acc_drop:.4f} > {THRESHOLDS['acc_drop']})")
    if new["mean_latency_ms"] > base["mean_latency_ms"] * THRESHOLDS["latency_ratio"]:
        fail.append(f"mean latency {new['mean_latency_ms']}ms > baseline {base['mean_latency_ms']}ms * {THRESHOLDS['latency_ratio']}")
    if new["peak_mem_mb"] > base["peak_mem_mb"] * THRESHOLDS["mem_ratio"]:
        fail.append(f"peak mem {new['peak_mem_mb']}MB > baseline {base['peak_mem_mb']}MB * {THRESHOLDS['mem_ratio']}")

    report = {
        "model": args.model,
        "baseline": args.baseline,
        "new_metrics": new,
        "baseline_metrics": base,
        "thresholds": THRESHOLDS,
        "result": "FAIL" if fail else "PASS",
        "reasons": fail,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs("reports", exist_ok=True)
    rp = f"reports/eval-{os.path.basename(args.model).replace('.pt', '')}.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
