"""
train.py — Fashion-MNIST 服装分类训练脚本（CPU 友好）

用法:
    python train/train.py --epochs 10 --batch-size 128 --out models/baseline.pt
    python train/train.py --epochs 1 --max-samples 5000 --out models/bad.pt   # 造坏模型

每次训练自动记录参数 + 训练 acc 到 reports/train-{ts}.json
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Fashion-MNIST 全局归一化参数（torchvision 官方推荐）
MEAN, STD = 0.2860, 0.3530


class FashionCNN(nn.Module):
    """小型 CNN：2 conv + 2 fc，单通道 28x28 输入，CPU 5-15 分钟可训完"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    ap = argparse.ArgumentParser(description="Fashion-MNIST 训练")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--max-samples", type=int, default=0, help="0=全量 60000；>0 只取前 N 条（造坏模型用）")
    ap.add_argument("--out", required=True, help="模型输出路径，如 models/baseline.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train_ds = datasets.FashionMNIST(root="data", train=True, download=False, transform=tf)
    if args.max_samples and args.max_samples < len(train_ds):
        train_ds = Subset(train_ds, range(args.max_samples))
        print(f"[data] 使用子集 {args.max_samples}/{60000} 条（故意缩小 = 造坏模型）")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = FashionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = corr = 0
        loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        acc = corr / tot
        print(f"[epoch {ep}/{args.epochs}] loss={loss_sum / tot:.4f} train_acc={acc:.4f} ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)

    report = {
        "model": args.out,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_samples": args.max_samples,
        "final_train_acc": round(acc, 4),
        "train_seconds": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs("reports", exist_ok=True)
    rp = f"reports/train-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved {args.out} | report {rp}")


if __name__ == "__main__":
    main()
