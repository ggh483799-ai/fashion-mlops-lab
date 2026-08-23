# 证据包：评估门禁拦截坏模型（面试展示用）

> 日期：2026-08-23
> 事件：坏模型（5000 样本 + 1 epoch）提交 → 评估门禁 FAIL → 阻断上线
> 定位：这是项目二「评估门禁卡过模型吗」追问的**实锤证据**

## 判定结论（screenshots/eval-bad.json）

```json
{
  "model": "models/bad.pt",
  "baseline": "models/baseline.pt",
  "new_metrics":   { "accuracy": 0.6966, "mean_latency_ms": 0.12, "peak_mem_mb": 325.9 },
  "baseline_metrics": { "accuracy": 0.9204, "mean_latency_ms": 0.13, "peak_mem_mb": 326.0 },
  "result": "FAIL",
  "reasons": ["accuracy 0.6966 vs baseline 0.9204 (drop 0.2238 > 0.005)"]
}
```

## 复现命令（面试现场可跑）

```bash
# 1) 训练坏模型（故意缩小数据 + 少 epoch）
.venv/Scripts/python.exe train/train.py --epochs 1 --max-samples 5000 --out models/bad.pt
# 2) 过门禁 → 输出 FAIL，退出码 1
.venv/Scripts/python.exe eval/eval.py --model models/bad.pt --baseline models/baseline.pt; echo $?
# 3) 对照：好模型（基线）→ PASS，退出码 0
.venv/Scripts/python.exe eval/eval.py --model models/baseline.pt --baseline models/baseline.pt; echo $?
```

## 为什么这个坏模型会被卡（根因）

- 只用了 8.3% 训练数据（5000/60000）+ 只训 1 epoch → 欠拟合，学不到泛化特征
- 训练集 acc 0.66 vs 基线训练集 acc 0.96；测试集 acc 0.6966 vs 基线 0.9204
- 掉 22.4 个点，远超 0.5 点阈值 → 拦截
- 叙事价值：**只看训练曲线"在收敛"，一上评估集就现原形**——这正是门禁存在的意义

## 真实踩坑（Postmortem 配套素材）

- 早期用 P99 latency + 1.2 倍阈值，CPU 单张推理 <1ms 噪声大，**同一模型自比都误判 FAIL**
- 修复：预热 3 批 + 改用 mean latency + 阈值放宽到 1.5 倍 → 稳定（好模型 PASS ×2 / 坏模型 FAIL ×2）
- 完整复盘见 `docs/postmortem-20260823-bad-model.md`

## 云端证据（GitHub Actions，2026-08-23）

- 仓库：https://github.com/ggh483799-ai/fashion-mlops-lab
- PR #1（坏模型 candidate.pt 提交）：https://github.com/ggh483799-ai/fashion-mlops-lab/pull/1
- CI run #32615657961 → **eval-gate FAIL，exit 1**，判定约 48 秒完成
- 云端判定输出（与本地完全一致）：

```json
{
  "new_metrics": { "accuracy": 0.6966, "mean_latency_ms": 0.216 },
  "baseline_metrics": { "accuracy": 0.9204, "mean_latency_ms": 0.215 },
  "result": "FAIL",
  "reasons": ["accuracy 0.6966 vs baseline 0.9204 (drop 0.2238 > 0.005)"]
}
```

- main 分支保护：要求 eval-gate 检查通过才能 merge（PR #1 因此无法合并）
