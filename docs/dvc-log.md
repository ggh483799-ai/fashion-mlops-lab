# DVC 数据版本化记录（fashion-mlops-lab）

> 日期：2026-08-23
> 用途：项目二「数据版本化(DVC)」动作的真实操作存档（面试背书）

## 一、做了什么

| 步骤 | 命令 | 结果 |
|---|---|---|
| 初始化 DVC | `dvc init` | 生成 `.dvc/` + `.dvcignore` |
| 纳入数据集 | `dvc add data` | 生成 `data.dvc` 指针（md5 + 82MB + 8 文件） |
| 提交指针 | `git commit` + `git push` | commit `ab6eac4`，指针已上 GitHub |

## 二、data.dvc 指针内容（数据指纹）

```yaml
outs:
- md5: 18d3ba9f82aa57f2f5267753f1659e05.dir
  size: 85828693        # 82MB
  nfiles: 8             # 8 个文件
  hash: md5
  path: data
```

**机制**：Git 里只存这个 2KB 指针；真实数据在 `.dvc/cache`（82MB）+ 可选远端。数据一变 → md5 变 → 指针变 → Git 历史可追溯。

## 三、版本控制命令速查（面试可复现）

```bash
# 数据变更后重新纳入（md5 会变）
.venv/Scripts/python.exe -m dvc add data

# 查看数据版本历史
git log --oneline -- data.dvc

# 回滚到某个数据版本（先切指针，再恢复数据）
git checkout <commit> -- data.dvc
.venv/Scripts/python.exe -m dvc checkout

# 团队复现：clone 后拉数据
# dvc remote add -d myremote <远端路径>   # 配一次
# dvc pull
```

## 四、验证证据

- `dvc status` → `Data and pipelines are up to date.`（数据与指针一致）
- 缓存目录 `.dvc/cache` = 82MB（数据本体由 DVC 管理）
- GitHub 仓库 main 分支含 `data.dvc` 提交（`git log --oneline -- data.dvc` 可查）

## 五、面试话术

> "数据版本化我用 DVC 落地：`dvc add data` 生成 md5 指针进 Git，82MB 数据本体进 DVC 缓存/远端。数据一更新 md5 就变，`git log -- data.dvc` 能看到每个数据版本，`dvc checkout` 能回滚——这样模型回滚时数据也能同步回滚，训练永远可复现。"
