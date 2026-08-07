# ROGII 井筒地质预测代码包

本仓库是 ROGII Wellbore Geology Prediction 项目的精简、可审计源码包。它仅保留五条核心模型臂和第一名 Ruby 方案的公开代码/本地严格复现代码：

- `models/full249/`：249 维 LightGBM anchor、Geo5 OOF 特征重建和训练入口；
- `models/warp/`：GR 序列 WARP（SCA-U2Net + typewell cross-attention）；
- `models/hmm/`：GR/typewell likelihood 状态空间解码；
- `models/gsn/`：GeoSteerNet SDF + segmentation 路径模型及 v3.51 Geo5 OOF 复现；
- `models/sg_path/`：候选 path、SG cluster selector、member selector 和严格 OOF sidecar 构建；
- `models/ruby_1st/`：第一名公开 `0724v1` 与 `0803v2` 代码，以及禁用 XY 邻井和 target-PF 泄漏后的 Geo5 工作副本。

详细中文说明见 [docs/代码使用与复现指南.md](docs/代码使用与复现指南.md)。各模型原理分别见 `docs/full249.md`、`docs/warp.md`、`docs/hmm.md`、`docs/gsn.md`、`docs/sg_path.md` 与 `docs/1st.md`。

## 固定验证协议

所有当前 OOF 比较使用 `metadata/geo_kmeans_5fold.csv`：

```text
SHA256: ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da
```

这是以整口井为单位的 canonical Geo5 划分。不得混入旧 GroupKFold、物理 54-group 或非 OOF 缓存。

## 不包含的内容

为避免把数 TB 的缓存、权重、提交 CSV 和训练标签产物误当成源码，本仓库不含：比赛数据、fold checkpoint、OOF parquet/npy、PF 缓存、Kaggle dataset 导出、模型权重和任何 submission.csv。运行训练前需将竞赛数据放入 `datasets/rogii-wellbore-geology-prediction/`，并为相应臂重建训练派生产物。

## 数据安全

部署与 OOF 必须只使用：水平井可见 `MD/X/Y/Z/GR`、可见 heel `TVT_input`、typewell 文件和训练 fold 学到的冻结资产。未知段真实 `TVT` 只可作为训练标签和 held-fold 评估目标，绝不能进入测试或验证特征。
