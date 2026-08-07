# 最终第一名 Ruby：二维对齐 posterior 与 XY 邻井安全路由

## 1. 资料范围与结论

本笔记基于最终 leaderboard 第一名 Ruby 的公开复现 notebook、公开训练数据集和代码整理：

* 公开 notebook：[submit-reproduce](https://www.kaggle.com/code/w5833946/submit-reproduce)
* 公开主版本资产：`w5833946/rogii-0801-v2-cv480`
* 本地缓存：[reports/final_leaderboard_ruby](/root/rogii/reports/final_leaderboard_ruby)

结束后的 leaderboard 显示 Ruby 分数为 `5.639`。该方案的最终提交不是单一模型，而是七个训练版本的条件集成：当训练邻井形成的 XY 地质先验被判定安全时，使用 XY-based 集成；否则使用 GR-only 集成。

代码中存在 PF、Transformer、ConvGRU、two-stage U-Net 等可选框架，不能因此认定它们进入了最终版本。以下模型和特征说明以 `0801_V2` 的实际公开配置为准，并以公开 submission notebook 的最终路由和权重为准。

## 2. 问题重写：不是直接回归 TVT

Ruby 将每个 horizontal well 位置的预测写成 typewell 候选层位上的条件分布：

```text
p(T | H)

T：typewell 的候选 TVT level
H：horizontal well 沿 MD 的下采样位置
```

对每个二维 cell `(H, T)`，网络判断：该 horizontal 位置是否与该 typewell level 相对应。最终 TVT 不是从独立回归头产生，而是从此 posterior 得到：

```text
p(T | H) = softmax_T(logits(H, T))
TVT_pred(H) = sum_T p(T | H) * typewell_TVT(T)
```

这使模型同时保留所有候选 level 的概率，而不是先指定唯一 GR match 或先手工构造少量整井候选 path。

## 3. 候选 lattice 如何生成

对每口井，首先从 heel 可见前缀获得：

```text
TVT0 = 最后一个非空 TVT_input
```

然后只在 typewell 的局部窗口中搜索：

```text
candidate_TVT_rel in [-100 ft, +100 ft]
typewell_len = 400
grid step = 0.5 ft
```

代码实际以半格中心生成：

```text
-99.75, -99.25, ..., +99.25, +99.75 ft
candidate_TVT(T) = TVT0 + candidate_TVT_rel(T)
```

对应 typewell GR 用插值构造：

```text
tw_gr_grid_type = interpolate
```

因此，所谓“候选”是每一个 horizontal 位置共享的 400 个 typewell level，不是 full249/sg_path 风格的 anchor 平移、beam 或完整候选 path。二维网络后续自动形成一条连续高概率匹配带。

## 4. 水平井压缩与二维张量

每口井被截取为固定长度：

```text
visible heel prefix = 1,024 个原始位置
target future region = 10,000 个原始位置
raw length = 11,024
downsample = 32
horizontal bins = 345
```

所以模型内部二维搜索图约为：

```text
H x T = 345 x 400 = 138,000 cells / well
```

这让 2D CNN 可以在 GPU 上学习水平向 GR pattern 和垂向 typewell 层序；输出再映射回原始未来采样位置。直接在原始 `11,024 x 400` 图上训练将显著增加显存和推理代价。

## 5. `0801_V2` 的 16 个实际输入通道

所有特征最终堆叠为：

```text
X.shape = [batch, 16, 345, 400]
```

| 分组 | 通道 | 构造与作用 |
|---|---|---|
| typewell | `tw_gr` | 候选 level 的 typewell GR，沿 H 广播。 |
| horizontal | `gr` | 当前水平位置 GR，沿 T 广播。 |
| 配对证据 | `gr_abs_diff` | `abs(GR_horizontal(H) - GR_typewell(T))`，保留真正 H x T 匹配结构。 |
| GR 质量 | `gr_isnan_rate` | 下采样 bin 的 GR 缺失比例。 |
| GR 形状 | `gr_std`, `gr_first_last_delta`, `gr_slope` | 当前 bin 的局部波动和变化。 |
| GR 二次局部形状 | `gr_quadratic_a/b/c/rmse` | 局部曲率、线性项、常数项及二次拟合残差。 |
| typewell 有效性 | `tw_gr_is_nan` | typewell 候选 level 的 GR 是否缺失。 |
| candidate level | `tw_tvt_rel` | 候选 TVT 相对 heel TVT0 的位置。 |
| heel 历史 | `seen_tvt_rel` | 当前位置已知 TVT 相对 heel 的值或相应可见性信息。 |
| heel 与候选关系 | `tw_seen_tvt_abs_diff` | `abs(typewell_TVT(T) - seen_TVT(H))`。 |
| XY 邻井先验 | `geo_tvt_diff` | 从训练邻井估计的地层/TVT 变化先验，沿 T 广播。 |

`0801_V2` 只将 V1 的 `z_diff` 替换为 `geo_tvt_diff`；其余主体输入保持相同。公开配置中的 PF、PF posterior、更多 X/Y Fourier 特征和若干 surface 特征均未启用。

特征在二维图中的放置方式是：

```text
仅依赖 H：gr(H)、GR shape(H)、geo_tvt_diff(H)
  -> 复制到所有 T，得到 [345, 400]

仅依赖 T：tw_gr(T)、tw_tvt_rel(T)、tw_gr_is_nan(T)
  -> 复制到所有 H，得到 [345, 400]

同时依赖 H 与 T：gr_abs_diff(H, T)
  -> 直接形成二维局部岩性匹配面

所有通道 stack
  -> [16, 345, 400]
```

## 6. 神经网络

主模型配置为：

```text
model_name = unet
unet_arch = convnext_small
unet_emb_dim = 32
unet_stem_stride = (2, 4)
unet_dropout = 0.05
regression_head_mode = logits
share_logits_head = True
```

结构可概括为：

```text
16-channel H x T alignment map
-> ConvNeXt-Small style 2D encoder
-> U-Net decoder with skip connections
-> 1-channel alignment logits map [345, 400]
-> softmax on candidate-T axis
-> posterior expected TVT
```

`share_logits_head=True` 表示同一张 logits map 同时用于对齐监督和 TVT posterior expectation，不存在一个独立且不受对齐约束的最终回归头。

## 7. 联合 loss

实际启用权重为：

```text
regression = 0.05 / 3
alignment  = 0.05 * 4 / 3
GR_penalty = 0.05 / 3
```

即相对比例：

```text
TVT regression : alignment posterior : GR consistency = 1 : 4 : 1
```

### 7.1 TVT regression

预测的 posterior mean 与真实目标 `TVT - TVT0` 使用 masked Huber：

```text
L_reg = Huber(TVT_pred - TVT_true, delta = 1.0)
```

Huber 在小误差处类似 MSE，在 hard well 的大误差处类似 MAE，从而避免少数极端井完全主导梯度。

### 7.2 Soft alignment cross-entropy

真实 TVT 在 candidate lattice 上不做 one-hot 标记，而是以真实 level 为中心构造平滑软标签：

```text
alignment_target_mode = exp_smooth
alignment_exp_smooth_sigma = 1.25

L_align = -sum_T q(T | H) * log p(T | H)
```

它直接训练网络把 posterior 放到正确的 typewell level 附近，给出比单一 TVT 回归更密集的二维监督。

### 7.3 GR penalty

对每个 cell 预先计算 typewell GR 和 horizontal GR 的误差。网络若把概率放在 GR 明显不一致的 candidate 上，就会被惩罚：

```text
L_GR = sum_T p(T | H) * GR_error(H, T)
```

它是软物理一致性，而非硬性强迫两侧 GR 相等；网络仍可在 heel、轨迹和空间先验支持时偏离局部 GR 最优点。

未启用的可选 loss 包括 `offset`、CDF、surface curvature / `dS_penalty` 和辅助 `unet_GR_RMSE`。

## 8. XY-neighbor 地质先验

高价值版本 `0801_V2` 用 `geo_tvt_diff` 替代 V1 的 `z_diff`。这条先验由训练井的空间位置、轨迹和标签构造，并对当前井给出沿 H 变化的地层/TVT 趋势。

公开的 GeoPrior 配置大意为：

```text
method = IDW dS XY weighted least squares
neighbor_wells = 12
point_neighbors = 112
distance metric = anisotropic
```

它不是直接将邻井 TVT 复制给测试井，而是作为二维网络的一条水平位置特征。若重建该方法，OOF 时必须对 outer held fold 排除完整 held-fold 井，不能让该井标签进入邻井库。

## 9. Well-level safety router

Ruby 没有对每口井都使用含 XY prior 的模型。每口井汇总以下诊断：

```text
geo_nbr_distance
geo_nbr_distance_q10
geo_nbr_path_alignment
geo_radial_extrap_score
geo_prefix_weight_ratio
```

只有同时满足：

```text
geo_nbr_distance < 2459
geo_nbr_distance_q10 < 1426
geo_nbr_path_alignment > 0.96
geo_radial_extrap_score < 1.45
geo_prefix_weight_ratio < 0.315
```

才定义为 `xy_safe=True`，并启用 XY-based 集成。否则使用更保守的 GR-only 集成。这是该方案保护难井/外推井、避免不可靠空间插值破坏结果的关键。

## 10. 最终七版本集成

公开 notebook 会分别对七个版本运行 GPU 推理，再按井路由。各版本公开资产标签为：

| marker | 自称训练 CV |
|---|---:|
| `0719_V1` | 5.09 |
| `0724_V1` | 4.86 |
| `0729_V3` | 5.53 |
| `0801_V1` | 5.16 |
| `0801_V2` | 4.80 |
| `0802_V2` | 5.13 |
| `0803_V2` | 5.00 |

这些值使用 Ruby 自己的验证方案，不能和本项目 canonical Geo5 CV 直接比较。

GR-only 分支为：

```text
0719_V1: 1.00
0729_V3: 0.50
0801_V1: 1.00
```

XY-based 分支为：

```text
0719_V1: 0.25
0801_V1: 0.25
0724_V1: 1.00
0801_V2: 1.00
0803_V2: 1.00
```

每个分支内按权重归一化后输出最终预测。

## 11. 与本项目模型的关系

Ruby 与 GSN 最接近：二者都对 typewell candidate TVT 与 horizontal MD 建立二维图。主要不同是：

| Ruby | GSN |
|---|---|
| 输出 typewell candidate posterior，最终取 posterior mean。 | 输出 SDF field 和 segmentation field，再由 argmin/Viterbi/dual 解码。 |
| 对齐 soft cross-entropy 是主监督。 | SDF 回归与 segmentation 是主监督。 |
| 路径连续性主要由二维 CNN 隐式学习。 | 可用 Viterbi/dual 显式优化路径连续性。 |
| 加入安全门控的 XY-neighbor prior。 | 主体以本井 GR、轨迹、typewell 证据为主。 |

它与 `sg_path` 的差异也很重要：Ruby 用密集、固定的 400-level lattice；sg_path 用稀疏、异构的整井 path candidates。Ruby 不会因为 selector 未把正确的 2 ft offset path 放进 top-k 而漏掉 level，但 posterior mean 在多模态条件下可能落在两个真实 level 的中间。sg_path/GSN 的 mode、cluster 和 Viterbi 更适合显式处理这种多解。

## 12. 可复盘的经验

Ruby 的成功不应被概括为“使用 ConvNeXt 就赢了”。更有价值的结构是：

```text
密集 typewell candidate lattice
+ 二维 H x T 对齐 posterior
+ 数值 / 对齐 / GR 三重监督
+ 受 strict safety gate 约束的空间邻井先验
+ 多版本、条件集成
```

对本项目最安全的迁移方式是先在 canonical Geo5 下构建一条独立二维 posterior 臂，并输出 posterior mode、mean、entropy、top1-top2 gap 和多模态诊断给 anchor/gate；先验证其 standalone 与低相关性，再考虑小权重融合。不能因 Ruby 的最终分数直接把 XY prior、posterior mean 或多个版本预测无约束地写进 full249、GSN、WARP、HMM 的所有路径。
