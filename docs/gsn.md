# GSN：GeoSteerNet 二维地质对齐模型

## 1. 模型定位

GSN 是队友实现的 GeoSteerNet。它不是把每个水平井采样点独立回归为 TVT，也不是 SCA-U2Net 的一维 drift 回归器。它把问题改写为：对水平井的每个 MD 位置，在该井对应的 typewell 上寻找最一致的 TVT 位置，并要求所有位置连成连续的地质路径。

因此 GSN 的核心输出不是普通的一维预测，而是一个二维的匹配场（cost / SDF map）：

```text
纵轴 T：typewell 上的候选 TVT 网格
横轴 H：水平井沿 MD 的位置
单元格 (T, H)：该水平井位置对应此 typewell TVT 的程度
```

该建模方式同时保留局部 GR 匹配、多位置候选和整井连续性，因而与 full249 的表格特征、WARP 的一维 sequence-to-sequence 增量模型、HMM 的显式状态转移具有有价值的残差差异。

## 2. 输入与二维对齐图

每口井会把 typewell 重新采样到 TVT 网格，并将水平井重采样到固定的 MD 位置网格。当前队友 v3.51 配置为：

```text
typewell TVT 网格长度 T_SIZE = 272
水平井原始长度约 12480，按 H_S = 24 下采样为 520 个位置
可见 heel history = 128 个原始点，未来段 = 144 个原始点
```

typewell 侧的有效主特征是 `GR`；当前 v3.51 的 `TYPEWELL_FEATURES` 只启用了这一项。水平井侧使用经过 heel 可见段仿射校准的 `GR`，再附加下列几何和方向特征：

```text
dz, dx, dy               相对 heel 的轨迹位移
sin_dip, cos_dip         井斜方向
sin_dir, cos_dir         方位方向
dmd                      MD 增量
```

也就是说，水平井侧共有 9 个特征通道。它们与 typewell GR 通过广播组合为二维图；图中另外加入手工 `gr_diff` 和 `LearnedGRCorrelation` 生成的 correlation volume。后者用于学习“不是完全相同、但形状和层序仍相符”的 GR 对应关系。

```text
typewell GR (T)
           +-- 广播到 T x H --+
horizontal GR / geometry (H)   +--> 二维对齐特征图
手工 gr_diff                   |
learned GR correlation volume -+
```

heel 可见 `TVT_input` 只被用作可见历史、GR 校准和最终路径锚定；未来真实 TVT 不进入输入。测试阶段使用的也是完整 horizontal GR/轨迹、heel 前缀 TVT_input 和测试 typewell，满足 Private-safe 条件。

## 3. 监督目标：SDF 与分割线

GSN 的主监督目标是 signed distance field（SDF）：

```text
sdf(T, H) = (horizontal_TVT(H) - typewell_TVT(T)) / 40
```

该值在训练中限制在大约正负 120 ft 的有效范围。真实对应位置附近 SDF 接近零；上方和下方则有相反符号。相比把唯一正确 cell 写成单个尖点标签，SDF 提供了连续的距离和方向信息，训练信号更稳定。

网络还有第二个 segmentation head，预测每一列中真实路径附近的概率。它不是独立提交的路径，而是给后续 decode 提供“此处确实像地层线”的补充证据。

```text
二维特征图
  -> SDF head：各候选 TVT 距真实对应线的有符号距离
  -> segmentation head：各候选 TVT 属于地层线附近的概率
```

## 4. GeoSteerNet 网络结构

队友的 GSN 采用二维 ResNet/U-Net 风格的 encoder-decoder，而不是将 T 和 H 当成普通特征列：

```text
二维对齐图
-> ResNet-style 2D encoder
-> ASPP 多尺度上下文模块
-> 带 skip connection 的 U-Net decoder
-> SDF regression head + segmentation head
```

二维卷积可同时看到 typewell TVT 方向和水平井 MD 方向的局部纹理。ASPP 负责聚合不同尺度的 GR pattern；decoder 恢复精细路径位置；双头分别学习连续几何距离和离散线存在性。

v3.51 在 backbone 和 head 中都没有直接使用 history token（`USE_HISTORY_IN_BACKBONE=False`、`USE_HISTORY_IN_HEAD=False`）。这并不表示忽略 heel：heel 信息仍用于输入校准和最终 anchor，只是避免把历史片段作为会过强主导二维匹配的网络 token。

## 5. 训练与防泄漏方式

每个 fold checkpoint 仅用 Geo5 训练井训练。已采用的统一划分是：

```text
OOF/geo_kmeans_5fold.csv
SHA256: ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da
```

训练损失由两类部分组成：

* masked hybrid SDF loss：MSE 与 L1 的组合，重点关注要预测的未来区域。
* segmentation loss：BCE、Dice，以及对远离真实线区域误报的惩罚。

主要增强和正则包括：

* H 轴翻转，避免模型把固定 MD 方向当作标签捷径。
* 随机 Savitzky-Golay GR 平滑与 GR noise-transfer，增强不同井之间的测井形态变化适应性。
* PFE（Predicting From Earlier），人为缩短可见历史，使训练接近较少前缀信息的预测情景。
* dropout 与 learned correlation，使网络不依赖单一的绝对 GR 幅值匹配。

因此 OOF 的每个预测行来自未见该井的 fold 模型；不能将全训练集模型对训练行的 in-fold 输出混作 GSN OOF。

## 6. 推理、场级五折集成与路径解码

对一口测试井，五个 fold checkpoint 都输出整张 SDF field 和 segmentation field。正确的集成顺序是先对 field 平均，再解码，不是先各自变成 TVT 路径后简单平均：

```text
5 个 fold checkpoint
-> 每个输出 SDF map 和 segmentation map
-> 分别对两个 map 做 field-level average
-> decode 成一条连续 TVT path
-> 映射回原始 MD 采样
-> 以最后可见 heel TVT 对齐 / 锚定
```

使用过的主要 decoder 为：

| decoder | 做法 | 作用 |
|---|---|---|
| `argmin` | 每个 H 列选取 `abs(SDF)` 最小的 T。 | 保留最直接的局部匹配。 |
| `Viterbi` | cost 由 `abs(SDF)` 减去 segmentation 奖励构成，并加 TVT 跳变惩罚。 | 强制整井路径连续。 |
| `argsg` | 将 argmin 与 segmentation 信息共同用于路径选择。 | 用线概率抑制局部伪匹配。 |
| `dual` | 组合互补的 decoder。 | 在本地 OOF 中最稳健。 |

canonical Geo5 的 v3.1/v3.51 map-average OOF 结果为：

| 输出 | standalone RMSE |
|---|---:|
| argmin | 7.501681 |
| Viterbi original | 7.998088 |
| vit005 | 7.484451 |
| argsg | 7.484158 |
| dual | 7.447392 |

这里的 `dual` 是后期主 GSN 臂。结果也说明“路径更平滑”并不必然更准确：过强的 Viterbi 连续性可能压掉真实的局部地层变化。

## 7. v3.1 / v3.51 集成与后续改进

后期 GSN 使用 v3.1/v3.51 的 map-level ensemble，再通过 dual decode 得到最终路径。该步骤改善的是二维对齐证据本身，而不是仅把两个最终 TVT 序列平均。

Track A 曾进一步加入 RCR（residual correction / ranker）作为 GSN 内部小修正：RCR 单独 RMSE 为 7.414928，`dual` 与 RCR 的组合为 7.236930。冻结五臂系统中只替换 GSN slot 为：

```text
gsn_star = 0.75 * dual + 0.25 * RCR
```

严格 Geo5 CV 从约 6.101265 变为 6.083359。正确做法是把这种改进留在单一 GSN 臂内，再进入严格融合；不要把同一条 GSN/RCR 修正复制到 full249、WARP、HMM 等多条臂，否则各臂残差会变得更相关，最终融合增益可能消失。

## 8. 优点、局限与正确使用方式

| 方面 | GSN 的表现 | 对系统的启示 |
|---|---|---|
| 地层对应 | 直接表示 typewell TVT 与 horizontal MD 的二维匹配。 | 适合处理一个 GR 值对应多个候选 level 的情况。 |
| 路径结构 | SDF + segmentation + decode 同时使用局部和整井信息。 | 比逐行回归更自然地施加连续性。 |
| 多样性 | 与 full249/PF、WARP、HMM 的错误机制不同。 | 即使单臂不必处处最强，也适合独立融合槽位。 |
| 计算量 | 二维 map、五折 checkpoint 和 decode 都较重。 | 部署时需缓存公共输入、field 级集成一次并避免重复解码。 |
| level 风险 | GR 重复纹理会造成错误 level 的连续路径。 | 需要 heel anchor、segmentation、保守 blend 或 gate。 |
| 线上迁移 | 更低 OOF 不必自动等于更好 LB。 | 只采用完整 OOF、统一 Geo5 checkpoint 与训练时相同的解码语义。 |

在最终系统中，GSN 最合适的角色是独立地层路径臂：先用其 map-level ensemble 和固定 dual decode 产生一条 path，再以严格 cross-fitted 权重与 full249、WARP、HMM 或其他真正独立的臂融合。它不应因为局部 OOF 优势而无约束地覆盖安全 anchor。
