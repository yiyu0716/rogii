# sg_path：候选整井路径与在线证据系统

## 1. 定位

`sg_path` 不是一个直接预测 TVT 的端到端模型，也不是一条固定的最终路径。它是一套围绕安全 anchor 构造候选整井路径、筛选候选解释、量化不确定性，并将这些结果提供给 ModelB、pair-gate 和 Scheme2 的 online sidecar 系统。

它要解决的问题是：当 anchor 在一口困难井上可能处于错误 level 时，是否存在由 GR、typewell、轨迹和地层先验共同支持的另一类连续路径解释；同时，系统必须让下游模型知道这个解释究竟是“多源一致的强证据”，还是“多个相似候选堆出来的假置信度”。

```text
安全 anchor
+ 多来源候选整井路径
-> 规范化与聚类
-> SG Cluster Selector
-> Member Selector
-> Diversity Retention
-> sg_path / posterior / 分歧 sidecar
-> ModelB、pair-gate 或 Scheme2 的受限修正
```

测试阶段只使用水平井完整 GR/轨迹/地层日志、可见 heel `TVT_input`、测试 typewell 和训练得到的模型资产，因此候选本身可做到 Private-safe。任何学习式 selector 的训练与评分都必须使用对应 outer Geo5 fold 的 OOF 输出。

## 2. anchor 与候选池

每口井先有一条默认的、安全 anchor，例如 full249、full249 + WARP、exp249、exp265 或后期的保守融合路径。anchor 的职责是保护普通井已经正确的 level；`sg_path` 的职责是提出可检验的替代解释，而不是默认推翻 anchor。

候选以完整未来段路径的形式构造，曾使用的主要来源包括：

| 候选来源 | 构造思想 | 提供的证据 |
|---|---|---|
| anchor path | 默认预测路径本身。 | 安全基准和零移动假设。 |
| level-grid | 对 anchor 作不同的整井或局部 TVT offset。 | 检验错误是否主要来自 level 偏移。 |
| U-space 路径 | 在 `U = TVT + Z` 空间中构造分段连续路径。 | 弱化井眼垂向起伏，表达地层延续。 |
| heel U-rate 外推 | 用可见 heel 尾段的 U-rate / slope 延续到未来。 | 利用局部历史趋势。 |
| GR matching 路径 | 用 horizontal GR 与 typewell GR 的局部或序列匹配产生候选。 | 提供独立的岩性对应证据。 |
| formation / surface | 结合地层界面、surface 或 level 先验生成路径。 | 提供地质结构约束。 |
| 模型路径 | 引入 WARP、HMM、GSN 等独立臂及其路径变体。 | 补充不同归纳偏置下的整井解释。 |

候选只要能由线上可见输入生成，就可以作为 online candidate；训练时只能使用 held fold 外训练得到的生成器、imputer、selector 和特征统计。

## 3. 为什么先做候选聚类

候选池里会产生许多几乎相同的路径。例如同一条错误解释的多个 1--3 ft 平移版，可能在 GR 分数上都很高。若将它们直接相加为 posterior 支持，就会因为重复数量而夸大该解释的置信度。

因此先根据整井 path 距离、平均 level delta、局部 shape 差异和分段行为聚类：

```text
候选路径集合
-> 以 level / shape / path distance 聚类
-> 每个 cluster 代表一类独立地层解释
-> 先评估 cluster，再挑选其中的 member
```

这一步使“多个近似变体”与“真正互相独立的地层假设”分开。后续分歧、entropy 和 posterior 特征才有可信语义。

## 4. SG Cluster Selector

SG Cluster Selector 的决策对象是候选 cluster：它评估一类地层解释是否值得下游系统关注，而不是直接输出最终 TVT。

典型实现是 ExtraTrees 与 LightGBM 的 selector，训练目标使用训练阶段可计算的 cluster oracle 质量，例如：

```text
target = exp(-cluster_oracle_RMSE / 8)
```

这里的 oracle RMSE 只可在训练标签中构造，不能在测试井上计算。线上 selector 只接收可见证据，主要包括：

```text
GR / typewell 匹配质量、ANCC/NCC、likelihood、PF 支持
可见 heel 前缀 replay 误差与校准质量
候选的 U-space 连续性、slope、curvature、jump
formation / surface 一致性
候选与 anchor 的 level / shape 差异
cluster 内 spread、cluster 间分歧和不同来源支持
WARP、HMM、GSN 等独立路径是否支持该解释
```

输出是 cluster 分数、排序、top cluster 的支持强度及其与其他解释的差异。这些都是后续模型识别 hard well 和候选多解的重要信号。

## 5. Member Selector

同一 cluster 内可能仍有多个可行成员，例如不同 offset、切点、平滑尺度或局部 rate 假设。Member Selector 的任务是从同一类解释中找到具体最可信的路径成员。

你的典型版本是约 118 个特征的 LightGBM，训练目标近似为：

```text
target = exp(-candidate_absolute_level_error / 6)
```

member-level 输入包括：

```text
候选 GR match / NCC / likelihood
heel history replay 一致性
offset、slope、curvature、U-rate
与 cluster center 的距离
与 anchor、WARP、HMM、GSN 的差异
局部支持度、边界距离、family 来源和质量
```

它输出 member rank、score、top-k member path 和选中成员相对 anchor 的移动量。训练时它必须对 held fold 使用真正 OOF 的候选和预测，不能将训练井的 in-fold 候选打分结果当作 OOF 特征。

## 6. Diversity Retention：避免 posterior 坍缩

若只保留分数最高的成员，top-k 往往来自同一来源、同一 level 或同一近似路径，导致 posterior 看似非常集中，却没有真正的独立支持。

Diversity Retention 在候选保留时施加约束，例如：

```text
TOPK                 总保留数量，例如 60
score head           优先保留的高分候选数量，例如 12
family quota         每个候选来源 family 的最大数量，例如 6
level-delta bin      按相对 anchor 的 level delta 分箱，例如 6 ft
```

实际含义是：高分候选先进入，但同一家族或同一 level 区间不能无限重复占满候选池；不同来源、不同 level 解释都要保留一部分。这样下游可观察到：

```text
多个独立 family 在同一 level 达成一致
-> 更强、更可信的候选证据

多个 family 指向不同 level，或只有单一 family 支持
-> 多解 / 不确定，应保护 anchor
```

## 7. 输出给 ModelB 与 gate 的 sidecar

`sg_path` 的产物不止一条 top-1 路径。它会生成逐行与整井两类 sidecar：

| 类型 | 主要输出 |
|---|---|
| 路径 | top member、top cluster center、posterior mean / median / mode 的 TVT path。 |
| 相对移动 | 各候选、posterior 和 top path 相对 anchor 的 delta。 |
| 不确定性 | spread、entropy、multimodality、cluster 数、top1-top2 gap。 |
| 支持度 | family 支持比例、cluster size、GR/PF/heel replay 分数。 |
| 分歧 | 候选之间、候选与 anchor、候选与 WARP/HMM/GSN 的 level 与 shape disagreement。 |
| 选择质量 | cluster/member rank、score、coverage 与是否满足保留约束。 |

ModelB 用这些特征估计“当前 anchor 是否可能需要修正，以及修正多大”；pair-gate/Scheme2 用井级分歧和改写幅度决定是否允许整井应用该修正。

## 8. 最大优势

`sg_path` 最大的优势不是给出一条看起来最好的路径，而是把原本不可见的候选空间显式化为可学习、可审计的不确定性证据：

```text
它既告诉下游“替代路径是什么”，
也告诉下游“有多少独立证据支持它，以及候选是否存在多解”。
```

普通 row-wise 回归模型很难自然表达“两个 TVT level 都能匹配当前 GR”。`sg_path` 通过整井候选、cluster 和 diversity-retained posterior，使 ModelB/gate 有机会只在强证据 hard well 上修正 anchor，同时在普通井上保持不动。

## 9. 优点与缺点

| 方面 | 优点 | 缺点 / 风险 |
|---|---|---|
| 整井结构 | 候选是连续 path，而非相互独立的行级预测。 | 候选生成质量不足时，后续 selector 无法凭空创造正确路径。 |
| 多解表达 | cluster、posterior、entropy 能显式表达 level ambiguity。 | 大量近似候选会产生假置信度，必须做聚类和 diversity retention。 |
| 信息融合 | 可结合 GR、typewell、PF、U-space、formation、独立模型路径。 | 输入来源多，特征语义和 OOF provenance 容易不一致。 |
| hard well 救援 | oracle candidate 常可大幅改善困难井。 | oracle 与线上 selector 存在明显迁移落差；不能把 oracle 改善当成可部署收益。 |
| 部署安全 | 生成过程只依赖线上可见数据时可 Private-safe。 | learned selector、候选标签、imputer、后校准均必须严格 outer-fold OOF。 |
| 融合 | 作为 sidecar 可提供与 anchor 不同的证据。 | direct sg_path 或 unrestricted posterior 容易系统性改坏原本良好的井。 |

## 10. 正确的部署方式

比赛经验表明，最稳定的结构不是：

```text
top candidate / posterior path -> 直接提交
```

而是：

```text
anchor 作为默认输出
-> sg_path 提供候选、分歧和置信度特征
-> ModelB 学习受限 residual 修正
-> pair-gate / Scheme2 仅在井级证据充分时允许改写
-> cap、shrink 或保守融合限制异常偏离
```

direct candidate、direct ModelB 或全候选 posterior 往往能显著降低本地 hard well 的误差，却会在 Public/Private 分布中轻微而系统性地破坏好井。因而 `sg_path` 的最佳角色是候选特异性证据系统，而非无条件替代 anchor 的预测器。
