# ROGII 比赛复盘总览

## 1. 比赛任务

这是一个井筒地质 / TVT 预测任务。每口水平井在 heel 一侧有一段可见的 `TVT_input`，其余未知段需要预测 TVT。提交文件需要给出每个未知行的 TVT。

项目中的主要建模目标是相对最后一个可见 TVT 的漂移量：

```text
drift_i = TVT_i - last_tvt
TVT_hat_i = last_tvt + drift_hat_i
```

这样可以把已知的起始 level 固定下来，让模型主要学习未来的 level 变化和路径形状。

## 2. 原始数据与输入边界

每口井由两类文件组成：

```text
{well_id}__horizontal_well.csv  水平井沿 MD 方向的序列
{well_id}__typewell.csv         对应的竖直参考井 / 参考地层剖面
```

### 2.1 训练数据的输入与标签

训练集的每口井包含完整水平井记录。模型实际使用的信息如下：

| 文件 / 信息 | 含义 | 训练时用途 |
|---|---|---|
| `MD` | 沿井眼的测深位置 | 序列位置、速度、距 heel 的距离和路径延续。 |
| `X`, `Y`, `Z` | 井眼空间轨迹 | 轨迹导数、相对垂向位置、U-space 和空间先验。 |
| `GR` | 自然伽马测井 | 与 typewell 对齐、PF/likelihood、ANCC/NCC、WARP 和 HMM 的主要观测信号。 |
| 地质/地层列 | 水平井上的 formation 或界面信息 | 地层对应、surface 先验、候选路径约束。 |
| `TVT_input` | 可见 heel 前缀的 TVT；后段为缺失 | 所有模型的起点、heel calibration 与可见历史。 |
| `TVT` | 训练井的完整真实 TVT | 仅作为监督标签、OOF 评估和训练期诊断；不能进入正式推理特征。 |
| typewell 的 `TVT` | 竖直参考剖面的 TVT 坐标 | 建立 `TVT → GR`、`TVT → Geology` 参考曲线。 |
| typewell 的 `GR` | 参考井对应 TVT 的自然伽马 | GR matching、likelihood 和 cross-attention 的参考信号。 |
| typewell 的 geology 列 | 参考地层标签 | formation-contact 与地层一致性特征。 |

训练标签通常写为：

```text
target_drift = true_TVT - last_visible_TVT_input
```

这里的 `true_TVT` 仅能在训练/验证阶段出现；任何使用它构建的统计量都必须在 held fold 外计算。

### 2.2 测试数据实际可见的输入

测试集没有未知段的真实 TVT。每个测试井在推理时可见：

| 可见信息 | 是否可以使用 | 说明 |
|---|---|---|
| 水平井完整 `MD`, `X`, `Y`, `Z` | 可以 | 未知段的井眼几何完整可见。 |
| 水平井完整 `GR` | 可以 | 未知段的 GR 也是输入，不是隐藏标签。 |
| 水平井完整地质/地层日志 | 可以 | 若该列在测试文件中存在，可作为地层证据。 |
| heel 前缀 `TVT_input` | 可以 | 提供 `last_tvt`、heel slope、GR 校准和初始 level。 |
| 未知段 `TVT_input` | 不可以 | 为缺失值，必须预测。 |
| 测试井 typewell 的 TVT / GR / geology | 可以 | 题目直接提供的竖直参考剖面，不是隐藏测试标签。 |
| 训练集学习的模型、normalizer、imputer、先验 | 可以 | 必须只由对应训练 folds 或最终全训练集构建。 |
| 隐藏测试真实 `TVT` | 不可以 | 不可读取、不可推导、不可用于覆盖规则。 |

因此，测试阶段的最基本输入形式为：

```text
可见 heel TVT_input
+ 全段水平井 GR / MD / X / Y / Z / geology
+ 测试井 typewell
+ 训练得到的冻结资产
→ 预测未知段 drift
→ last_tvt + drift = 提交 TVT
```

不允许使用隐藏测试 TVT、面向 Public LB 重叠样本的覆盖规则，或任何由隐藏标签导出的测试特征。

## 3. 验证原则

最终采用的本地验证划分为 `OOF/geo_kmeans_5fold.csv`。每个 held fold 都以整口井为单位排除；用于预测该 fold 的模型、imputer、先验、融合权重和后校准参数均不能见到该 fold。

复盘时必须区分两种分数：

* 早期 full249 分数来自历史 GroupKFold 口径，适合了解发展过程。
* 后期 canonical Geo5 分数来自更严格的统一口径，适合横向比较。

两种口径不能直接互相比较。

## 4. 主要模型臂

| 模型臂 | 主要定位 | canonical Geo5 单臂 RMSE | 保留价值 |
|---|---|---:|---|
| full249 | 249 维 LightGBM drift anchor | 8.149 | 普通井稳定，是后续系统的安全锚。 |
| WARP | GR 序列 U-Net 与 typewell cross-attention | 10.824 | 具有不同的序列形状归纳偏置。 |
| HMM | 基于 GR/typewell likelihood 的状态空间解码器 | 11.044 | 显式表达连续路径与 level 假设。 |
| GSN ensemble | GeoSteerNet 二维 SDF / segmentation 路径臂 | 7.447 | 后期最强的独立模型臂。 |

WARP 和 HMM 的单独 RMSE 更高，并不代表没有价值。它们与 full249、GSN 的残差不同，在严格的 convex blend 中以小权重加入时可以改善总体误差。

## 5. 模型与候选系统笔记

* [full249.md](full249.md)
* [warp.md](warp.md)
* [hmm.md](hmm.md)
* [gsn.md](gsn.md)
* [sg_path.md](sg_path.md)
* [1st.md](1st.md)（最终第一名 Ruby 公开复现方案）

## 5.1 为什么我们的单模没有 Ruby 强

这里比较的是模型结构和信息使用方式，不直接比较不同 CV split 下的数值。Ruby 公开资产中的 `4.80`、`5.09` 等版本 CV 使用其自己的空间邻井划分；本项目后期使用 canonical Geo5。因此，不能将两个 CV 数字的差异全部归因于模型优劣。Ruby 最终 leaderboard `5.639` 仍说明其整体系统在隐藏测试上非常强，但它是七个版本和井级路由的集成，不是单个裸模型。

从根本上，Ruby 将任务写成了条件分布学习：

```text
p(typewell level | horizontal position, GR, heel, geometry, optional XY prior)
```

每个水平井位置面对的是 heel level 附近 400 个密集、固定的 typewell TVT 候选；二维 CNN 为每个候选直接学习 posterior。相比之下，full249 主要学习 row-wise drift 回归，WARP 学增量序列，HMM 使用参数化状态空间，GSN 学 SDF/segmentation，而 sg_path 则先人工产生稀疏的完整候选 path 再选择。我们的系统有丰富证据，但“哪一个 typewell level 是当前对应层位”通常不是端到端的主预测对象。

这种差异带来四个后果：

| 原理层面 | Ruby 的偏置 | 我们单模的限制 |
|---|---|---|
| 候选表示 | 400 个密集 typewell level 同时进入网络，覆盖固定且无 selector 漏选。 | full249 将匹配压缩为 PF/likelihood/beam 特征；sg_path 的候选覆盖依赖 family、grid、cluster 和 top-k。 |
| 监督 | TVT Huber、soft alignment CE、GR penalty 同时监督数值、level posterior 和岩性一致性。 | 多数模型直接监督 TVT/drift；物理一致性常作为手工特征或独立后处理，监督信号更间接。 |
| 空间信号 | 仅在安全井中使用训练邻井的 XY 地质趋势。 | 主四臂以本井 GR、轨迹、typewell 和 heel 为核心，缺少一个成熟、严格门控的空间邻井专家。 |
| 结构学习 | 2D CNN 同时沿 MD 和 candidate-TVT 方向学习连续匹配带。 | full249 是表格/逐行整合；WARP 主要是一维增量；HMM 是固定形式动力学；GSN 最接近但输出/解码目标不同。 |

这不表示 GSN 必然应该改成 Ruby。GSN 的 SDF + segmentation + dual decode 在多模态情况下避免 posterior mean 落入两个真实 mode 中间；Ruby 的 posterior expectation 则在真实后验双峰时可能产生不受任何证据支持的中间 TVT。两者是不同的归纳偏置。

### 可迁移的改进方向

最有价值的不是继续给 full249 添加同族 GR/PF 变体，而是建立一个独立、严格 OOF 的二维 posterior 专家：

```text
heel TVT0
-> typewell 在 TVT0 +/- W 内建立固定、密集 candidate lattice
-> horizontal GR / GR shape / trajectory 与 typewell GR 组成 H x T 图
-> 训练 p(T | H) 的 soft alignment head
-> Huber TVT + soft alignment CE + soft GR consistency
-> 输出 posterior mean、mode、entropy、top1-top2 gap、multi-mode flag
-> 作为独立专家臂或 sg_path 的候选证据，不直接替换安全 anchor
```

空间邻井信息也应是单独、可审计的支路：每个 outer Geo5 held fold 只允许使用其余训练井构建 XY prior；推理时只用训练井和测试井可见坐标/轨迹；再由 visible-prefix replay、邻井距离、路径对齐和外推风险组成 strict safety gate。不得将邻井 prior 扩散到所有模型臂，也不得用 held-fold 标签生成其 OOF 特征。

最终部署仍应遵循本项目已有的线上经验：

```text
安全 anchor 默认输出
-> 仅在 posterior / XY safety / prefix replay 同时可信时启用二维专家
-> 使用 posterior mode、entropy 和分歧避免均值假置信度
-> 以外层 cross-fitted 权重或井级 gate 融合
```

这条路线保留 Ruby 的核心信息表示，同时避免其 posterior mean、空间先验和 CV 语义可能带来的迁移风险。

## 6. PF / likelihood：物理匹配证据支路

PF 是 particle filter，`likelihood` 是观测似然。它们在 full249 中共同回答一个问题：在未知段的某个位置，水平井当前观测到的 GR 最可能对应 typewell 上的哪个 TVT level？

它们不是直接提交的预测器，而是生成路径、level、mode 和置信度特征，交由 full249 的 LightGBM 与其他证据共同判断。

### 6.1 为什么需要 PF

直接以单个 GR 值查 typewell 往往有多个解。例如，typewell 的不同 TVT 位置可能都具有相近 GR。即使某个位置当前 GR 最相似，也未必能和前后位置连接成一条连续、合理的地层路径。

PF 不只判断单行匹配，而是同时保留许多连续路径假设，并让后续 GR 证据逐步淘汰不合理路径。

### 6.2 状态表示与初始条件

为了弱化井眼垂向起伏的影响，PF 在下列空间追踪地层位置：

```text
U = TVT + Z
state = [U, dU/dMD]
```

其中：

* `U` 是结合井眼轨迹后的地层位置。
* `dU/dMD` 是地层沿井眼 MD 的局部延续 rate。

PF 从可见 heel 的最后一行开始：

```text
last_tvt = 最后一个可见 TVT_input
last_U   = last_tvt + last_Z
initial_rate ≈ 可见尾段 median(d(TVT + Z) / dMD)
```

它会在 `last_U` 和 `initial_rate` 周围初始化大量粒子。每个粒子都代表一条可能的未来 level 与 slope 路径。

### 6.3 typewell GR 校准

typewell 给出参考关系：

```text
typewell TVT → expected typewell GR
```

但水平井的实测 GR 与 typewell 的 GR 可能存在整体幅度与偏移差。full249 会仅在可见 heel 上拟合校准关系：

```text
horizontal_GR ≈ a × typewell_GR(TVT_input) + b
```

再以 heel 上的匹配残差估计 GR 噪声尺度：

```text
sigma_GR = std(horizontal_GR - calibrated_typewell_GR)
```

因此 PF 比直接 GR 查表更能容忍仪器偏移、岩性变化和局部幅值差异。

### 6.4 粒子传播

对未知段的每一行，粒子根据 MD 增量推进：

```text
U_i = U_(i-1) + rate_(i-1) × delta_MD + process_noise
TVT_i = U_i - Z_i
```

这产生很多可能的候选 TVT。粒子并不会立刻只保留 GR 最像的一条，而是保留多个可行路径，等待后续序列证据。

### 6.5 likelihood：如何用 GR 给粒子加权

对每个粒子的候选 TVT，在 typewell 曲线上插值：

```text
expected_GR_i = a × typewell_GR(candidate_TVT_i) + b
residual_i = observed_horizontal_GR_i - expected_GR_i
```

默认的 Gaussian likelihood 可写为：

```text
likelihood_i ∝ exp(-0.5 × (residual_i / sigma_GR)^2)
```

含义是：候选 TVT 所对应的 typewell GR 越接近当前水平井 GR，该粒子权重越高；差异越大，权重越低。代码也曾支持 Student-t、Cauchy 与 Huber 等重尾 likelihood 消融，用于降低异常 GR 点的影响。

### 6.6 归一化与重采样

每一步的粒子权重更新为：

```text
new_weight = old_weight × likelihood
```

归一化后，如果有效粒子数过低，说明只剩很少的路径仍合理，就进行 resampling：

```text
高权重粒子被复制
低权重粒子被淘汰
复制后的粒子加入轻微位置与 rate 扰动
```

这样可以把计算资源集中在仍被 GR 证据支持的 level/path 假设上。

### 6.7 PF 输出给 full249 的内容

粒子加权均值可产生一条 PF 路径，但 full249 不会盲目提交它。PF/likelihood 支路会导出例如：

```text
PF level 与 drift 候选
校准与未校准 GR 下的路径
多 seed PF 的均值、spread 和 disagreement
posterior mode 的位置、质量、间隔和权重
PF 与 beam、ANCC/NCC、其他候选路径的差值
likelihood 强度和模态分离置信度
```

这些特征让 LightGBM 能学习：PF 何时可靠、何时有多解、何时应相信其他路径或 anchor。

### 6.8 与其他方法的关系

```text
likelihood：某个 TVT 假设在当前 GR 下是否合理
PF：沿全段递推并维护多个连续 TVT 假设
beam：显式保留多条高分完整候选路径
ANCC/NCC：比较一段 GR 形状，而不是单点幅值
LightGBM：综合这些证据及其分歧，预测最终 drift
```

因此，PF 是 full249 的重要物理证据支路，不是一个取代 LightGBM 的单独最终模型。

## 7. sg_path 与候选证据

`sg_path` 是围绕 anchor 构造整井候选路径的系统。其候选来源包括：

```text
anchor path
level-grid 偏移
分段 U = TVT + Z 路径
可见前缀 U-rate 外推
同井 GR 匹配
formation / surface 假设
```

候选路径按 level/mode 聚类，再使用可观测的 GR、前缀连续性、路径平滑度和 member 证据打分；部分版本还使用 segmented posterior 得到合成路径。最终可输出候选移动量、候选分歧、level 改变量和 posterior 不确定性等 sidecar 特征。

比赛中最重要的发现是：候选池的 oracle 往往很好，但在线 selector 的迁移能力远弱于 oracle。因此候选证据更适合作为特征或受限残差修正，而不适合不加约束地替代 anchor。

## 8. ModelB 与路由

ModelB 是二阶段 LightGBM。它输入 anchor 与 SG/member 候选 sidecar，可显著修复部分 hard well。但 direct ModelB 或大幅候选 posterior 改写常会破坏 Public LB 上原本已预测良好的井。

较稳健的结构是：

```text
稳定 anchor 默认输出
→ 使用分歧和置信度判断风险
→ 仅在证据充分时施加有限修正
```

Pair-gate、Scheme2、稀疏路由和 anchor-default residual 都是这一原则的不同实现。

## 9. 后处理

后处理不产生新的地质信息，作用是约束和校准模型输出。

| 方法 | 作用 |
|---|---|
| row-wise Winsorization | 限制某一模型臂相对整体模型臂集合的过大偏离。 |
| convex blend | 在训练 folds 上学习非负且和为一的权重。 |
| odd cubic calibration | 校正整体幅度与非线性漂移，同时保持正负方向行为。 |
| U projection | 在 `U = TVT + Z` 空间中处理路径，使延续性更符合几何含义。 |
| F-rectification、SG smoothing、MD ramp、chord shrink | 抑制不合理的局部振荡、突变和过度形状修正。 |
| gate / route | 默认保留 anchor，只在可观测分歧说明可能是 hard well 时切换或修正。 |

后处理能够改善校准并保护好井，但无法挽救在线选择语义不稳定的候选模型。

## 10. 代表性融合结果

一个严格 Geo5 四臂 convex blend 的平均权重约为：

```text
full249       28.3%
WARP           7.2%
HMM           12.0%
GSN ensemble  52.6%
```

该方案在严格 weight-crossfit 下的 RMSE 约为 `6.69`。后续加入 FG2/A2/M2、GSN sidecar 和后处理后，冻结五臂系统的正式本地 CV 约为 `6.08`；但本地下降并不总会按比例迁移到 leaderboard。

## 11. 核心经验

1. hard well 带来的本地 CV 改善，不等于 Public 或 Private LB 改善。
2. anchor 的价值在于保护已经预测良好的普通井。
3. 正交模型臂最适合以小权重、严格 cross-fit 的方式进入融合。
4. 候选池质量与候选 selector 质量是两个不同问题。
5. 每个学习型 sidecar 和每个融合权重都必须满足严格 OOF 语义。
6. 最稳健的整体结构不是无限制的超级模型，而是稳定默认预测加上有证据约束的修正。

## 12. 比赛复盘：事实、机制与下一次的判断框架

### 12.1 比赛真正难点是什么

这不是普通的逐行回归任务。真正困难的部分是：同一个 horizontal GR 局部纹理可能对应 typewell 上多个 TVT level；但提交要求的是一条整井连续、绝对 level 正确的 TVT path。

| 实验事实 | 为什么会这样 | 背后的通用原理 | 以后看到什么信号应想到它 |
|---|---|---|---|
| PF、GR matching、beam、sg_path 的候选 oracle 常显著优于可部署 selector。 | 正确路径常在候选集合中，但不同 level 的重复 GR 纹理使在线评分无法可靠区分它们。 | **candidate coverage 不等于 candidate identification**；有 oracle gap 时，问题是后验多模态和选择，而不只是候选不够多。 | “top-k 里有正确答案，但 top-1/selector 不稳定”；“更多候选令 CV 更低、LB 更差”。 |
| 强 direct ModelB、大 KEEP 或 unrestricted posterior 能修复 hard well，却经常损害 LB。 | pooled RMSE 被少数灾难井的 SSE 强烈主导；模型为了降低 CV 学会大幅改写，但普通井的轻微改坏在线上累计更多。 | **平均回归风险与条件修正风险不同**；默认安全模型与困难样本专家必须分角色。 | “某模型只在误差最大的井上收益巨大”；“CV 大降但好井 bucket、Public 或多提交不升”。 |
| Ruby、AnchorCNN、GSN 都把问题显式表示在 horizontal MD x typewell TVT 的二维空间。 | typewell level 是问题的关键隐变量；只输出一个 TVT 标量会过早压缩多个可行 level 假设。 | **先选择正确的状态空间，再选择预测器**。正确的表示通常比在错误表示上堆特征更重要。 | “参考剖面存在重复形态”；“局部匹配有多个峰”；“输出必须是连续路径而非独立标量”。 |

因此，比赛的本质是：在可见 heel、horizontal GR、轨迹和 typewell 条件下，对整井的多模态层位后验做受约束推断；而不是单纯把单行 TVT RMSE 压低。

### 12.2 最终方案为什么有效

这里的“最终方案”包括本项目后期较稳健的 anchor-plus-expert 结构，以及最终榜首/第二名方案共同证明的有效原则。

| 实验事实 | 为什么会这样 | 背后的通用原理 | 以后看到什么信号应想到它 |
|---|---|---|---|
| full249/exp265 等 anchor 在普通井、好井 bucket 和若干 LB 提交中更稳定；直接专家常更激进。 | 普通井不需要复杂的候选改写，锚点的偏差小且方差低。 | **先验默认值应承担大多数样本的风险**；复杂模型只应负责条件残差。 | “简单模型的 Q1--Q3 更好”；“更强模型的改善集中于少数井”。 |
| GSN、WARP、HMM 虽单臂较弱，严格融合仍有增益。 | 它们错误来源不同：表格物理证据、序列 shape、状态空间连续性、二维地层对齐分别覆盖不同失效模式。 | **融合价值由残差相关性决定，不由 standalone RMSE 单独决定**。 | “弱模型与强 anchor 的 residual correlation 低”；“小权重比独立替换更稳”。 |
| Ruby 只在空间邻井安全时启用 XY expert；本项目 pair-gate/Scheme2 也比 direct ModelB 更能迁移。 | 外推、邻井错配或候选多解时，激进证据的风险高于收益。 | **gate 的价值是风险控制，不是追求覆盖率最大化**。 | “专家在部分 well class 很强、其余类别不稳定”；“可观测分歧、prefix replay、距离或外推分数存在”。 |
| AnchorCNN 最终使用 DP marginal expectation；第二名的 reranker、rollout selector 和 RL 都未稳定超过它。 | 多条候选路径的 oracle 很强，但学习哪条路径最好容易对少数验证井过拟合；边缘化比过早离散决策稳定。 | **在可信概率模型下，先边缘化不确定性，后做保守决策**。 | “bundle oracle 远好于 selector”；“reranker CV 波动大或 LOWO 后收益消失”。 |

最终有效的不是某一个 magic feature，而是以下组合：安全 anchor、真正正交的路径臂、显式不确定性、严格 OOF、以及对专家改写范围的控制。

### 12.3 最重要的三个实验

#### 实验一：direct ModelB / 大候选 posterior 与 anchor-default gate 的对照

```text
实验事实：direct ModelB、较大 KEEP 和全候选 posterior 往往显著降低本地 hard well CV；
          pair-gate、Scheme2、WARP anchor 和 cap 的本地 CV 未必最低，却更常获得可迁移的 LB。
↓
为什么：direct 形式将 hard-well SSE 当作最重要信号，系统性改变大量好井；gate 将问题改为“是否允许离开 anchor”。
↓
通用原理：面对异质样本，先训练或设计 conditional action，再训练强回归器；不能把 pooled gain 当作全体样本的收益。
↓
下次信号：若改动井覆盖率很高、修改幅度大、收益集中在 top-error wells，应优先做 gate/route/cap 消融，而不是继续加大模型容量。
```

#### 实验二：统一 Geo5 OOF 资产审计与重建

```text
实验事实：旧缓存、不同 fold map、full-train sidecar、候选标签与 OOF 语义混用，会造成 CV 之间无法比较；
          后期 Geo5 重建把模型、imputer、UPF、WARP、GSN 和融合口径统一后，才具备可信的实验面板。
↓
为什么：二阶段模型特别容易把 in-fold 预测质量、held-fold 标签信息或全训练统计误当作可部署信号。
↓
通用原理：在 stack、selector、neighbor prior、post-calibration 中，验证数据流比模型层数更重要。
↓
下次信号：任何 feature 无法说明“由哪些训练井生成、何时 fit、对哪个 held fold 预测”，就不应进入模型比较或权重拟合。
```

#### 实验三：独立 GSN 路径臂与严格融合

```text
实验事实：GSN dual 的 standalone Geo5 RMSE 为 7.447392，
          与 full249/WARP/HMM 的错误不完全相同；严格 convex blend 能显著优于单臂。
↓
为什么：二维 typewell-TV​​T x horizontal-MD 对齐提供了表格回归和一维模型没有的候选 level 几何。
↓
通用原理：先寻找新的、可验证的观测机制，再讨论复杂 stacking；独立信号通常比同族特征微调更有上限。
↓
下次信号：新模型能改变错误井集合、残差相关性低、且每个 fold 有稳定收益时，应作为独立融合臂；
          若只是复制其他臂的修正，则不应扩散到所有链路。
```

### 12.4 最大的三个错误判断

#### 错误一：把 pooled CV 的持续下降视为主要目标

```text
实验事实：KEEP60、富 sidecar、A2/M2/后处理等多次取得很低本地 CV，
          但 leaderboard 改善不成比例，部分方案甚至弱于更简单的 exp265/exp280 类路径。
↓
为什么：hard well 对 RMSE 的平方损失贡献过大；局部 CV 改善可以来自少数井的大幅修复，掩盖多数普通井的轻微退化。
↓
通用原理：对于 heavy-tail、well-level heterogeneous 的指标，必须同时报告 per-well RMSE、bucket、LOWO/LCO、改动井和未改动井表现。
↓
以后：只要一个 gain 主要由少量井贡献，就要求 leave-largest-contribution-out 或 bootstrap well-level 稳定性通过后再推广。
```

#### 错误二：过度相信 candidate oracle 可以被在线 selector 恢复

```text
实验事实：sg_path、member selector、cluster selector、reranker 和多路径 posterior 的 oracle 往往很强，
          但 direct selection、宽 posterior 和 rerank 在 LB 上不稳定。
↓
为什么：oracle 使用真实 TVT 事后区分近似路径；线上 GR、heel 和轨迹不足以稳定区分这些多模态解释。
↓
通用原理：oracle gap 是可识别性上限的证据，不是“再训练一个 selector 就能得到的免费收益”。
↓
以后：先测 online score 与 oracle rank 的关系、top1-top2 gap 校准和 family-independent agreement；
      若关系弱，则输出不确定性、做边缘化或保持 anchor，而不是继续扩大候选池。
```

#### 错误三：过晚把完全 OOF 的资产语义和空间信息放到第一优先级

```text
实验事实：多个阶段需要重新审计 Geo5 fold、UPF、WARP、GSN、sidecar 和 cross-fit 权重；
          同时，最终第一名证明了 gated XY-neighbor prior 可以是高价值独立信号，第二名则证明它不应被无条件相信。
↓
为什么：早期花费较多时间在同族 GR/PF 分数、局部后处理和叠加特征上，而这些信号共享相似的失败模式；
          OOF provenance 不统一也放大了局部比较的不确定性。
↓
通用原理：先固定数据流与验证协议，再寻找条件独立的新信息源；空间邻井信号需要被当作高风险专家而不是普通特征。
↓
以后：训练前先建立 asset manifest、fold hash、fit/predict lineage 和 outer-fold neighbor exclusion；
      对任何 XY prior 先做安全性 gate 与 replay 消融，再允许进入主模型。
```

### 12.5 如果重新开始，会先做什么

1. **先冻结验证与审计基础。** 使用 canonical Geo5，从第一天起保存 fold hash、每个资产的 train-fold provenance、OOF prediction ID 和所有 fit statistic 的来源；同时建立 pooled、per-well、bucket、LCO/LOWO 四层报告。

2. **建立三个小而清晰的 baseline。**

```text
A. 安全 anchor：heel + trajectory + 基础 GR/typewell 的 full249 风格模型
B. 二维 posterior：H x T 对齐 lattice，输出 level posterior、mode、mean、entropy
C. 物理路径：U-space / PF 或 HMM，但不将其直接当作最终提交
```

每个 baseline 必须先有完整 Geo5 OOF，再讨论融合。

3. **优先验证两个正交信息源。**

```text
本井 pre-PS 高分辨率 GR reference
受严格 outer-fold 排除与 safety gate 控制的 XY-neighbor prior
```

它们都应作为独立专家或二维 posterior 的输入，不能在尚未验证安全性时扩散到所有臂。

4. **将“选择”推迟到最后。** 先从 posterior mean/marginal、严格小权重 blend 和 anchor-default residual 开始；只有当 selector 的 online rank、LOWO/LCO 和跨 fold 覆盖都稳定时，才引入 hard switch、pair-gate 或 reranker。

5. **以线上迁移为晋级标准。** 新方案需要同时满足：完全 OOF、改动范围可解释、好井不恶化、收益不依赖极少数 catastrophic well、且与现有臂残差具有独立性。CV 更低但不能通过这些检查的方案，只保留为研究资产，不进入正式部署。
