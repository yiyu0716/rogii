# HMM

## 定位

HMM 是显式状态空间 geosteering decoder。它在离散 TVT grid 上选择一条连续路径，并把 GR/typewell 一致性作为观测 likelihood。

与 full249 的多特征 row-wise 回归不同，HMM 直接表达以下物理假设：

```text
路径应在 TVT / U-space 中连续演化
且该路径隐含的 typewell GR 应与水平井 GR 匹配
```

## 状态和转移

未知段每一行都有一组候选 TVT 状态，以及有限的路径 rate 状态。canonical exp224 风格配置大致为：

```text
TVT grid step: 0.2 ft
level 搜索带: last_tvt ± 150 ft
rate 状态数: 41
rate span: 约 ±0.10
rate momentum: 0.998
position process scale: 0.02
rate process scale: 0.002
```

transition cost 会惩罚不合理的局部 rate 改变，避免路径在重复 GR 匹配之间随意跳跃。

## emission / 观测 likelihood

对每个候选 TVT 状态：

```text
expected_GR = typewell_GR(candidate_TVT)
emission score = horizontal GR 与 expected_GR 的
                 Gaussian 或 Student-t 兼容度
```

GR 噪声尺度通过可见 heel 的 raw typewell mismatch 测量并裁剪。一些版本会拟合稳健的 heel affine calibration：

```text
horizontal_GR ≈ a × typewell_GR + b
```

保守版本在 calibration 后仍保持 raw heel 噪声尺度，不会通过缩小 sigma 制造虚假的高置信度。

## 解码

forward-backward 或 Viterbi 类动态规划综合：

```text
GR/typewell 的 emission compatibility
+ rate 模型给出的 transition continuity
+ 最后可见 heel TVT 与尾段趋势给出的 start prior
```

输出是整段 TVT/drift 路径；部分版本还输出 posterior 不确定性或状态边缘概率。

## 训练和推理

核心 HMM 不是高容量监督神经网络。它主要由固定物理参数、GR/typewell 输入与 heel calibration 决定。canonical decoder 可通过 CPU/Numba 动态规划运行，不需要 GPU。

但用于 HMM 的 fold 内 calibration、学习型辅助量与融合权重仍必须严格排除 held fold。

## 结果与作用

canonical `hmm_exp224` 的 Geo5 standalone RMSE 约为 `11.04433`。它单独明显弱于最佳 tabular 或 GSN 模型臂，但能提供物理约束下的整井路径假设。在 full249/WARP/HMM/GSN 严格融合中，HMM 平均获得约 `12%` 权重。

这说明单臂 RMSE 较高的模型，若残差不同，也可能在小的 cross-fitted 权重下改善融合。

## 优点

| 优点 | 具体含义 |
|---|---|
| 整井连续性 | transition model 约束相邻位置，避免逐行 GR matching 在相似层位间任意跳跃。 |
| 物理可解释 | 每个路径分数可拆为 heel 起点、GR/typewell emission 和 rate/跳变惩罚。 |
| 可表达不确定性 | forward-backward 可得到 posterior；多个高分状态意味着 level 仍存在多解。 |
| 不依赖大规模标签学习 | 核心能力来自状态模型和观测模型，适合监督数据有限或希望保留物理约束的情形。 |
| 正交残差 | 即使单独 RMSE 较高，也可能与 tabular 或 neural arm 犯不同的错，因此适合小权重融合。 |
| 部署简单 | canonical decoder 可用 CPU/Numba 动态规划运行；推理只需要可见 heel、全段日志和 typewell。 |

## 缺点与失败机理

| 缺点 | 为什么会发生 | 常见表现 |
|---|---|---|
| level 多解 | 不同 TVT 层位可能有相似 GR motif。 | GR 匹配看似很好，但整口井整体偏移。 |
| 路径锁定 | 早期选错分支后，强 transition 连续性会让错误沿后续序列累积。 | 曲线平滑、视觉合理，但 level RMSE 很高。 |
| typewell 对应偏差 | 参考井与目标水平井可能存在厚度、相位、岩性或断层差异。 | 全段都朝同一方向偏，单纯平滑无效。 |
| 观测模型过度自信 | heel affine calibration 后若过度缩小 sigma，会把局部 GR 对齐误当成确定地层对应。 | 本地少数井提升很大，跨 fold 或 LB 不稳定。 |
| 状态模型过于简单 | `[TVT level, dU/dMD]` 难以表示断层、regime change、厚度突变和复杂横向变化。 | 局部真实跳变被抹平，或只能缓慢走向错误 level。 |
| 信息源较窄 | 核心依赖 GR/typewell，无法自然利用所有 surface、formation、PF、beam 和候选分歧证据。 | standalone 通常弱于多证据 LightGBM 或 GSN。 |

canonical `hmm_exp224` 的 Geo5 误差结构也符合这一判断：整体 RMSE 约 `11.04433`，其中 level RMSE 约 `8.62189`，表明主要瓶颈是绝对 level 识别，而不仅是局部 shape 噪声。

## 什么时候应该采用 HMM

HMM 适用于以下类别的问题：

| 问题特征 | 为什么 HMM 合适 |
|---|---|
| 隐藏状态沿序列连续演化 | 例如地层位置、设备健康状态、目标轨迹、语音状态、市场 regime。transition model 能利用连续性。 |
| 每个位置有可比较的观测参考曲线或模板 | 例如 GR 对 typewell、传感器读数对物理模型、波形对模板库。emission 可定义为匹配 likelihood。 |
| 单点观测有噪声，但相邻点共同判断更可靠 | HMM 通过整段证据减少单点匹配的偶然性。 |
| 需要可解释的路径和不确定性 | 可以检查起点、transition、emission、posterior mode，而非只有黑箱回归值。 |
| 已知明确的物理平滑先验 | 例如最大变化率、合理速度、状态切换成本或连续空间路径。 |
| 监督标签有限，但有领域观测模型 | HMM 不必完全依赖大量端到端标注数据。 |

在本比赛中，最适合 HMM 的角色是：

```text
GR/typewell 驱动的连续路径专家
→ 输出独立 drift path、level、slope、posterior 与不确定性
→ 作为 full249 / GSN 的小权重融合臂或 sidecar
```

## 什么时候不应直接依赖 HMM

以下信号出现时，HMM 不应作为无约束主预测：

| 观察到的信号 | 应采取的策略 |
|---|---|
| GR 匹配高但 level error 高 | 认为存在重复 motif / 多解；增加独立 level 证据，而不是只加强 emission。 |
| 路径很平滑但整井整体偏移 | 检查 datum、heel 起点和 mode 分支；不要继续提高平滑惩罚。 |
| typewell 与水平井存在明显横向岩性变化 | 使用 HMM 作为弱专家，并加入 calibration 不确定性或其他模型的 gate。 |
| 真实过程经常有断层或 abrupt regime change | 放宽 transition，加入 change-point 状态，或改用多路径候选与 router。 |
| HMM 单臂很差但与主模型低相关 | 保留小的 strict cross-fitted blend 权重，而不是直接删除或大权重替换 anchor。 |

## 可复用的判断原则

以后看到以下现象，应再次想到 HMM 的这些原则：

```text
局部观测匹配很好，但隐藏状态仍不唯一
→ 这是状态不可识别 / 多模态后验问题。

路径平滑且视觉合理，但整体 level 错误很大
→ 这是早期分支错误被连续性先验锁住的问题。

校准后观测残差变小，但跨场景表现变差
→ 校准修正的是测量域，不一定修正状态语义或地层对应。

单臂较弱，却在严格融合中始终有小正权重
→ 它可能是正交专家，应受限使用而不是成为主模型。
```
