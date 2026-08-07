# WARP

## 定位

WARP 是从水平井 GR 序列中学习路径形状、并以 typewell 为条件的序列模型臂。它不是 full249 的 tabular 替代品，而是提供不同归纳偏置的正交模型臂。

## 核心结构

canonical 部署实现的结构为：

```text
15 通道 GR / 轨迹序列特征
→ TypewellCrossAttention
→ 一维 SCA-U2Net encoder-decoder
→ 预测增量 drift
→ cumsum 积分
→ TVT / drift 路径
```

U-Net 使用嵌套 residual U-block、多尺度 skip connection 和 attention。cross-attention 使每个水平井位置可以查询 typewell token 序列；输出头先预测增量，再经 `cumsum` 积分成整段路径。

## 输入

纯 GR WARP 的部署特征为 15 个通道：

```text
GR
GR 一阶导数
GR 二阶导数
GR envelope
短/中窗口 GR 平滑
局部 GR 标准差
相对 Z
dZ/dMD、dX/dMD、dY/dMD
距 heel 的 MD
归一化序列进度
距 heel 的水平距离
azimuth
```

typewell 分支把 typewell 重采样为约 192 个 token：

```text
[resampled typewell GR, typewell TVT - last_tvt]
```

未知未来 TVT 不作为输入。

## 训练和推理

WARP 采用按井 held-out 的训练方式，每个 fold 保存自己的 normalization 参数。推理时，将测试井转换为 GR/轨迹 tensor 和 typewell token，分别经过保存的 fold 模型后取平均。

神经网络推理适合 GPU，但特征构造和积分是确定性的，仅依赖可见测试输入与 typewell，因此可以部署。

## 结果与作用

canonical `warp_exp207` 的 Geo5 standalone drift RMSE 约为 `10.82410`，显著弱于 full249 与 GSN。但它的残差并不相同，因此在严格的 full249/WARP/HMM/GSN convex blend 中平均仍获得约 `7%` 权重。

WARP 的价值不是“自身最准确”，而是提供一个不同的序列形状意见。

## 最大优势

WARP 最大的优势是：它能在 typewell 条件下，直接从整段水平井 GR 序列学习多尺度的相对形状和连续变化，而不是把每一行当作独立样本。

full249 主要通过 row-wise 特征和 LightGBM 判断当前行应处于哪个 drift；HMM 主要通过显式 GR/typewell likelihood 与平滑 transition 解码。WARP 则能同时看到附近 GR 峰谷、导数和局部 motif，较长范围的 GR 重复结构，MD/Z/轨迹变化和序列进度，以及 typewell 中多个可能相关的区域。

cross-attention 不要求先指定唯一的 typewell TVT 对齐点，而是让每个水平井位置从 typewell token 中学习性地提取相关上下文。因此，当局部 GR 形状、相对层序和路径变化比绝对 datum 更有信息时，WARP 可以给 full249/GSN 提供真正不同的 shape 证据。

## 优点

| 优点 | 具体含义 |
|---|---|
| 多尺度序列建模 | U-Net、pooling、skip connection 和 dilated block 同时捕获局部 GR motif 与长程结构。 |
| typewell 条件化 | TypewellCrossAttention 让不同水平井位置查询不同 typewell 区域，而不是只做固定单点插值。 |
| 连续路径归纳偏置 | 先预测增量再做累积积分，天然把输出组织成一条连续路径。 |
| 输入较干净 | 主要使用 GR 与轨迹 15 通道，不依赖完整的高维 tabular 特征或复杂候选选择器。 |
| 残差正交 | 与 full249、HMM、GSN 的错误来源不同，因此即使单臂较弱，也能在严格融合中提供小幅增益。 |
| 可端到端部署 | 测试时只需要可见 heel、全段日志、typewell、fold-specific normalization 和保存的 checkpoint。 |

## 缺点与失败机理

| 缺点 | 为什么会发生 | 常见表现 |
|---|---|---|
| level 锚定弱 | 网络主要从 GR shape 和相对变化学习，绝对 datum 证据弱于 full249 的多源 level 特征。 | 整段曲线形状像，但整体上下偏移。 |
| 累积偏差 | 每一步微小的 increment bias 会沿未知段不断累积。 | 越到未知段末端，level error 越大。 |
| GR motif 多解 | 相同或相似 GR shape 可在 typewell 的多个层位出现。 | cross-attention 找到合理参考片段，但并非正确绝对 TVT。 |
| 序列分布敏感 | 网络学习到的 GR motif、长度和轨迹关系可能不适用于新井。 | 本地 CV 改善但 leaderboard 迁移不稳定。 |
| 长井窗口拼接风险 | 超过训练窗口长度的井需要 chunk/window 推理。 | 窗口边界处可能出现不连续或上下文缺失。 |
| 训练成本较高 | 五折序列网络训练、特征缓存和 GPU 推理都比 LightGBM 更重。 | 实验迭代慢，Kaggle runtime 需要工程优化。 |

canonical warp_exp207 的 Geo5 level RMSE 约为 8.65975，高于 full249 的 6.02666。这说明 WARP 不适合作为无约束的绝对 level anchor；其强项是提供不同的连续 shape 路径意见。

## 后续尝试

后续曾通过小型 residual adapter 向 WARP 注入 formation 或 SCA 证据。一些单 fold 实验有所改善，但没有形成稳定、可替代原始 WARP 的部署版本。纯 GR/typewell WARP 因此仍是最清晰的参考模型。

## 最适合的使用方式

full249 或 GSN 应负责默认 level anchor；WARP 以小的 strict cross-fitted 权重提供序列 shape 证据，必要时再配合 Winsor、cap 或 gate 限制异常偏离。

当观察到“WARP 局部 shape 看起来正确，但绝对 level 偏移较大”时，应把它视为 shape 专家，而不应直接提高它的全局融合权重。

## 局限

* `cumsum` 会将增量偏差累计成较大的 level 偏差。
* 相比 full249，它对序列分布变化更敏感。
* GR shape 合理不等于绝对 datum / level 正确。
