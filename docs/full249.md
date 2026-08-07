# full249

## 定位

`full249` 是项目的主要 tabular anchor。它逐行预测未知段，而不是直接解码整条路径。输出为相对最后一个可见 TVT 的 drift：

```text
drift_hat = LightGBM(feature_1 ... feature_249)
TVT_hat   = last_tvt + drift_hat
```

它是后续 WARP、HMM、GSN、候选路径和 ModelB 的比较基线，也是很多系统的默认安全锚。

## 输入

每口水平井的输入包括：

* 可见 `TVT_input` 前缀，特别是最后一个可见值 `last_tvt`。
* 全段 MD、X/Y/Z 轨迹和 GR 日志。
* typewell 的 TVT-GR 剖面，以及可用时的地层标签。
* 仅由训练 folds 构建的空间和地层先验。

推理时未知未来 TVT 不会作为任何输入特征。

## 249 个特征族

精确列顺序冻结在模型资产中。概念上，249 个特征包括：

| 特征族 | 例子与作用 |
|---|---|
| 几何与进度 | 距 heel 的 MD、未知段相对位置、X/Y/Z 轨迹量与局部方向导数。 |
| 已知段延续 | `last_tvt`、尾段 TVT/U-space slope、外推 level 和趋势。 |
| GR/typewell 对齐 | 候选 TVT 对应的 typewell GR、GR 残差、滚动统计、heel affine calibration 与匹配 margin。 |
| PF/likelihood | particle filter level、likelihood 路径、posterior mode 与置信度。 |
| beam/path 候选 | 多条连续候选路径的 mean、median、spread。 |
| ANCC/NCC 类证据 | 局部 GR 形状相关匹配和对齐置信度。 |
| 空间/地层信息 | 从训练井得到的 surface 与区域几何证据。 |
| mode confidence | PF、beam、GR 和其他候选估计器之间的 gap 与 disagreement。 |

设计重点是把多个地质假设及其分歧暴露给 LightGBM，而不是让某个物理模型单独决定输出。

## PF / likelihood 子系统

物理状态通常表示为：

```text
U = TVT + Z
state = [U, dU/dMD]
```

粒子从最后可见 heel 的状态出发，依据尾段 U-rate 和过程噪声向未知段传播。每个粒子给出一个候选 TVT；typewell 曲线则给出该 TVT 下的期望 GR。观测到的水平井 GR 与期望 GR 越一致，粒子的 likelihood 越大。

GR 一致的粒子权重上升，不一致的粒子权重下降；当有效粒子数过低时进行重采样。最终得到的不是直接提交结果，而是 level、路径、mode 和不确定性特征，供 LightGBM 使用。

## 训练和 OOF

原始路线按整口井训练五个 LightGBM fold 模型。训练井的每一行都由未见过该井标签的 held-fold 模型预测，形成 OOF drift。

测试时，以相同配置构造同顺序的 249 个特征，并对五个 fold 模型输出取平均：

```text
test_drift = mean(fold0 ... fold4 LightGBM predictions)
```

训练 OOF 与测试五折均值之间的语义对应，是 full249 的核心部署约束。

## 结果与作用

full249 在早期历史划分下的 CV 约为 `7.87697`；在后期 canonical Geo5 口径下，可比较的 standalone OOF 约为 `8.14873`。

它不是后期单臂 CV 最低的模型，但对普通井更稳定，因此一直具有 anchor 价值。许多实验都表明，直接用候选路径或高容量修正替换 full249 虽能修复部分 hard well，却可能损害 leaderboard 上的好井。

## 局限

* 它是 row-wise 模型，整井路径一致性较弱。
* 重复地层会使 PF、beam 和 GR 证据多模态。
* 它自身不会判断候选 sidecar 修正是否安全，因此后续需要 gate 和 ModelB。
