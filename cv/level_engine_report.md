# LevelEngine — 物理 drift 预测器 CV 报告

参数：beta=0.3, clip=20.0, k_use=15, stride_anchor=40, cloud_stride=80, F=ANCC

参考刻度：CF 15.9099 / const-oracle 9.04 / smooth-201 0.39

## 三套 split 真分

| split    | rw_rmse | drift_corr | shape_corr | p50     | p90     | max     | n_wells |
| -------- | ------- | ---------- | ---------- | ------- | ------- | ------- | ------- |
| well     | 13.3715 | 0.5898     | 0.5520     | 8.5244  | 18.4698 | 58.6430 | 773     |
| typewell | 13.3766 | 0.5904     | 0.5411     | 8.6435  | 18.7798 | 58.6430 | 773     |
| spatial  | 16.1897 | 0.0815     | 0.0405     | 10.9710 | 23.5245 | 70.6394 | 773     |

> well=井级随机；typewell=典井母本绑同折；spatial=KMeans 地理区留出（压力下界）。

## worst-well by SSE (spatial, top 20)

| well_id  | n_hidden | rmse    | sse           | sse_share |
| -------- | -------- | ------- | ------------- | --------- |
| 86454a6f | 7964     | 63.9070 | 32525846.8442 | 0.0328    |
| 1b1eba53 | 4655     | 70.6394 | 23228083.4032 | 0.0234    |
| 2fd68f7b | 4730     | 60.7402 | 17450701.8459 | 0.0176    |
| 5f4d2a52 | 5225     | 56.9149 | 16925383.2892 | 0.0171    |
| 389ae58f | 6463     | 49.7630 | 16004721.4057 | 0.0161    |
| f6d009f4 | 6715     | 47.0392 | 14858219.4864 | 0.0150    |
| f88ddb26 | 4990     | 51.4031 | 13184962.8925 | 0.0133    |
| fb03ae90 | 6431     | 42.8876 | 11828809.2462 | 0.0119    |
| c8d9680c | 6281     | 42.7704 | 11489898.9494 | 0.0116    |
| ba48188d | 4166     | 50.9372 | 10809097.6666 | 0.0109    |
| a959858c | 4404     | 48.2736 | 10262801.8185 | 0.0103    |
| 7e721392 | 6174     | 40.3318 | 10042975.5438 | 0.0101    |
| 25050f63 | 6124     | 37.9966 | 8841477.0462  | 0.0089    |
| 206b6193 | 6535     | 36.5090 | 8710547.4097  | 0.0088    |
| 91b301ce | 6570     | 35.3151 | 8193806.1082  | 0.0083    |
| 708caea9 | 6548     | 34.9165 | 7983053.4979  | 0.0080    |
| 43e16325 | 3720     | 45.7767 | 7795270.5105  | 0.0079    |
| fef8af96 | 3826     | 44.5517 | 7594062.4570  | 0.0077    |
| 4c2208f5 | 5384     | 37.3819 | 7523630.3192  | 0.0076    |
| 9a95e33f | 6447     | 32.2893 | 6721646.7854  | 0.0068    |

## 提交结果（2026-06-02）— 物理引擎首次在真 LB 上破 CF

| 量 | 值 | 说明 |
|---|---|---|
| **public LB** | **12.657** | kernel `yiyu0716/rogii-physics-level-v001` v1，hidden public 井 |
| LB constant 基线 | 15.88 | 公开 const |
| 此前自己最好 | 9.589 | 公开 DWT level 复刻 |
| 公开前沿 | ~9.25–9.5 | ≈ const-oracle 9.04（只估对 level） |
| 离线 3 可见井 (in train) | 7.35 vs CF 11.54 | 这 3 井在 train 里、邻井致密、非 LB 目标集 |

**判读（关键）：**
1. **物理 drift 信号是真的、且 fold-safe**：在 hidden public 井（不在 train）上 15.88(CF)→**12.66**，
   与 well/typewell CV 13.37 一致 → 证明真实 test 井邻居致密（像 well/typewell，非 spatial 最坏 16.2）。
   这是首个**捕获 shape（drift_corr 0.59）**且不靠 level-plateau 旋钮的 fold-safe 模型。
2. **但单体引擎还打不过现有 level 模型（~9.5）**：瓶颈是 **ΔF（level）插补太糙** →
   bias-variance 最优 β 只能压到 0.3 → 只修正了约 40% 的 level → 13.37/12.66 卡在 const-oracle(9.04) 上方。
   IDW ΔF 噪声大，吃不到 level。
3. **战略含义**：level 模型(9.04 plateau)吃 level、吃不到 shape；物理引擎吃 shape、level 还糙。
   冠军 6.5–7.3 = level + shape 都要。物理引擎是正确主干，下一步必须：
   - **Stage 1 强化 ΔF**（kriging/LWR 替 IDW、prefix-F 连续性锚定、6 top 平均降噪）→ 把 β 顶上去、level 压向 9.04；
   - **Stage C GBM 融合**：level 取自 level 模型 / 强化 ΔF，shape 取自物理 drift，让树学 per-well 信多少。

## Ring 0/1/2 + 前沿对标（2026-06-02/03）

**Ring 0（天花板）**：oracle-level + 物理 shape = **7.69**（冠军带）；oracle-level only=9.04；
只用已知 -ΔZ 的 shape 加不了任何东西（全部 shape 价值在 ΔF）。mean/dev 分离收缩+clip=13.06（微胜 13.37）。
→ **瓶颈 100% 是 per-well LEVEL（段内均值 drift），不是 shape。**

**Ring 1（prefix 锚定）**：失败。prefix-F 偏置校准把 level corr 从 0.453 拉到 0.426（更差）；
局部 offset 噪声大且不外推到远场。

**Ring 2（level GBM）**：HGB+CatBoost 24 特征预测 mean-drift，corr 仅 **0.586**（RMSE 10.3）；
融合后 13.02。单特征 corr 排名：pdrift_mean 0.453 最强，其余 ≤0.33。**特征集里没有强 level 信号。**

**前沿对标（mitch LB 8.905 writeup）**：163 特征 GBDT 集成（XGB+CB+HGB, GroupKFold, NNLS, Savgol），
OOF 9.85。所有单信号都弱（NCC standalone 28ft、PF 13.4、xcorr 118ft）；**level 由 GBDT 堆叠众多弱正交信号涌现**，
无任何单一强信号。我的物理引擎(13.0)是现存最强单信号之一。

**结论（战略修正）**：纯物理分解封顶≈13，因为 LEVEL 学不出来——它只能靠 GBDT 堆叠多弱信号涌现（mitch 配方）。
物理 drift = 强核心特征 + 被前沿低估的 shape 优势（oracle-level+物理shape=7.69）。
→ 冲金 = Stage C 多信号 GBDT（物理 drift 核心 + NCC + anchor extrap + PF/beam + GR/几何）先对齐前沿 9.85，
再用物理 shape 优势压向 7–8。此路线与现有 9.589 管线必然重叠。

## Stage C 实测（2026-06-03）— clean-from-scratch 撞上同一堵 LEVEL 墙

- **v1**（22 特征：物理 drift 核心 + anchor-extrap + 几何 + GR）行级 GBDT = **13.088**（物理单体 13.37，CF 15.91）。
  已**优于 mitch R3（52 特征 OOF 14.99）**——物理特征独自carry。
- **denser ANCC 云**（stride 80→20→8）：13.37→13.52→13.54，**更差**。密度不是杠杆。
- **v2 + 多尺度 NCC**（per-row + 聚合）= **13.164**，**没帮助**（略差）。
- **NCC level 正交性检验**：corr(target, ncc_mean)=0.128，与 physics 冗余（合并 R 0.458 vs 0.453）。
  → GR/NCC 携带的独立 level 信息≈0（lateral GR 横向相变主导，与原 EDA 一致）。PF 同属 GR，大概率同样无效。

**LEVEL 墙性质**：physics(spatial formation) 给 level corr≈0.59 已是上限；GR、几何、密度、plane-dip、prefix-anchor 都加不动。
mitch 9.85 OOF 推出其 level corr≈0.9，靠 163 特征多估计器堆叠涌现——而在我这条强物理基线上，单加 NCC 不涌现。
要复现≈10 必须把 mitch 整套估计器族都搭齐（beam×4/PF×3/formation 变体/slopes/divergence），逐个微增、工作量巨大，
**且这恰好就是你已有的 9.589 管线在做的事**。

**关键事实**：我的 clean 物理 GBDT(13.09) 仍**劣于你现有 9.589**。物理工作要兑现，唯一高效路径是把
**物理 shape（Ring0 证 oracle-level+物理shape=7.69）叠到你现有的好 level 上**，而不是从零把 level 再造一遍到 9.5。

## 嫁接实测（2026-06-03）— 物理 shape 叠到现有 pipeline ≈ 0 增益（决定性负结果）

用 thbdh5765 v10 OOF（3.78M 行，drift 空间，部署 hill-climb≈10.44；最优单列 col6=11.28，NNLS blend≈11.13）：
- base + β·conf·物理shape：β_opt≈0.05，RMSE Δ **−0.015**（blend 上）/ −0.004（col6）。**无增益。**

**误差分解（col6, RMSE 11.28）**：level 残差 **8.89**（主导）/ swing 残差 6.95。
- base level corr **0.737** ＞ 物理 level 0.442；base shape corr **0.655** ＞ 物理 shape 0.471。
- base_shape vs 物理_shape 冗余 0.529。边际：swing~base_sh R=0.655 → +物理_sh R=0.671（物理系数 +0.10，微弱）。
- 反事实 oracle-level：only 9.04 / +base_sh 6.95 / +物理_sh 7.98 / +两者 **6.70**(a0.7,b0.1)。

**决定性结论**：
1. 现有模型的 **shape 已优于物理**（0.655 vs 0.471），物理 shape 大半冗余，边际仅 oracle 下 6.95→6.70(0.25ft)、真实 level 主导基线上≈0。
2. 普遍瓶颈是 **LEVEL**（base level 残差 8.89），而现有模型 level(0.737) 已**优于**物理(0.44)。物理在 level 上帮不上。
3. **物理-shape 作为嫁接增益的金牌主干，证伪**。物理是干净的独立信号(单体13.0)，但叠到强基线上≈0。
4. 真正的金牌杠杆是 **LEVEL**（要 corr 0.74→0.95+），而 physics/GR/几何/密度/anchor/plane 全部无法攻克。
   冠军 6.5–7.3 若真实，必有一个公开方案都没有的 level 信号（或私榜异象）。下一步只有"专攻 level 的全新信号"才有意义。

## level_combine 终判（2026-06-03）— LEVEL 在已知信号下封顶 ~0.75

base(NNLS-blend) level corr **0.751**（resid 8.39）；physics level 0.45。
- 线性组合（同OOF 乐观上界）：base+physics=0.752（+0.001）、base+全部 extra=0.765（+0.014，且乐观）。
- **诚实 GBM 学新 level（GroupKFold）= 0.708，比 base 还差**；new_level+base_shape pooled=11.62 ＞ base 11.13。
- oracle_level + base_shape = **6.827**（=金牌带）——11.1→6.8 的全部差距都是 LEVEL。

**最终结论**：LEVEL（per-well 均值 drift）在所有已知信号（physics/GR/几何/密度/anchor/plane/组合）下 corr 封顶≈0.75，
无法推向金牌所需的 ~0.95。物理对现有 pipeline 既不加 level（0.45<0.75）也不加 shape（冗余、≈0）。
**物理-shape 冲金主干 + level 可破，双双证伪。** 现有 ~9.7 LB / ~10.4 OOF 已近"已知信号天花板"。
冠军 6.5–7.3 若真实，必依赖一个公开方案都没有的 level 信号（或私榜异象）；用现有数据拿不到。
