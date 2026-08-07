import torch
import os
from pathlib import Path

root_path = Path(__file__).parent.parent


def _resolve_competition_root() -> Path:
    """Local data dir, or Kaggle competition input when present."""
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        root_path / "data/raw/rogii-wellbore-geology-prediction",
    ]
    for path in candidates:
        if (path / "test").is_dir() or (path / "train").is_dir():
            return path
    return candidates[-1]


_competition_root = _resolve_competition_root()


class Config:
    # Paths
    TRAIN_DIR = _competition_root / "train"
    TEST_DIR = _competition_root / "test"
    TYPEWELL_SUFFIX = "__typewell.csv"
    HORIZONTAL_SUFFIX = "__horizontal_well.csv"

    #垂直井
    T_SIZE = 272 # Typewell image size(不需要采样,只统一了尺度)
    T_H = 128  # Typewell history image size
    T_F = T_SIZE - T_H   # Typewell future image size
    #水平井
    H_SIZE_ORIGINAL = 12480  #采样前
    H_SIZE_ORI_H = 2240   #需要留的历史长度
    H_SIZE_ORI_F = H_SIZE_ORIGINAL - H_SIZE_ORI_H   #需要留的未来长度
    H_S =24   # Horizontal sample step size
    H_SIZE = H_SIZE_ORIGINAL // H_S  # Horizontal image size(采样后)
    H_H = H_SIZE_ORI_H // H_S    # Horizontal history image size
    H_F = H_SIZE - H_H        # Horizontal future image size
    H_GR_FILTER = (5, 31)       # Savgol filter window for Horizontal GR

    H_FLIP_PROB = 0.5       # 整条 H 轴反转增强；

    # GR noise transfer augmentation (GRNoiseAugDataset)
    # 每 epoch: 1× 原井 + GR_NOISE_N_SYNTH × 合成井（不同 random donor）
    # h_gr_aug = h_gr_sim(A) + noise(C)；仅 future 段替换 h_gr
    GR_NOISE_AUG = True           # 开关；True 启用
    GR_NOISE_N_SYNTH = 4          # 每口原井额外合成样本数（总数据量 = 1+N_SYNTH 倍）
    GR_NOISE_DONOR = "random"     # "random" | "knn"
    GR_NOISE_K   = 5              # donor="knn" 时最近邻井数
    GR_NOISE_FUTURE_ONLY = True   # 仅 PS 后 future 段替换 h_gr

    # # Training Specifics
    ORIG_PAD_LEN = 16384

    # Reproducibility
    SEED = 2026
    DETERMINISTIC = True        # True: exact replay; disables CUDNN_BENCHMARK
    CUDNN_BENCHMARK = True        # only used when DETERMINISTIC=False (faster, may vary)

    # Training speed (tune NUM_WORKERS / BATCH_SIZE for your GPU & CPU)
    BATCH_SIZE = 8
    EPOCHS = 60
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    EVAL_RMSE_EVERY = 1           # Viterbi RMSE every N epochs (1 = 与 early stop 对齐)
    EARLY_STOP_PATIENCE = 40       # 连续 N 次 RMSE eval 无改善则停训
    EARLY_STOP_MIN_EPOCHS = 5     # 至少训满 N epoch 再 early stop
    SHOW_PROGRESS = True

    # SDF 输入：history 通道（C1 消融：关略好）
    USE_HISTORY_IN_BACKBONE = False
    USE_HISTORY_IN_HEAD = False
    # Loss / decode 与 Viterbi RMSE 评估区域（future）对齐
    SDF_LOSS_FUTURE_ONLY = True
    SDF_LOSS_GAMMA = 2.0       # 零带权重衰减；↑ 更集中零交叉附近（原 2.0）
    SDF_LOSS_EPS = 1e-3
    SDF_LOSS_L1_WEIGHT = 0.5   # base = MSE + w·L1；↑ 零带小误差梯度更强（原 0.5）
    SEG_LOSS_FUTURE_ONLY = False
    SEG_LINE_THICKNESS = 3      # 线宽
    SEG_VITERBI_WEIGHT = 1       # cost = |sdf| - weight * seg_prob
    SEG_BCE_POS_WEIGHT = 20    # BCE 正样本权重
    SEG_DICE_WEIGHT = 0.9        # BCE 与 Dice 的权重
    SEG_LOSS_WEIGHT = 5          # 线性组合 SDF 与 Segmentation 的损失
    USE_COORD_ENCODING = False
    USE_GR_DIFF_INPUT = True    # learned correlation 消融：去掉手工 t_gr - h_gr

    # Learned GR correlation volume: 1D encoders -> cosine similarity [B, T, H].
    # 仅作为额外输入通道；现有 UNet+ASPP / SDF+Seg loss / decode 均不改变。
    USE_LEARNED_CORRELATION = True
    CORRELATION_EMBED_DIM = 64
    CORRELATION_SCALE = 1.0

    LR = 5e-4
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    N_FOLDS = 5
    # CV split: "typewell_group" (legacy GroupKFold) | "geo_kmeans" (XY k-means + stratified group)
    CV_SPLIT_STRATEGY = "geo_kmeans"
    GEO_KMEANS_SEED = 5
    GEO_KMEANS_N_CLUSTERS = None   # None → use N_FOLDS
    
    # GeoSteerNet / Dataset Specifics
    TVT_DIFF_SCALE = 40.0
    SDF_TVT_RANGE_FT = 120.0        # clip 对应的 TVT 差上限 (ft)
    SDF_CLIP = SDF_TVT_RANGE_FT / TVT_DIFF_SCALE  # label clip & SDF 输出硬裁剪 ±SDF_CLIP
    SDF_OUTPUT_SCALE = 1.0   # 线性缩放：sdf = clamp(raw * scale, ±SDF_CLIP)；不用 tanh

    # Dropout2d (train only; model.eval() 自动关闭)
    USE_DROPOUT2D = True
    DROPOUT_ASPP = 0.2           # ASPP 融合后
    DROPOUT_DECODER = 0.20        # 每个 UpBlock 末尾
    DROPOUT_ENCODER_DEEP = 0.15   # encoder layer3/4 的 ResBlock
    DROPOUT_HEAD = 0.15           # head 中间 Conv 之后、1×1 输出之前

    # Predicting From Earlier (PFE) — train only; val/infer 始终用真 PS
    USE_PFE_TRAIN        = True
    PFE_PROB             = 0.75    # 其余 25% 样本仍用真 PS
    # 伪 PS 候选：history 中 TVT > ps_tvt - PFE_TVT_SHIFT_THRESHOLD (ft)
    # 默认约 (T_F - 真PS后future行数p90) * 0.5ft/行 ≈ (144-68)*0.5
    PFE_TVT_SHIFT_THRESHOLD = 40
    PFE_MIN_HISTORY_ORIG = 800     # 伪 PS 前至少保留多少已知点（含 PS 点）
    PFE_MIN_FUTURE_ORIG  = 1500    # 伪 PS 后至少多少原始点用于 future 监督

    # ── MTP (Multiple Trajectory Prediction) pipeline ────────────────────────
    # Independent of the SDF pipeline; uses separate downsampling and model.
    MTP_H_S       = 64      # downsample step  (64 orig samples ≈ 64 ft spacing)
    MTP_H_H_ORIG  = 2240    # history window in original samples
    MTP_H_F_ORIG  = 10240   # future window in original samples
    MTP_H_H       = MTP_H_H_ORIG // MTP_H_S    # = 35  history steps
    MTP_H_F       = MTP_H_F_ORIG // MTP_H_S    # = 160 future prediction steps
    MTP_K         = 8      # number of mixture trajectories
    MTP_HIDDEN    = 512     # backbone feature dim
    MTP_ALPHA     = 0.5     # cls loss weight: total = reg + alpha * cls
    MTP_LR        = 1e-3
    MTP_EPOCHS    = 60

    # ── Prior strategy (THE key design choice) ───────────────────────────────
    # The MTP net predicts geo_dz = geo_z_true - geo_z_prior, i.e. it REFINES a
    # prior.  How good that prior is determines the whole task difficulty.
    #
    #   "true_noise"     prior = geo_z_true + smooth (kriging-like) error.
    #                    Reproduces the reference method (LB<7).  The net only
    #                    learns a small residual.  REQUIRES a real kriging prior
    #                    at inference time (test set has no TVT) — local CV in
    #                    this mode is optimistic until kriging is wired in.
    #
    #   "extrapolation"  prior = TVT_input held constant past PS (+ noise).
    #                    Fully honest / submittable today, but the net must
    #                    predict the entire formation movement (~12 ft target)
    #                    from GR alone — much harder, ~12 ft floor.
    #
    #   "particle_filter" prior = PF-estimated TVT + Z (GR/typewell matching).
    #                    Honest at train/val/test; MTP refines PF residual.
    #                    This is the deployable version of the reference method.
    MTP_PRIOR_MODE  = "particle_filter"
    MTP_NOISE_PRIOR = 8.0   # ft, used by true_noise / extrapolation modes
    MTP_NOISE_SLOPE = 5.0   # (legacy, unused by sinusoid noise)

    # Particle-filter prior (MTP_PRIOR_MODE == "particle_filter")
    MTP_PF_N_PARTICLES = 500
    MTP_PF_N_SEEDS     = 16     # notebook uses 64–128; lower for dataset speed
    MTP_PF_SCALE       = 5.0
    MTP_PF_TRAIN_NOISE = 2.0    # ft std of extra noise on PF prior during train; 0=off
    MTP_PF_CACHE       = True   # cache PF TVT per well inside the dataset

    # ── DDPM noise augmentation pipeline ─────────────────────────────────────
    DDPM_SEQ_LEN       = 512
    DDPM_STRIDE        = 128
    DDPM_GR_FILTER     = H_GR_FILTER   # match main model Savgol smoothing
    DDPM_T             = 1000          # training diffusion steps
    DDPM_SAMPLE_STEPS  = 50            # DDIM inference steps (fast augment)
    DDPM_BASE_CH       = 32            # lighter 1D UNet for quick experiments
    DDPM_CHANNEL_MULTS = (1, 2, 4)     # 3 levels (vs 4) — fewer params
    DDPM_NUM_RES       = 2
    DDPM_TIME_EMB      = 128
    DDPM_LR            = 1e-4
    DDPM_EPOCHS        = 100           # full training; quick exp overrides
    DDPM_BATCH         = 64

    # ── Formal single-fold experiment (see ddpm_formal_experiment.py) ───────
    DDPM_FORMAL_FOLD           = 0
    DDPM_FORMAL_HOLDOUT_RATIO  = 0.2   # 从 fold-train 切 20% 作 holdout
    DDPM_FORMAL_DDPM_EPOCHS    = 50
    DDPM_FORMAL_SDF_EPOCHS     = 30
    DDPM_FORMAL_SAMPLE_STEPS   = 100   # DDIM 步数（正式实验用更多步）