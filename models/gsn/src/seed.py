import os
import random

import numpy as np
import torch

from config.cnn_sdf_config import Config


def seed_everything(seed: int | None = None) -> int:
    """Fix RNG seeds for Python / NumPy / PyTorch."""
    seed = int(Config.SEED if seed is None else seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if Config.DETERMINISTIC:
        # Reproducibility mode: disable autotune & non-deterministic cuDNN ops.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    elif Config.CUDNN_BENCHMARK and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    return seed


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Per-worker seed so augmentations are reproducible with num_workers > 0."""
    worker_seed = int(Config.SEED) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
