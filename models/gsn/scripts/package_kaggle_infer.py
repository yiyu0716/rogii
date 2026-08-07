#!/usr/bin/env python3
"""Package GeoSteerNet code + fold checkpoints for Kaggle Dataset upload.

Usage:
  python scripts/package_kaggle_infer.py outputs/cnn_sdf_v2.0 -o dist/kaggle_infer_v2.0

Upload ``dist/kaggle_infer_v2.0`` as a Kaggle Dataset, then in the submission
notebook:

  from src.kaggle_env import bootstrap, default_model_dir
  bootstrap(bundle_root="/kaggle/input/datasets/<you>/<slug>")
  from src.infer import run_inference
  run_inference(model_dir=default_model_dir())
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kaggle_env import infer_code_files, repo_root, setup_path_source  # noqa: E402


def _read_history_tvt_mode(dataset_py: Path) -> str:
    text = dataset_py.read_text(encoding="utf-8")
    match = re.search(r'^HISTORY_TVT_INPUT_MODE\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return match.group(1) if match else "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_infer_bundle(
    weights_dir: Path,
    output_dir: Path,
    *,
    version: str,
    checkpoint: str = "best",
) -> Path:
    weights_dir = weights_dir.resolve()
    output_dir = output_dir.resolve()
    code_dir = output_dir / "code"
    weights_out = output_dir / "weights"

    if code_dir.exists():
        shutil.rmtree(code_dir)
    if weights_out.exists():
        shutil.rmtree(weights_out)
    code_dir.mkdir(parents=True)
    weights_out.mkdir(parents=True)

    # Preserve import layout: config/ + src/
    for rel in infer_code_files():
        src = repo_root() / rel
        if not src.is_file():
            raise FileNotFoundError(f"Missing inference file: {src}")
        dst = code_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    setup_src = setup_path_source()
    if not setup_src.is_file():
        raise FileNotFoundError(f"Missing {setup_src}")
    shutil.copy2(setup_src, code_dir / "setup_path.py")

    # Entry script at bundle root: python run_infer.py (no import chicken-egg)
    run_infer = f'''\
"""One-shot Kaggle inference. Usage: python run_infer.py"""
from pathlib import Path
import importlib.util

BUNDLE = Path(__file__).resolve().parent
setup_file = BUNDLE / "code" / "setup_path.py"
spec = importlib.util.spec_from_file_location("rogii_setup", setup_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.run_inference(BUNDLE)
'''
    (output_dir / "run_infer.py").write_text(run_infer, encoding="utf-8")

    weight_files: list[dict] = []
    for fold in range(16):
        name = f"fold_{fold}_{checkpoint}.pth"
        src = weights_dir / name
        if not src.is_file():
            continue
        dst = weights_out / name
        shutil.copy2(src, dst)
        weight_files.append({
            "file": name,
            "sha256": _sha256(dst),
            "bytes": dst.stat().st_size,
        })

    if not weight_files:
        raise FileNotFoundError(
            f"No checkpoints fold_*_{checkpoint}.pth found in {weights_dir}"
        )

    dataset_py = repo_root() / "src/dataset.py"
    manifest = {
        "name": "rogii-geosteernet-sdf",
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "history_tvt_input_mode": _read_history_tvt_mode(dataset_py),
        "weights": weight_files,
        "layout": {
            "code": "code/",
            "weights": "weights/",
            "imports": "sys.path.insert(0, code/) then from src.infer import run_inference",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = f"""# ROGII GeoSteerNet inference bundle ({version})

## Kaggle notebook（推荐，无需手写路径）

```python
import importlib.util
from pathlib import Path

# 自动搜索 /kaggle/input 下任意层级的 setup_path.py
setup_file = next(Path("/kaggle/input").rglob("setup_path.py"))
spec = importlib.util.spec_from_file_location("rogii_setup", setup_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

bundle, code_root = mod.discover_and_install()
print(mod.load_manifest(bundle))

from src.infer import run_inference
run_inference(model_dir=bundle / "weights")
```

或一行跑完：

```python
!python /kaggle/input/datasets/aiwody/v2-0-baseline/kaggle_infer_v2.0/run_infer.py
```

## 目录说明

- `code/` — 含 `config/`、`src/`、`setup_path.py`（**sys.path 应指向 code/，不是 code/src/**）
- `weights/` — {len(weight_files)} 个 fold checkpoint（`{checkpoint}`）
- `run_infer.py` —  bundle 根目录一键推理
- `manifest.json` — 版本与 checksum

## 注意

- 竞赛数据自动从 `/kaggle/input/competitions/rogii-wellbore-geology-prediction` 读取
- `history_tvt_input_mode={manifest["history_tvt_input_mode"]}`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Package GeoSteerNet for Kaggle inference")
    parser.add_argument(
        "weights_dir",
        type=Path,
        help="Directory containing fold_*_best.pth (e.g. outputs/cnn_sdf_v2.0)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Output bundle directory (upload this folder as Kaggle Dataset)",
    )
    parser.add_argument("--version", default="v1", help="Bundle version tag")
    parser.add_argument(
        "--checkpoint", default="best", choices=("best", "last"),
        help="Which checkpoint suffix to pack",
    )
    args = parser.parse_args()

    out = package_infer_bundle(
        args.weights_dir,
        args.output_dir,
        version=args.version,
        checkpoint=args.checkpoint,
    )
    print(f"Bundle written to: {out}")
    print(f"  weights: {len(list((out / 'weights').glob('*.pth')))} files")
    print(f"  upload:  zip or upload folder as Kaggle Dataset")


if __name__ == "__main__":
    main()
