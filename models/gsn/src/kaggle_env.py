"""Bootstrap project imports and paths for local vs Kaggle inference."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_INFER_CODE_FILES = (
    "config/cnn_sdf_config.py",
    "src/__init__.py",
    "src/dataset.py",
    "src/model.py",
    "src/seed.py",
    "src/train.py",
    "src/infer.py",
    "src/kaggle_env.py",
)

_KAGGLE_COMPETITION_DIRS = (
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
)

_MANIFEST_NAME = "manifest.json"


def repo_root() -> Path:
    return _REPO_ROOT


_SETUP_PATH_REL = "code/setup_path.py"


def infer_code_files() -> tuple[str, ...]:
    return _INFER_CODE_FILES


def setup_path_source() -> Path:
    return repo_root() / _SETUP_PATH_REL


def is_kaggle_runtime() -> bool:
    return Path("/kaggle").exists()


def find_competition_data_dir() -> Path | None:
    for path in _KAGGLE_COMPETITION_DIRS:
        if (path / "test").is_dir() or (path / "train").is_dir():
            return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for manifest in kaggle_input.rglob(_MANIFEST_NAME):
            bundle = manifest.parent
            for base in (bundle, bundle.parent):
                for path in (base, base / "rogii-wellbore-geology-prediction"):
                    if (path / "test").is_dir() or (path / "train").is_dir():
                        return path
    return None


def _is_bundle_dir(path: Path) -> bool:
    path = path.resolve()
    if not (path / _MANIFEST_NAME).is_file():
        return False
    return (path / "code" / "src").is_dir() or (path / "src").is_dir()


def find_bundle_root(hint: str | Path | None = None) -> Path | None:
    """Locate packaged infer bundle (manifest.json + code/)."""
    if hint is not None:
        path = Path(hint).resolve()
        for candidate in (path, path.parent, path.parent.parent):
            if _is_bundle_dir(candidate):
                return candidate

    env_hint = os.environ.get("ROGII_INFER_BUNDLE")
    if env_hint and _is_bundle_dir(Path(env_hint)):
        return Path(env_hint).resolve()

    if is_kaggle_runtime():
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.is_dir():
            for manifest in kaggle_input.rglob(_MANIFEST_NAME):
                bundle = manifest.parent
                if _is_bundle_dir(bundle):
                    return bundle
    return None


def code_root_for_bundle(bundle: Path) -> Path:
    bundle = bundle.resolve()
    nested = bundle / "code"
    if (nested / "src").is_dir():
        return nested
    if (bundle / "src").is_dir():
        return bundle
    raise FileNotFoundError(f"No src/ under bundle {bundle}")


def bootstrap(
    *,
    code_root: str | Path | None = None,
    bundle_root: str | Path | None = None,
) -> Path:
    """Put code on ``sys.path``. Must be called before other ``from src...`` imports."""
    if code_root is not None:
        root = Path(code_root).resolve()
    else:
        bundle = find_bundle_root(bundle_root)
        if bundle is not None:
            root = code_root_for_bundle(bundle)
        else:
            root = _REPO_ROOT

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_setup_path_module(hint: str | Path | None = None):
    """Load standalone ``setup_path.py`` without ``from src...`` (for notebooks)."""
    if hint is not None:
        p = Path(hint).resolve()
        candidates = [
            p / "code" / "setup_path.py",
            p / "setup_path.py",
            p.parent / "code" / "setup_path.py",
        ]
        for setup_file in candidates:
            if setup_file.is_file():
                return _import_setup_module(setup_file)

    if is_kaggle_runtime():
        for setup_file in Path("/kaggle/input").rglob("setup_path.py"):
            return _import_setup_module(setup_file)

    local = _REPO_ROOT / "code" / "setup_path.py"
    if local.is_file():
        return _import_setup_module(local)
    raise FileNotFoundError("setup_path.py not found")


def _import_setup_module(setup_file: Path):
    spec = importlib.util.spec_from_file_location("rogii_setup_path", setup_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {setup_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_and_install(hint: str | Path | None = None) -> tuple[Path, Path]:
    """Notebook-safe entry: find bundle, add ``code/`` to sys.path."""
    mod = load_setup_path_module(hint)
    return mod.discover_and_install(hint)


def default_model_dir(bundle_root: str | Path | None = None) -> Path:
    bundle = find_bundle_root(bundle_root)
    if bundle is not None:
        weights = bundle / "weights"
        if weights.is_dir():
            return weights
    env_weights = os.environ.get("ROGII_MODEL_DIR")
    if env_weights:
        return Path(env_weights)
    return _REPO_ROOT / "outputs" / "cnn_sdf"


def load_manifest(bundle_root: str | Path | None = None) -> dict:
    bundle = find_bundle_root(bundle_root)
    if bundle is None:
        return {}
    manifest_path = bundle / _MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
