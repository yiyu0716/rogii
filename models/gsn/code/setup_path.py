"""Bootstrap sys.path for Kaggle inference — no ``from src...`` required to load this file.

Notebook (copy-paste, no hard-coded paths):

```python
import importlib.util
from pathlib import Path

setup_file = next(Path("/kaggle/input").rglob("setup_path.py"))
spec = importlib.util.spec_from_file_location("rogii_setup", setup_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

bundle, code_root = mod.discover_and_install()
print("bundle:", bundle)
print("code:  ", code_root)

from src.infer import run_inference
run_inference(model_dir=bundle / "weights")
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
_MANIFEST_NAME = "manifest.json"


def find_bundle_root(hint: str | Path | None = None) -> Path:
    """Locate infer bundle (manifest.json + code/ + weights/)."""
    if hint is not None:
        path = Path(hint).resolve()
        for candidate in (path, path.parent, path.parent.parent):
            if (candidate / _MANIFEST_NAME).is_file() and (candidate / "code").is_dir():
                return candidate
            if (candidate / _MANIFEST_NAME).is_file() and (candidate / "src").is_dir():
                return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for manifest in kaggle_input.rglob(_MANIFEST_NAME):
            bundle = manifest.parent
            if (bundle / "code" / "src").is_dir():
                return bundle
            if (bundle / "src").is_dir() and (bundle / "config").is_dir():
                return bundle

    # Local dev: repo root when running from source tree
    if (CODE_ROOT / "src").is_dir() and (CODE_ROOT / "config").is_dir():
        return CODE_ROOT.parent
    raise FileNotFoundError(
        "GeoSteerNet infer bundle not found. "
        "Pass hint= to discover_bundle() or set ROGII_INFER_BUNDLE."
    )


def code_root_for_bundle(bundle: Path) -> Path:
    bundle = bundle.resolve()
    nested = bundle / "code"
    if (nested / "src").is_dir():
        return nested
    if (bundle / "src").is_dir():
        return bundle
    raise FileNotFoundError(f"No src/ under bundle {bundle}")


def install(code_root: Path | None = None) -> Path:
    """Insert code root on sys.path (the dir that contains ``src/`` and ``config/``)."""
    root = (code_root or CODE_ROOT).resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def discover_and_install(hint: str | Path | None = None) -> tuple[Path, Path]:
    bundle = find_bundle_root(hint)
    code = code_root_for_bundle(bundle)
    install(code)
    return bundle, code


def load_manifest(bundle: Path) -> dict:
    path = bundle / _MANIFEST_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_inference(
    hint: str | Path | None = None,
    *,
    checkpoint: str = "best",
) -> None:
    bundle, _ = discover_and_install(hint)
    manifest = load_manifest(bundle)
    if manifest:
        print(f"manifest: {manifest.get('name')} {manifest.get('version')}")

    from src.infer import run_inference as _run

    _run(model_dir=bundle / "weights", checkpoint=checkpoint)


if __name__ == "__main__":
    run_inference()
