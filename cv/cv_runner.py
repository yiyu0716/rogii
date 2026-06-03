#!/usr/bin/env python3
"""ROGII 第 0 步：锁 CV runner（唯一基础设施）。

三套 split（well / 典井母本 typewell-parent / 空间压力 spatial-pressure），每套 5 折，
对每口井用真实 masked-tail（post-PS 段 TVT_input 缺失、TVT 为真值）评分。

提供:
  - 冻结的折分配 CSV（一次生成后复用，所有后续实验共用同一套 split）。
  - Predictor 协议: fit(train_ids, store) + predict(val_id, store)。
  - 统一评分: row-weighted RMSE、per-well p50/p90/max、按 SSE 排 worst-well、
    相对 CF 改善、按井型分桶。
  - Markdown 报告 + per-well CSV artifacts。

本文件自带两个 baseline（CarryForward 训练无关、GlobalMeanDrift 用训练折）做 runner 自检；
真正的 baseline ladder（第 1 步）与模型（第 2 步）只需实现 Predictor 接口接进来即可。

用法:
  python3 cv_runner.py                 # 生成/复用 split，跑自检 baseline，出报告
  python3 cv_runner.py --refresh-splits  # 重新随机生成并覆盖冻结的 split
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------------------------------------------------------------- 常量
def _find_root() -> Path:
    """定位项目根：优先环境变量 ROGII_ROOT，否则在常见位置中探测。"""
    import os
    env = os.environ.get("ROGII_ROOT")
    candidates = ([Path(env)] if env else []) + [
        Path("/home/yiyu/rogii"), Path("/root/ROGII"),
        Path(__file__).resolve().parent.parent,
    ]
    for c in candidates:
        if (c / "datasets" / "rogii-wellbore-geology-prediction" / "train").is_dir():
            return c
    raise FileNotFoundError("无法定位 ROGII 根目录；请设置 ROGII_ROOT 环境变量")


ROOT = _find_root()
DATA_ROOT = ROOT / "datasets" / "rogii-wellbore-geology-prediction"
DASHBOARD = ROOT / "eda" / "eda2_artifacts" / "baseline_per_well_dashboard.csv"
OUT_DIR = ROOT / "cv"
SPLIT_DIR = OUT_DIR / "splits"
ART_DIR = OUT_DIR / "artifacts"
REPORT = OUT_DIR / "cv_report.md"

N_FOLDS = 5
RANDOM_SEED = 20260602
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
SPLIT_SCHEMES = ["well", "typewell", "spatial"]
EPS = 1e-9


# ---------------------------------------------------------------- 工具
def well_id_from_path(path: Path) -> str:
    name = path.name
    return name.split("__", 1)[0] if "__" in name else name.split(".", 1)[0]


def first_missing_index(arr: np.ndarray) -> int | None:
    mask = ~np.isfinite(arr)
    if not mask.any():
        return None
    return int(np.flatnonzero(mask)[0])


def md_table(df: pd.DataFrame, *, floatfmt: str = ".4f", index: bool = False) -> str:
    """无依赖的 markdown 表格（避免 tabulate 外部依赖）。"""
    if df is None or df.empty:
        return "_无记录_"
    df = df.reset_index() if index else df.copy()

    def fmt(v):
        if isinstance(v, float) or isinstance(v, np.floating):
            return format(float(v), floatfmt)
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    cols = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
              for i in range(len(cols))]
    head = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |"
            for r in rows]
    return "\n".join([head, sep] + body)


# ---------------------------------------------------------------- 井数据
@dataclass
class Well:
    well_id: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt: np.ndarray          # 真值（全段）
    tvt_input: np.ndarray    # 输入（post-PS 为 NaN）
    ps_idx: int              # 第一个缺失行 = prediction start
    last_known: float        # ps_idx-1 处的已知 TVT

    @property
    def hidden_slice(self) -> slice:
        return slice(self.ps_idx, len(self.md))

    @property
    def hidden_truth(self) -> np.ndarray:
        return self.tvt[self.ps_idx:]

    @property
    def n_hidden(self) -> int:
        return len(self.md) - self.ps_idx

    @property
    def ps_xy(self) -> tuple[float, float]:
        return float(self.x[self.ps_idx - 1]), float(self.y[self.ps_idx - 1])


class WellStore:
    """加载并缓存全部 train 水平井 + 典井 hash。"""

    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root
        self.wells: dict[str, Well] = {}
        self.typewell_hash: dict[str, str] = {}
        self._typewell_path: dict[str, Path] = {}
        self._load()

    def _load(self) -> None:
        train_dir = self.data_root / "train"
        h_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
        t_paths = {well_id_from_path(p): p for p in train_dir.glob("*__typewell.csv")}
        usecols = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]
        for p in h_paths:
            wid = well_id_from_path(p)
            df = pd.read_csv(p, usecols=usecols)
            tvt_input = df["TVT_input"].to_numpy(dtype=float)
            ps = first_missing_index(tvt_input)
            if ps is None or ps <= 0:
                continue  # 无隐藏段，跳过
            tvt = df["TVT"].to_numpy(dtype=float)
            self.wells[wid] = Well(
                well_id=wid,
                md=df["MD"].to_numpy(dtype=float),
                x=df["X"].to_numpy(dtype=np.float32),
                y=df["Y"].to_numpy(dtype=np.float32),
                z=df["Z"].to_numpy(dtype=np.float32),
                gr=df["GR"].to_numpy(dtype=np.float32),
                tvt=tvt,
                tvt_input=tvt_input,
                ps_idx=ps,
                last_known=float(tvt_input[ps - 1]),
            )
            # 典井 hash（用于母本分组）
            tp = t_paths.get(wid)
            if tp is not None:
                self._typewell_path[wid] = tp
                self.typewell_hash[wid] = hashlib.md5(tp.read_bytes()).hexdigest()

    def ids(self) -> list[str]:
        return sorted(self.wells)

    def load_formation_tops(self, well_id: str) -> pd.DataFrame:
        """按需加载某口 train 井的 6 个 formation top 列（第 2 步 surface predictor 用）。"""
        p = self.data_root / "train" / f"{well_id}__horizontal_well.csv"
        return pd.read_csv(p, usecols=FORMATION_COLS)

    def load_typewell(self, well_id: str) -> pd.DataFrame:
        p = self._typewell_path.get(well_id)
        return pd.read_csv(p) if p is not None else pd.DataFrame(columns=["TVT", "GR", "Geology"])


# ---------------------------------------------------------------- 折分配（冻结）
def build_well_folds(ids: list[str], seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    return {ids[i]: int(perm[k] % N_FOLDS) for k, i in enumerate(range(len(ids)))}


def build_typewell_folds(store: WellStore, seed: int) -> dict[str, int]:
    """母本分组：典井 hash 相同的井绑在同一组，再对组做 KFold。"""
    ids = store.ids()
    groups: dict[str, list[str]] = {}
    for wid in ids:
        key = store.typewell_hash.get(wid, f"__solo__{wid}")
        groups.setdefault(key, []).append(wid)
    group_keys = sorted(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(group_keys))
    fold_of: dict[str, int] = {}
    for k, gk in enumerate(group_keys):
        f = int(perm[k] % N_FOLDS)
        for wid in groups[gk]:
            fold_of[wid] = f
    return fold_of


def build_spatial_folds(store: WellStore, seed: int) -> dict[str, int]:
    """空间压力：对 PS 点 XY 做 KMeans(K=N_FOLDS)，每个紧致区域 = 一折（留出整片地理区域）。"""
    ids = store.ids()
    xy = np.array([store.wells[w].ps_xy for w in ids], dtype=float)
    xy_std = (xy - xy.mean(axis=0)) / (xy.std(axis=0) + EPS)
    km = KMeans(n_clusters=N_FOLDS, random_state=seed, n_init=10).fit(xy_std)
    fold_of = {ids[i]: int(km.labels_[i]) for i in range(len(ids))}
    # 强制典井母本组原子性：同 hash 的井（多为同一井场、空间已共址）归到该组多数折，
    # 杜绝任何母本跨折泄露，让 spatial 成为最干净的压力测试。
    groups: dict[str, list[str]] = {}
    for wid in ids:
        groups.setdefault(store.typewell_hash.get(wid, f"__solo__{wid}"), []).append(wid)
    for members in groups.values():
        if len(members) > 1:
            majority = pd.Series([fold_of[w] for w in members]).mode().iloc[0]
            for w in members:
                fold_of[w] = int(majority)
    return fold_of


def get_or_build_splits(store: WellStore, refresh: bool) -> dict[str, dict[str, int]]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    builders: dict[str, Callable[[], dict[str, int]]] = {
        "well": lambda: build_well_folds(store.ids(), RANDOM_SEED),
        "typewell": lambda: build_typewell_folds(store, RANDOM_SEED + 1),
        "spatial": lambda: build_spatial_folds(store, RANDOM_SEED + 2),
    }
    out: dict[str, dict[str, int]] = {}
    for scheme in SPLIT_SCHEMES:
        path = SPLIT_DIR / f"{scheme}_folds.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, dtype={"well_id": str})
            out[scheme] = dict(zip(df["well_id"], df["fold"].astype(int)))
        else:
            fold_of = builders[scheme]()
            pd.DataFrame({"well_id": list(fold_of), "fold": list(fold_of.values())}) \
                .sort_values("well_id").to_csv(path, index=False)
            out[scheme] = fold_of
    return out


def folds_to_pairs(fold_of: dict[str, int]) -> list[tuple[list[str], list[str]]]:
    folds = sorted(set(fold_of.values()))
    pairs = []
    for f in folds:
        val = sorted(w for w, ff in fold_of.items() if ff == f)
        train = sorted(w for w, ff in fold_of.items() if ff != f)
        pairs.append((train, val))
    return pairs


# ---------------------------------------------------------------- Predictor 协议
class Predictor(Protocol):
    name: str
    def fit(self, train_ids: list[str], store: WellStore) -> None: ...
    def predict(self, val_id: str, store: WellStore) -> np.ndarray: ...


class CarryForward:
    """训练无关：隐藏段恒等于最后已知 TVT。三套 split 下分数必须完全一致（自检不变量）。"""
    name = "carry_forward"
    def fit(self, train_ids, store): pass
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]
        return np.full(w.n_hidden, w.last_known, dtype=float)


class GlobalMeanDrift:
    """用训练折：预测 = last_known + 全训练井平均 level drift。会随 split 变化（验证折机制生效）。"""
    name = "global_mean_drift"
    def __init__(self): self.drift = 0.0
    def fit(self, train_ids, store):
        drifts = [float(store.wells[w].hidden_truth.mean()) - store.wells[w].last_known
                  for w in train_ids]
        self.drift = float(np.mean(drifts)) if drifts else 0.0
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]
        return np.full(w.n_hidden, w.last_known + self.drift, dtype=float)


# ---------------------------------------------------------------- 评分
@dataclass
class WellScore:
    well_id: str
    n_hidden: int
    rmse: float
    sse: float


def score_oof(oof: dict[str, np.ndarray], store: WellStore) -> list[WellScore]:
    out = []
    for wid, pred in oof.items():
        truth = store.wells[wid].hidden_truth
        err = pred.astype(float) - truth
        sse = float(np.sum(err * err))
        n = len(truth)
        out.append(WellScore(wid, n, float(np.sqrt(sse / n)), sse))
    return out


def run_predictor(make: Callable[[], Predictor], pairs: list[tuple[list[str], list[str]]],
                  store: WellStore) -> dict[str, np.ndarray]:
    oof: dict[str, np.ndarray] = {}
    for train_ids, val_ids in pairs:
        model = make()
        model.fit(train_ids, store)
        for vid in val_ids:
            oof[vid] = model.predict(vid, store)
    return oof


# ---------------------------------------------------------------- 聚合 / 报告
def aggregate(scores: list[WellScore]) -> dict[str, float]:
    n = np.array([s.n_hidden for s in scores], dtype=float)
    sse = np.array([s.sse for s in scores], dtype=float)
    rmse = np.array([s.rmse for s in scores], dtype=float)
    return {
        "row_weighted_rmse": float(np.sqrt(sse.sum() / n.sum())),
        "per_well_rmse_mean": float(rmse.mean()),
        "per_well_rmse_p50": float(np.percentile(rmse, 50)),
        "per_well_rmse_p90": float(np.percentile(rmse, 90)),
        "per_well_rmse_max": float(rmse.max()),
        "n_wells": int(len(scores)),
    }


def buckets(scores_df: pd.DataFrame, dash: pd.DataFrame) -> pd.DataFrame:
    """按井型分桶看 row-weighted RMSE 与相对 CF 改善。"""
    df = scores_df.merge(dash, on="well_id", how="left")
    rows = []

    def rw(sub: pd.DataFrame, col: str) -> float:
        sse = (sub[col] ** 2 * sub["n_hidden"]).sum()
        return float(np.sqrt(sse / sub["n_hidden"].sum())) if sub["n_hidden"].sum() else float("nan")

    def add(name: str, label: str, sub: pd.DataFrame):
        if sub.empty:
            return
        rows.append({
            "bucket": name, "level": label, "wells": int(len(sub)),
            "hidden_rows": int(sub["n_hidden"].sum()),
            "rmse": rw(sub, "rmse"), "cf_rmse": rw(sub, "rmse_cf_ref"),
            "delta_vs_cf": rw(sub, "rmse") - rw(sub, "rmse_cf_ref"),
        })

    # 隐藏跨度四分位
    if "hidden_tvt_span" in df:
        q = pd.qcut(df["hidden_tvt_span"], 4, labels=["span_q1", "q2", "q3", "q4_long"])
        for lv in q.cat.categories:
            add("hidden_tvt_span", str(lv), df[q == lv])
    # lateral 前缀 ≈ 0
    if "lateral_prefix_len" in df:
        add("lateral_prefix", "len_0", df[df["lateral_prefix_len"] <= 5])
        add("lateral_prefix", "len_gt0", df[df["lateral_prefix_len"] > 5])
    # post GR 缺失率
    if "post_gr_missing_rate" in df:
        add("post_gr_missing", "lt_0.2", df[df["post_gr_missing_rate"] < 0.2])
        add("post_gr_missing", "0.2_0.5", df[(df["post_gr_missing_rate"] >= 0.2) & (df["post_gr_missing_rate"] < 0.5)])
        add("post_gr_missing", "ge_0.5", df[df["post_gr_missing_rate"] >= 0.5])
    # 典井检索质量
    if "assigned_typewell_rank" in df:
        add("typewell_rank", "top1_3", df[df["assigned_typewell_rank"] <= 3])
        add("typewell_rank", "rank_gt10", df[df["assigned_typewell_rank"] > 10])
    return pd.DataFrame(rows)


def write_report(results: dict, store: WellStore, splits: dict, dash: pd.DataFrame) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append("# ROGII CV Runner — 第 0 步报告\n")
    lines.append(f"生成时间：`{ts}`  ·  seed=`{RANDOM_SEED}`  ·  折数=`{N_FOLDS}`  ·  井数=`{len(store.ids())}`\n")
    lines.append("真实 masked-tail：每口井 post-PS（TVT_input 缺失）段为隐藏段，TVT 为真值。\n")

    # split 概览
    lines.append("## 1. 三套 split 概览（冻结于 `cv/splits/`）\n")
    rows = []
    for scheme in SPLIT_SCHEMES:
        fold_of = splits[scheme]
        sizes = pd.Series(list(fold_of.values())).value_counts().sort_index()
        rows.append({
            "scheme": scheme, "groups": _n_groups(scheme, store, fold_of),
            "fold_sizes": ", ".join(str(int(s)) for s in sizes),
        })
    lines.append(md_table(pd.DataFrame(rows)) + "\n")
    lines.append("> well=井级随机；typewell=典井母本（byte 相同典井绑同折，防对齐泄露）；"
                 "spatial=PS 点 XY KMeans 紧致区域（留出整片地理区，压力测试邻井/surface 外推）。\n")

    # 主结果
    lines.append("## 2. 主结果：row-weighted RMSE × predictor × split\n")
    main_rows = []
    for pname, per_split in results.items():
        row = {"predictor": pname}
        for scheme in SPLIT_SCHEMES:
            row[scheme] = per_split[scheme]["agg"]["row_weighted_rmse"]
        main_rows.append(row)
    lines.append(md_table(pd.DataFrame(main_rows)) + "\n")
    lines.append("> 自检不变量：`carry_forward` 训练无关 → 三列必须完全相等且 ≈ 15.91（对账 eda）。"
                 "`global_mean_drift` 用训练折 → 列间应有细微差异（证明折机制生效）。\n")

    # per-well 分布 + worst-by-SSE
    for pname, per_split in results.items():
        lines.append(f"## 3. {pname} · per-well 分布与 worst-well\n")
        dist_rows = []
        for scheme in SPLIT_SCHEMES:
            a = per_split[scheme]["agg"]
            dist_rows.append({"split": scheme, **{k: a[k] for k in
                              ["per_well_rmse_mean", "per_well_rmse_p50",
                               "per_well_rmse_p90", "per_well_rmse_max"]}})
        lines.append(md_table(pd.DataFrame(dist_rows)) + "\n")
        # worst by SSE (用 spatial split 的 OOF)
        sdf = per_split["spatial"]["per_well"].copy()
        sdf["sse_share"] = sdf["sse"] / sdf["sse"].sum()
        worst = sdf.sort_values("sse", ascending=False).head(20)[
            ["well_id", "n_hidden", "rmse", "sse", "sse_share"]]
        lines.append(f"**worst-well 按 SSE 排（spatial split，top 20，占总 SSE 比）：**\n")
        lines.append(md_table(worst, floatfmt=".4f") + "\n")

    # 分桶（用 global_mean_drift / spatial 演示，含相对 CF 改善）
    lines.append("## 4. 按井型分桶（spatial split · 含相对 CF 改善）\n")
    for pname in results:
        sdf = results[pname]["spatial"]["per_well"].copy()
        sdf = sdf.merge(dash[["well_id", "rmse_cf"]].rename(columns={"rmse_cf": "rmse_cf_ref"}),
                        on="well_id", how="left")
        bdf = buckets(sdf, dash)
        lines.append(f"### {pname}\n")
        lines.append(md_table(bdf) + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _n_groups(scheme: str, store: WellStore, fold_of: dict[str, int]) -> int:
    if scheme == "typewell":
        keys = {store.typewell_hash.get(w, f"__solo__{w}") for w in fold_of}
        return len(keys)
    if scheme == "spatial":
        return N_FOLDS
    return len(fold_of)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-splits", action="store_true", help="重新随机生成并覆盖冻结 split")
    args = ap.parse_args()

    ART_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] 加载井数据 ...")
    store = WellStore()
    print(f"      井数={len(store.ids())}；隐藏行合计={sum(store.wells[w].n_hidden for w in store.ids()):,}")

    print("[2/4] 构建/复用冻结 split ...")
    splits = get_or_build_splits(store, refresh=args.refresh_splits)
    pairs = {s: folds_to_pairs(splits[s]) for s in SPLIT_SCHEMES}

    dash = pd.read_csv(DASHBOARD, dtype={"well_id": str})

    print("[3/4] 跑自检 baseline（carry_forward / global_mean_drift）× 三套 split ...")
    makers: dict[str, Callable[[], Predictor]] = {
        "carry_forward": CarryForward,
        "global_mean_drift": GlobalMeanDrift,
    }
    results: dict[str, dict] = {}
    for pname, make in makers.items():
        results[pname] = {}
        for scheme in SPLIT_SCHEMES:
            oof = run_predictor(make, pairs[scheme], store)
            scores = score_oof(oof, store)
            per_well = pd.DataFrame([{"well_id": s.well_id, "n_hidden": s.n_hidden,
                                      "rmse": s.rmse, "sse": s.sse} for s in scores])
            per_well.sort_values("sse", ascending=False).to_csv(
                ART_DIR / f"{pname}__{scheme}__per_well.csv", index=False)
            results[pname][scheme] = {"agg": aggregate(scores), "per_well": per_well}
            a = results[pname][scheme]["agg"]
            print(f"      {pname:18s} {scheme:9s} rw_rmse={a['row_weighted_rmse']:.4f} "
                  f"p50={a['per_well_rmse_p50']:.3f} p90={a['per_well_rmse_p90']:.3f} max={a['per_well_rmse_max']:.3f}")

    print("[4/4] 写报告 ...")
    write_report(results, store, splits, dash)
    print(f"      报告: {REPORT}")
    print(f"      折分配: {SPLIT_DIR}/{{well,typewell,spatial}}_folds.csv")
    print(f"      per-well artifacts: {ART_DIR}/")


if __name__ == "__main__":
    main()
