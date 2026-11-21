"""Single grid-search experiment for the 5-class cell-phase classifier.

Axes (validation/test always stay complete; only the training mix changes):
  I = real interphase crops kept in train/val  (--grid-i, "full" = whole ~26k pool)
  R = real crops per class (uniform cap, interphase included)   (--grid-r, "full" = no cap)
  S = synthetic images per class added to train                 (--grid-s, 0 = real-only)
  G = generator pool the synth images came from                 (--synth-dirs name=path)

Every (I,R,S,G) cell is additionally repeated per training seed (--seeds) and stored under
``<out>/seed{N}/i{I}_r{R}_s{S}_g{G}/`` so across-seed stats are always available.

Dataset for an I value: ``datasets/crops/mitosis_phase5`` for 500 (the standard DVC output)
and ``datasets/crops/mitosis_phase5_ic{I}`` otherwise; ``--prepare`` builds missing ones.

Stage presets (explicit flags override them):
  1 = original R x S grid at I=500, G=full
  2 = G axis (I=500, R=50, S in {0,100,full})
  3 = I axis at R=full, S in {0,250}
  all = I x R x S cross (tractable subset)

Usage:
    uv run python scripts/sweep_mitosis_data_budget.py --stage 1 --dry-run
    uv run python scripts/sweep_mitosis_data_budget.py --stage 3 --prepare --seeds 42 43 44
    uv run python scripts/sweep_mitosis_data_budget.py --grid-i 500 4000 full \\
        --grid-r full --grid-s 0 250 --seeds 42 43 44
    uv run python scripts/sweep_mitosis_data_budget.py --aggregate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
import yaml

from allium_cepa_classifier.config.base_config import find_project_root
from allium_cepa_classifier.config.mitosis_phase_config import MitosisPhaseConfig
from allium_cepa_classifier.training.mitosis_phase_trainer import (
    CELL_CYCLE_ORDER,
    run_training,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("experiments/mitosis_classifier/phase5_real_plus_synth/config.yaml")
DEFAULT_OUT = Path("experiments/mitosis_classifier/sweep")
DEFAULT_SYNTH = "full=datasets/crops/mitosis_phase5_synth"
DEFAULT_VAE = Path("datasets/crops/vae_256")
INTERPHASE_CAPS = [500, 1000, 2000, 4000, 8000, 16000, None]

# stage -> (grid_i tokens, grid_r tokens, grid_s tokens)
STAGE_GRIDS = {
    1: (["500"], ["25", "50", "100", "200", "full"], ["0", "25", "50", "100", "250", "full"]),
    2: (["500"], ["50"], ["0", "100", "full"]),
    3: (["500", "1000", "2000", "4000", "8000", "16000", "full"], ["full"], ["0", "250"]),
    "all": (
        ["500", "1000", "2000", "4000", "8000", "16000", "full"],
        ["25", "100", "250", "full"],
        ["0", "250", "full"],
    ),
}

CELL_RE = re.compile(r"^i(?P<i>[0-9]+|full)_r(?P<r>[0-9]+|full)_s(?P<s>[0-9]+|full)_g(?P<g>[^_]+)$")
_TRAIN_COUNT_RE = re.compile(r"train\[(?P<cls>[^\]]+)\] = (?P<n>\d+)")
_FULL_TOKENS = {"full", "all", "-1"}


def _stage_value(tok: str):
    return tok if tok == "all" else int(tok)


def label(v: int | None) -> str:
    return "full" if v is None else str(v)


def budget_key(v: int | None):
    """Sort key: ascending numbers, None ("full") last."""
    return (1, 0) if v is None else (0, v)


def parse_budget(tok: str, *, minimum: int) -> int | None:
    if tok.lower() in _FULL_TOKENS:
        return None
    try:
        v = int(tok)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad budget {tok!r}: expected int or 'full'") from None
    if v < minimum:
        raise argparse.ArgumentTypeError(f"bad budget {tok!r}: minimum is {minimum}")
    return v


def parse_synth_dirs(tokens: list[str]) -> dict[str, Path]:
    pools: dict[str, Path] = {}
    for tok in tokens:
        name, sep, path = tok.partition("=")
        if not sep:
            path = name
        key = re.sub(r"[^0-9A-Za-z_.-]+", "-", name)
        if key in pools:
            raise SystemExit(f"Duplicate synth pool name: {name!r}")
        pools[key] = Path(path)
    return pools


def interphase_dataset_dir(root: Path, cap: int | None) -> Path:
    """Dataset dir for an interphase cap (500 is the standard DVC output)."""
    if cap == 500:
        return root / "datasets" / "crops" / "mitosis_phase5"
    return root / "datasets" / "crops" / f"mitosis_phase5_ic{label(cap)}"


def prepare_interphase_datasets(
    root: Path, vae_dir: Path, caps: list[int | None], classes: list[str], *, prepare: bool
) -> list[str]:
    """Ensure one dataset dir exists per I value; build missing ones when --prepare."""
    problems: list[str] = []
    for cap in caps:
        out = interphase_dataset_dir(root, cap)
        if (out / "train" / "interphase").is_dir():
            continue
        if not prepare:
            problems.append(f"missing dataset dir {out} (rerun with --prepare)")
            continue
        cmd = [
            sys.executable,
            str(root / "scripts" / "utils" / "prepare_mitosis_crops.py"),
            "--vae-dir",
            str(vae_dir),
            "--out",
            str(out),
            "--classes",
            *classes,
            "--interphase-cap",
            "-1" if cap is None else str(cap),
        ]
        log.info(f"[prep] I={label(cap)} -> {out}")
        subprocess.run(cmd, check=True)
    return problems


@dataclass
class Cell:
    name: str
    i: int | None
    r: int | None
    s: int | None
    g: str
    run_dir: Path
    cfg: MitosisPhaseConfig
    seed: int


def build_matrix(
    base: MitosisPhaseConfig,
    root: Path,
    grid_i: list[int | None],
    grid_r: list[int | None],
    grid_s: list[int | None],
    pools: dict[str, Path],
    out_root: Path,
    seeds: list[int] | None = None,
    vary: str = "both",
) -> list[Cell]:
    explicit = seeds is not None
    seeds = seeds or [base.data.seed]
    cells: list[Cell] = []
    for seed in seeds:
        for g, pool in pools.items():
            for i in sorted(grid_i, key=budget_key):
                for r in sorted(grid_r, key=budget_key):
                    for s in sorted(grid_s, key=budget_key):
                        name = f"i{label(i)}_r{label(r)}_s{label(s)}_g{g}"
                        cfg = base.model_copy(deep=True)
                        cfg.experiment_name = name
                        cfg.data.crops_dir = interphase_dataset_dir(root, i)
                        cfg.data.train_max_per_class = r
                        cfg.data.include_synthetic = s != 0
                        cfg.data.synthetic_max_per_class = None if s == 0 else s
                        cfg.data.synthetic_dir = pool
                        cfg.data.save_misclassifications = False
                        if explicit:
                            if vary in ("both", "train"):
                                cfg.data.seed = seed
                            if vary in ("both", "data"):
                                cfg.data.budget_seed = seed
                        cells.append(
                            Cell(name, i, r, s, g, out_root / f"seed{seed}" / name, cfg, seed)
                        )
    return cells


def _validate_data(cells: list[Cell]) -> list[str]:
    problems: list[str] = []
    checked: set[Path] = set()
    for cell in cells:
        crops = cell.cfg.data.crops_dir
        if crops not in checked:
            checked.add(crops)
            for split in ("train", "validation", "test"):
                if not (crops / split).is_dir():
                    problems.append(f"missing split dir: {crops / split}")
        if cell.s != 0 and not cell.cfg.data.synthetic_dir.is_dir():
            problems.append(f"{cell.name}: missing synth pool {cell.cfg.data.synthetic_dir}")
    return sorted(set(problems))


def print_matrix(cells: list[Cell]) -> None:
    print(f"\n{'cell':<32}{'seed':>6} {'I':>6} {'R':>5} {'S':>5}  {'G':<10} crops_dir")
    for c in cells:
        print(
            f"{c.name:<32}{c.seed:>6} {label(c.i):>6} {label(c.r):>5} {label(c.s):>5}  "
            f"{c.g:<10} {c.cfg.data.crops_dir}"
        )
    print(f"Total cells: {len(cells)}")


def run_cells(cells: list[Cell], force: bool) -> tuple[list[str], list[str], list[str]]:
    done: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    root_logger = logging.getLogger()
    for cell in cells:
        if (cell.run_dir / "metrics.json").exists() and not force:
            log.info(f"[skip] {cell.name} seed={cell.seed}")
            skipped.append(cell.name)
            continue
        cell.run_dir.mkdir(parents=True, exist_ok=True)
        (cell.run_dir / "weights").mkdir(exist_ok=True)
        (cell.run_dir / "plots").mkdir(exist_ok=True)
        (cell.run_dir / "used_config.yaml").write_text(
            yaml.safe_dump(cell.cfg.model_dump(mode="json"), sort_keys=False)
        )
        handler = logging.FileHandler(cell.run_dir / "train.log")
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
        log.info(
            f"[run ] {cell.name}: I={label(cell.i)} R={label(cell.r)} S={label(cell.s)} "
            f"G={cell.g} seed={cell.seed}"
        )
        try:
            run_training(cell.cfg, cell.run_dir)
            done.append(cell.name)
            log.info(f"[done] {cell.name} seed={cell.seed}")
        except Exception:
            log.exception(f"[fail] {cell.name} seed={cell.seed}")
            failed.append(cell.name)
        finally:
            root_logger.removeHandler(handler)
            handler.close()
    return done, skipped, failed


# ---------------------------------------------------------------- aggregation


def _parse_cell_dir(d: Path) -> tuple[int | None, int | None, int | None, str] | None:
    m = CELL_RE.match(d.name)
    if not m:
        return None
    i = None if m["i"] == "full" else int(m["i"])
    r = None if m["r"] == "full" else int(m["r"])
    s = None if m["s"] == "full" else int(m["s"])
    return i, r, s, m["g"]


def _iter_cell_dirs(out_root: Path) -> list[Path]:
    if not out_root.is_dir():
        return []
    dirs: list[Path] = []
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        if _parse_cell_dir(d):
            dirs.append(d)
        elif re.fullmatch(r"seed[0-9]+", d.name):
            dirs.extend(c for c in sorted(d.iterdir()) if c.is_dir() and _parse_cell_dir(c))
    return dirs


def _config_seeds(cell_dir: Path) -> tuple[int | None, int | None]:
    """(data.seed, data.budget_seed) recorded in the cell's used_config.yaml."""
    cfg_path = cell_dir / "used_config.yaml"
    if not cfg_path.exists():
        return None, None
    try:
        data = (yaml.safe_load(cfg_path.read_text()) or {}).get("data", {}) or {}
        seed = data.get("seed")
        budget_seed = data.get("budget_seed")
        return (
            int(seed) if seed is not None else None,
            int(budget_seed) if budget_seed is not None else None,
        )
    except (yaml.YAMLError, AttributeError, TypeError, ValueError):
        return None, None


def _per_class_counts_from_log(log_path: Path) -> dict[str, int] | None:
    """Recover train composition from a cell's train.log (fallback for old runs)."""
    if not log_path.exists():
        return None
    counts = {
        m["cls"]: int(m["n"]) for m in _TRAIN_COUNT_RE.finditer(log_path.read_text(errors="ignore"))
    }
    return counts or None


CI_METRICS = ("macro_f1", "balanced_acc", "mit_only_f1", "bin_f1", "no_pro_f1")


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    flat = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


def _f1_from_counts(cm: np.ndarray, idx: int) -> float:
    tp = cm[idx, idx]
    fp = cm[:, idx].sum() - tp
    fn = cm[idx, :].sum() - tp
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def _metric_values(y_true, y_pred, class_names: list[str]) -> dict[str, float]:
    """Metric battery from one cell's predictions (all derived from the KxK counts).

    mit_only_f1: 4-class macro-F1 computed on true-mitotic rows only (interphase predictions
    count as FN, interphase true cells are ignored) -> isolates phase discrimination from the
    90%-interphase test set. no_pro_f1: macro-F1 over metaphase/anaphase/telophase with
    prophase (known labeling issue) and interphase predictions treated as errors.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    k = len(class_names)
    cm = _confusion(y_true, y_pred, k)
    ii = class_names.index("interphase")
    mit = [i for i in range(k) if i != ii]
    pro = class_names.index("prophase") if "prophase" in class_names else -1

    recalls = [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() else 0.0 for i in range(k)]
    cm_mit = cm.copy()
    cm_mit[ii, :] = 0

    tp_b = cm[np.ix_(mit, mit)].sum()
    fp_b = cm[ii, mit].sum()
    fn_b = cm[np.ix_(mit, [ii])].sum()
    prec_b = tp_b / (tp_b + fp_b) if tp_b + fp_b else 0.0
    rec_b = tp_b / (tp_b + fn_b) if tp_b + fn_b else 0.0
    bin_f1 = 2 * prec_b * rec_b / (prec_b + rec_b) if prec_b + rec_b else 0.0

    return {
        "macro_f1": float(np.mean([_f1_from_counts(cm, i) for i in range(k)])),
        "balanced_acc": float(np.mean(recalls)),
        "mit_only_f1": float(np.mean([_f1_from_counts(cm_mit, i) for i in mit])),
        "bin_f1": float(bin_f1),
        "no_pro_f1": float(np.mean([_f1_from_counts(cm, i) for i in mit if i != pro])),
        "mitotic_recall": float(rec_b),
        "interphase_fpr": float(fp_b / (fp_b + cm[ii, ii])) if (fp_b + cm[ii, ii]) else 0.0,
    }


def bootstrap_ci(
    y_true, y_pred, class_names: list[str], n_boot: int = 500, seed: int = 1234
) -> dict[str, tuple[float, float]]:
    """Stratified paired bootstrap over test samples -> 95% CIs (test-set sampling only)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = [g for g in (np.flatnonzero(y_true == c) for c in range(len(class_names))) if g.size]
    samples = {k: np.empty(n_boot) for k in CI_METRICS}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        vals = _metric_values(y_true[idx], y_pred[idx], class_names)
        for key in CI_METRICS:
            samples[key][b] = vals[key]
    return {
        key: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for key, v in samples.items()
    }


def aggregate(out_root: Path) -> list[dict]:
    rows: list[dict] = []
    for d in _iter_cell_dirs(out_root):
        parsed = _parse_cell_dir(d)
        metrics_path = d / "metrics.json"
        if parsed is None or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        i, r, s, g = parsed
        seed, budget_seed = _config_seeds(d)
        row = {
            "cell": d.name,
            "seed": seed,
            "budget_seed": budget_seed,
            "i": i,
            "i_label": label(i),
            "g": g,
            "r": r,
            "s": s,
            "r_label": label(r),
            "s_label": label(s),
            "include_synthetic": metrics["include_synthetic"],
            "synthetic_max_per_class": metrics.get("synthetic_max_per_class"),
            "n_train": metrics.get("n_train"),
            "n_train_real": metrics.get("n_train_real"),
            "n_train_synth": metrics.get("n_train_synth"),
            "n_train_per_class": metrics.get("n_train_per_class")
            or _per_class_counts_from_log(d / "train.log"),
            "train_acc": metrics.get("train_acc"),
            "val_acc": metrics.get("val_acc"),
            "test_acc": metrics.get("test_acc"),
            "test_macro_f1": metrics.get("test_macro_f1"),
            "test_balanced_acc": metrics.get("test_balanced_acc"),
            "best_val_loss": metrics.get("best_val_loss"),
            "epochs_run": metrics.get("epochs_run"),
            "classes": metrics.get("classes", []),
            "per_class_f1": metrics.get("per_class_f1", {}),
            "per_class_precision": metrics.get("per_class_precision", {}),
            "per_class_recall": metrics.get("per_class_recall", {}),
            "per_class_support": metrics.get("per_class_support", {}),
            "confusion_matrix": metrics.get("confusion_matrix"),
        }
        pred_path = d / "predictions.npz"
        if pred_path.exists() and metrics.get("classes"):
            z = np.load(pred_path)
            mv = _metric_values(z["y_true"], z["y_pred"], metrics["classes"])
            if (
                metrics.get("test_macro_f1") is not None
                and abs(mv["macro_f1"] - metrics["test_macro_f1"]) > 1e-6
            ):
                log.warning(
                    f"{d.name}: recomputed macro-F1 {mv['macro_f1']:.6f} != "
                    f"metrics.json {metrics['test_macro_f1']:.6f} (class-order mismatch?)"
                )
            row.update(
                {
                    "test_mit_only_f1": mv["mit_only_f1"],
                    "test_bin_f1": mv["bin_f1"],
                    "test_no_pro_f1": mv["no_pro_f1"],
                    "test_mitotic_recall": mv["mitotic_recall"],
                    "test_interphase_fpr": mv["interphase_fpr"],
                }
            )
            for key, (lo, hi) in bootstrap_ci(z["y_true"], z["y_pred"], metrics["classes"]).items():
                row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = lo, hi
        else:
            for key in (
                "test_mit_only_f1",
                "test_bin_f1",
                "test_no_pro_f1",
                "test_mitotic_recall",
                "test_interphase_fpr",
            ):
                row[key] = None
            for key in CI_METRICS:
                row[f"{key}_ci_lo"] = row[f"{key}_ci_hi"] = None
        rows.append(row)
    rows.sort(
        key=lambda x: (
            x["g"],
            budget_key(x["i"]),
            budget_key(x["s"]),
            budget_key(x["r"]),
            -1 if x["seed"] is None else x["seed"],
            x["cell"],
        )
    )
    return rows


def write_results(rows: list[dict], out_root: Path) -> None:
    base_cols = [
        "cell",
        "seed",
        "budget_seed",
        "i",
        "i_label",
        "g",
        "r",
        "s",
        "r_label",
        "s_label",
        "include_synthetic",
        "synthetic_max_per_class",
        "n_train",
        "n_train_real",
        "n_train_synth",
        "train_acc",
        "val_acc",
        "test_acc",
        "test_macro_f1",
        "macro_f1_ci_lo",
        "macro_f1_ci_hi",
        "test_balanced_acc",
        "balanced_acc_ci_lo",
        "balanced_acc_ci_hi",
        "test_mit_only_f1",
        "mit_only_f1_ci_lo",
        "mit_only_f1_ci_hi",
        "test_bin_f1",
        "bin_f1_ci_lo",
        "bin_f1_ci_hi",
        "test_no_pro_f1",
        "no_pro_f1_ci_lo",
        "no_pro_f1_ci_hi",
        "test_mitotic_recall",
        "test_interphase_fpr",
        "best_val_loss",
        "epochs_run",
    ]
    prefix_map = {
        "per_class_f1": "f1",
        "per_class_precision": "prec",
        "per_class_recall": "rec",
        "per_class_support": "sup",
    }
    per_class_cols = sorted(
        {f"{p}_{c}" for row in rows for key, p in prefix_map.items() for c in row[key]}
    )
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + per_class_cols)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in base_cols}
            flat.update(
                {f"{p}_{c}": v for key, p in prefix_map.items() for c, v in row[key].items()}
            )
            writer.writerow(flat)
    (out_root / "results.json").write_text(json.dumps(rows, indent=2))


SEED_STAT_METRICS = (
    "test_macro_f1",
    "test_balanced_acc",
    "test_mit_only_f1",
    "test_bin_f1",
    "test_no_pro_f1",
    "test_interphase_fpr",
    "test_mitotic_recall",
)


def seed_stats(rows: list[dict]) -> list[dict]:
    """Per-(I,R,S,G) mean/sd across seeds + paired delta vs S=0 at the same seed."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["g"], row["i"], row["r"], row["s"]), []).append(row)
    out: list[dict] = []
    for (g, i, r, s), group in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], budget_key(kv[0][1]), budget_key(kv[0][3]), budget_key(kv[0][2])),
    ):
        baselines = {row["seed"]: row for row in groups.get((g, i, r, 0), [])}
        for metric in SEED_STAT_METRICS:
            vals = [row[metric] for row in group if row.get(metric) is not None]
            deltas = [
                row[metric] - baselines[row["seed"]][metric]
                for row in group
                if s != 0
                and row.get(metric) is not None
                and row.get("seed") is not None
                and baselines.get(row["seed"]) is not None
                and baselines[row["seed"]].get(metric) is not None
            ]
            out.append(
                {
                    "g": g,
                    "i": i,
                    "i_label": label(i),
                    "r": r,
                    "s": s,
                    "r_label": label(r),
                    "s_label": label(s),
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else None,
                    "sd": float(np.std(vals, ddof=1)) if len(vals) >= 2 else None,
                    "n_pairs": len(deltas),
                    "delta_mean": float(np.mean(deltas)) if deltas else None,
                    "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else None,
                }
            )
    return out


def write_seed_stats(rows: list[dict], out_root: Path) -> None:
    stats = seed_stats(rows)
    if not stats:
        return
    cols = [
        "g",
        "i",
        "i_label",
        "r",
        "s",
        "r_label",
        "s_label",
        "metric",
        "n",
        "mean",
        "sd",
        "n_pairs",
        "delta_mean",
        "delta_sd",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "results_seedstats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(stats)


# ---------------------------------------------------------------- plotting


def _phase_order(classes: list[str]) -> list[str]:
    return sorted(
        classes,
        key=lambda c: (CELL_CYCLE_ORDER.index(c) if c in CELL_CYCLE_ORDER else 99, c),
    )


def _series(rows: list[dict], x_vals: list, x_key: str, metric: str):
    """[(x, mean, [per-seed values], representative row)] for one series."""
    out = []
    for x in x_vals:
        group = [r for r in rows if r[x_key] == x]
        vals = [r[metric] for r in group if r.get(metric) is not None]
        if not vals:
            continue
        out.append((x, float(np.mean(vals)), vals, group[0]))
    return out


def plot_metric_vs(
    rows: list[dict],
    metric: str,
    ylabel: str,
    out_png: Path,
    *,
    x_key: str,
    ci: tuple[str, str] | None = None,
    row_key: str | None = None,
    col_key: str = "g",
    series_key: str = "s",
) -> None:
    if not rows:
        return
    x_vals = sorted({r[x_key] for r in rows}, key=budget_key)
    series_vals = sorted({r[series_key] for r in rows}, key=budget_key)
    cols = sorted({r[col_key] for r in rows})
    row_vals = sorted({r[row_key] for r in rows}, key=budget_key) if row_key else [None]
    nrows, ncols = len(row_vals), len(cols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(4.5, 3.8 * ncols), max(3.4, 3.0 * nrows)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ri, rv in enumerate(row_vals):
        for ci_, cv in enumerate(cols):
            ax = axes[ri][ci_]
            facet = [r for r in rows if r[col_key] == cv and (row_key is None or r[row_key] == rv)]
            for s in series_vals:
                pts = _series([r for r in facet if r[series_key] == s], x_vals, x_key, metric)
                if not pts:
                    continue
                xs = [x_vals.index(p[0]) for p in pts]
                ys = [p[1] for p in pts]
                per_seed = [len([v for v in p[2] if v is not None]) for p in pts]
                (line,) = ax.plot(xs, ys, marker="o", label=f"{series_key.upper()}={label(s)}")
                color = line.get_color()
                if max(per_seed) >= 2:
                    sd = [float(np.std(p[2], ddof=1)) if len(p[2]) >= 2 else 0.0 for p in pts]
                    ax.errorbar(xs, ys, yerr=sd, fmt="none", ecolor=color, capsize=2)
                    for x, _, vals, _ in pts:
                        ax.scatter(
                            [x_vals.index(x)] * len(vals),
                            vals,
                            color=color,
                            alpha=0.35,
                            s=10,
                            marker=".",
                        )
                elif ci is not None:
                    lo = [p[3].get(ci[0]) for p in pts]
                    hi = [p[3].get(ci[1]) for p in pts]
                    if all(v is not None for v in lo) and all(v is not None for v in hi):
                        ax.errorbar(
                            xs,
                            ys,
                            yerr=[
                                [ys[i] - lo[i] for i in range(len(ys))],
                                [hi[i] - ys[i] for i in range(len(ys))],
                            ],
                            marker="o",
                            capsize=2,
                            color=color,
                        )
            ax.set_xticks(range(len(x_vals)), [label(v) for v in x_vals])
            title = []
            if row_key:
                title.append(f"{row_key.upper()}={label(rv)}")
            title.append(f"{col_key.upper()}={cv}")
            ax.set_title(" ".join(title))
            ax.set_xlabel(x_key.upper())
            ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="lower center", ncol=min(len(series_vals), 4), fontsize=8)
    fig.suptitle(f"{ylabel} vs {x_key.upper()} (line per {series_key.upper()})", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_per_class_f1(rows: list[dict], out_png: Path, i_filter: int | None = 500) -> None:
    sub = [r for r in rows if i_filter is None or r["i"] == i_filter]
    if not sub:
        return
    classes = _phase_order(list(sub[0]["classes"]))
    for row in sub:
        for c in row["per_class_f1"]:
            if c not in classes:
                classes.append(c)
    r_min = (
        min(row["r"] for row in sub if row["r"] is not None)
        if any(row["r"] is not None for row in sub)
        else None
    )
    sub = [row for row in sub if row["r"] == r_min]
    gs = sorted({row["g"] for row in sub})
    ss = sorted({row["s"] for row in sub}, key=budget_key)
    fig, axes = plt.subplots(
        1,
        len(gs),
        figsize=(max(6.0, 2.6 * len(classes) * len(gs) ** 0.5), 0.5 * len(ss) + 2),
        squeeze=False,
    )
    for gi, g in enumerate(gs):
        ax = axes[0][gi]
        mat = np.full((len(ss), len(classes)), np.nan)
        buckets: dict[tuple[int, int], list[float]] = {}
        for row in sub:
            if row["g"] != g:
                continue
            si = ss.index(row["s"])
            for ci_, c in enumerate(classes):
                value = row["per_class_f1"].get(c)
                if value is not None:
                    buckets.setdefault((si, ci_), []).append(value)
        for (si, ci_), values in buckets.items():
            mat[si, ci_] = float(np.mean(values))
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="rocket_r",
            xticklabels=classes,
            yticklabels=[f"S={label(s)}" for s in ss],
            cbar=bool(gi == len(gs) - 1),
            mask=np.isnan(mat),
            ax=ax,
        )
        ax.set_title(f"G = {g}")
        ax.set_xlabel("")
    axes[0][0].set_ylabel("Synthetic budget (S)")
    fig.suptitle(f"Per-class F1 at I={label(i_filter)} R={label(r_min)}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def make_plots(rows: list[dict], out_root: Path, i_filter: int | None = 500) -> None:
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "test_macro_f1",
            "Test macro-F1 (5-class)",
            "macro_f1",
            ("macro_f1_ci_lo", "macro_f1_ci_hi"),
        ),
        (
            "test_balanced_acc",
            "Test balanced accuracy",
            "balanced_acc",
            ("balanced_acc_ci_lo", "balanced_acc_ci_hi"),
        ),
        (
            "test_mit_only_f1",
            "Mitotic-only 4-class macro-F1",
            "mit_only_f1",
            ("mit_only_f1_ci_lo", "mit_only_f1_ci_hi"),
        ),
        (
            "test_bin_f1",
            "Binary mitosis-vs-interphase F1",
            "bin_f1",
            ("bin_f1_ci_lo", "bin_f1_ci_hi"),
        ),
    ]
    for metric, ylabel, slug, ci in specs:
        plot_metric_vs(
            rows,
            metric,
            ylabel,
            plots / f"{slug}_vs_real_budget.png",
            x_key="r",
            ci=ci,
            row_key="i",
            col_key="g",
            series_key="s",
        )
        full_rows = [r for r in rows if r["r"] is None]
        if full_rows and len({r["i"] for r in full_rows}) > 1:
            plot_metric_vs(
                full_rows,
                metric,
                ylabel,
                plots / f"{slug}_vs_interphase_budget.png",
                x_key="i",
                ci=ci,
                row_key=None,
                col_key="g",
                series_key="s",
            )
    plot_metric_vs(
        rows,
        "test_acc",
        "Test accuracy",
        plots / "test_acc_vs_real_budget.png",
        x_key="r",
        row_key="i",
        col_key="g",
        series_key="s",
    )
    plot_per_class_f1(rows, plots / "per_class_f1_heatmap.png", i_filter=i_filter)


# ---------------------------------------------------------------- reporting


def _fmt(value, ndigits: int = 4) -> str:
    return f"{value:>{ndigits + 4}.{ndigits}f}" if value is not None else f"{'--':>{ndigits + 4}}"


def print_summary_table(rows: list[dict]) -> None:
    print(
        f"\n{'cell':<32}{'seed':>5}{'I':>6}{'R':>5}{'S':>5}{'n_real':>8}{'n_synth':>8}"
        f"{'macroF1':>9}{'balAcc':>9}{'mitOnly':>9}{'binF1':>9}"
    )
    for r in rows:
        print(
            f"{r['cell']:<32}{str(r['seed']):>5}{r['i_label']:>6}{r['r_label']:>5}"
            f"{r['s_label']:>5}{str(r['n_train_real']):>8}{str(r['n_train_synth']):>8}"
            f"{_fmt(r.get('test_macro_f1'))}{_fmt(r.get('test_balanced_acc'))}"
            f"{_fmt(r.get('test_mit_only_f1'))}{_fmt(r.get('test_bin_f1'))}"
        )


def print_delta_table(
    rows: list[dict],
    metrics: tuple[str, ...] = ("test_macro_f1", "test_balanced_acc", "test_mit_only_f1"),
) -> None:
    """Paired mean delta vs S=0: per seed, then averaged over S>0 and seeds."""
    lookup = {(r["g"], r["i"], r["r"], r["s"], r["seed"]): r for r in rows}
    combos = sorted(
        {(r["g"], r["i"], r["r"]) for r in rows},
        key=lambda k: (k[0], budget_key(k[1]), budget_key(k[2])),
    )
    s_vals = sorted({r["s"] for r in rows if r["s"] != 0}, key=budget_key)
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    print("\nMean paired delta vs S=0 (per seed, then averaged over S>0 and seeds):")
    print(f"{'G':<8}{'I':>6}{'R':>6}" + "".join(f" {m.replace('test_', ''):>13}" for m in metrics))
    for g, i, r in combos:
        cells = []
        for m in metrics:
            values = []
            for s in s_vals:
                for seed in seeds:
                    cell = lookup.get((g, i, r, s, seed))
                    base = lookup.get((g, i, r, 0, seed))
                    if (
                        cell is not None
                        and base is not None
                        and cell.get(m) is not None
                        and base.get(m) is not None
                    ):
                        values.append(cell[m] - base[m])
            cells.append(f" {np.mean(values):+13.3f}" if values else f"{'--':>14}")
        print(f"{g:<8}{label(i):>6}{label(r):>6}" + "".join(cells))


def print_seed_delta_table(rows: list[dict]) -> None:
    metrics = ("test_macro_f1", "test_balanced_acc", "test_mit_only_f1")
    stats = [
        st
        for st in seed_stats(rows)
        if st["s"] != 0 and st["metric"] in metrics and st["n_pairs"] >= 2
    ]
    if not stats:
        return
    table: dict[tuple, dict] = {}
    for st in stats:
        table.setdefault((st["g"], st["i"], st["r"], st["s"], st["n_pairs"]), {})[st["metric"]] = st
    print("\nPaired seed deltas vs S=0 (mean ± sd over seeds; needs >=2 paired seeds):")
    print(
        f"{'G':<8}{'I':>6}{'R':>6}{'S':>6}{'n':>4}"
        + "".join(f" {m.replace('test_', ''):^21}" for m in metrics)
    )
    for (g, i, r, s, n_pairs), per_metric in sorted(
        table.items(),
        key=lambda kv: (kv[0][0], budget_key(kv[0][1]), budget_key(kv[0][2]), budget_key(kv[0][3])),
    ):
        cells = []
        for metric in metrics:
            st = per_metric.get(metric)
            if st is None or st["delta_mean"] is None:
                cells.append(f" {'--':^21}")
            elif st["delta_sd"] is None:
                cells.append(f" {st['delta_mean']:+.3f}{'':^16}")
            else:
                cells.append(f" {st['delta_mean']:+.3f} ± {st['delta_sd']:.3f}".ljust(22))
        print(f"{g:<8}{label(i):>6}{label(r):>6}{label(s):>6}{n_pairs:>4}" + "".join(cells))


def _aggregate_and_write(rows_root: Path, i_filter: int | None) -> list[dict]:
    rows = aggregate(rows_root)
    write_results(rows, rows_root)
    write_seed_stats(rows, rows_root)
    make_plots(rows, rows_root, i_filter=i_filter)
    print_summary_table(rows)
    print_delta_table(rows)
    print_seed_delta_table(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base experiment config (yes/no-synth and model/training defaults).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Study root directory.")
    parser.add_argument(
        "--stage",
        type=_stage_value,
        choices=[1, 2, 3, "all"],
        default=1,
        help="Preset grids (explicit --grid-* flags override them).",
    )
    parser.add_argument(
        "--grid-i",
        "--interphase-caps",
        nargs="+",
        metavar="I",
        default=None,
        help="Real interphase train/val caps. Accepts 'full'.",
    )
    parser.add_argument(
        "--grid-r",
        nargs="+",
        metavar="R",
        default=None,
        help="Real per-class caps; 'full' = whole pool.",
    )
    parser.add_argument(
        "--grid-s",
        nargs="+",
        metavar="S",
        default=None,
        help="Synthetic per-class caps (0 = real_only).",
    )
    parser.add_argument(
        "--synth-dirs",
        nargs="+",
        metavar="NAME=PATH",
        default=None,
        help=f"G pools (the G axis). Default: {DEFAULT_SYNTH}.",
    )
    parser.add_argument(
        "--vae-dir",
        type=Path,
        default=DEFAULT_VAE,
        help="Source 256px crops for --prepare (default datasets/crops/vae_256).",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Build missing per-I dataset dirs with prepare_mitosis_crops.py.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Repeat the matrix per seed into <out>/seed{N}/.",
    )
    parser.add_argument(
        "--vary",
        choices=["both", "train", "data"],
        default="both",
        help="Which config fields --seeds sets: data.seed / data.budget_seed / both.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        metavar="PATTERN",
        default=None,
        help="Only run cells whose name contains any of these substrings.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the matrix + model param count; do not train."
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed cells.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; rebuild results + plots from cell metrics.json.",
    )
    args = parser.parse_args()

    root = find_project_root()

    def as_abs(p: Path) -> Path:
        return p if p.is_absolute() else root / p

    args.config, args.out, args.vae_dir = (
        as_abs(args.config),
        as_abs(args.out),
        as_abs(args.vae_dir),
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    base = MitosisPhaseConfig.from_yaml(args.config)
    base.data.synthetic_dir = as_abs(base.data.synthetic_dir)
    base.data.experiments_dir = as_abs(base.data.experiments_dir)
    i_filter = 500
    if args.aggregate_only:
        rows = _aggregate_and_write(args.out, i_filter)
        print(f"\n{len(rows)} runs -> {args.out / 'results.csv'}")
        return

    preset_i, preset_r, preset_s = STAGE_GRIDS[args.stage]
    grid_i = [parse_budget(t, minimum=1) for t in (args.grid_i or preset_i)]
    grid_r = [parse_budget(t, minimum=1) for t in (args.grid_r or preset_r)]
    grid_s = [parse_budget(t, minimum=0) for t in (args.grid_s or preset_s)]
    pools = {k: as_abs(v) for k, v in parse_synth_dirs(args.synth_dirs or [DEFAULT_SYNTH]).items()}

    problems = prepare_interphase_datasets(
        root, args.vae_dir, grid_i, base.data.classes, prepare=args.prepare
    )
    if problems:
        for p in problems:
            log.error(p)
        raise SystemExit(1)

    cells = build_matrix(
        base, root, grid_i, grid_r, grid_s, pools, args.out, seeds=args.seeds, vary=args.vary
    )
    if args.cells:
        cells = [c for c in cells if any(p in c.name for p in args.cells)]
        if not cells:
            raise SystemExit(f"No cells match --cells {args.cells}")

    if args.dry_run:
        print(f"Base config : {args.config}")
        print(f"Study root  : {args.out}")
        print(f"Seeds       : {sorted({c.seed for c in cells})}")
        print(f"I values    : {[label(v) for v in sorted(grid_i, key=budget_key)]}")
        print(f"R values    : {[label(v) for v in sorted(grid_r, key=budget_key)]}")
        print(f"S values    : {[label(v) for v in sorted(grid_s, key=budget_key)]}")
        print(f"G pools     : {{ {', '.join(f'{k}: {v}' for k, v in pools.items())} }}")
        from allium_cepa_classifier.training.model_builder import build_model

        model_cfg = base.model.model_copy(deep=True)
        model_cfg.pretrained = False
        model = build_model(model_cfg, num_classes=len(base.data.classes))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,}")
        print_matrix(cells)
        problems = _validate_data(cells)
        if problems:
            print("\nDATA PROBLEMS:")
            for p in problems:
                print(f"  - {p}")
            raise SystemExit(1)
        print("\nDry run OK (all data dirs present).")
        return

    problems = _validate_data(cells)
    if problems:
        for p in problems:
            log.error(p)
        raise SystemExit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = run_cells(cells, force=args.force)

    rows = _aggregate_and_write(args.out, i_filter)
    print(
        f"\nSweep summary: {len(done)} ran, {len(skipped)} skipped, {len(failed)} failed "
        f"of {len(cells)} cells -> {args.out / 'results.csv'}"
    )
    if failed:
        print("Failed cells: " + ", ".join(failed))
        raise SystemExit(1)
    if not rows:
        log.warning("No completed cells found to aggregate.")


if __name__ == "__main__":
    main()
