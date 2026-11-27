"""Tests for the unified data-budget grid driver (scripts/sweep_mitosis_data_budget.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["anaphase", "interphase", "metaphase", "prophase", "telophase"]
MIT = [0, 2, 3, 4]
NO_PRO = [0, 2, 4]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return _load_script(
        "sweep_mitosis_data_budget", _ROOT / "scripts" / "sweep_mitosis_data_budget.py"
    )


def _random_predictions(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    y_true = rng.choice(5, size=n, p=[0.03, 0.85, 0.04, 0.04, 0.04])
    y_pred = np.where(rng.random(n) < 0.6, y_true, rng.integers(0, 5, size=n))
    return y_true, y_pred


def _write_cell(cell_dir: Path, seed: int, macro_f1: float = 0.3) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "metrics.json").write_text(
        json.dumps(
            {
                "include_synthetic": False,
                "classes": CLASSES,
                "test_macro_f1": macro_f1,
                "test_balanced_acc": macro_f1,
            }
        )
    )
    (cell_dir / "used_config.yaml").write_text(f"data:\n  seed: {seed}\n  budget_seed: {seed}\n")


# ---------------------------------------------------------------- metric helpers


def test_metric_values_match_sklearn(drv):
    y_true, y_pred = _random_predictions()
    mv = drv._metric_values(y_true, y_pred, CLASSES)

    assert mv["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=range(5), average="macro", zero_division=0)
    )
    assert mv["balanced_acc"] == pytest.approx(
        recall_score(y_true, y_pred, labels=range(5), average="macro", zero_division=0)
    )
    mb = np.isin(y_true, MIT)
    assert mv["mit_only_f1"] == pytest.approx(
        f1_score(y_true[mb], y_pred[mb], labels=MIT, average="macro", zero_division=0)
    )
    assert mv["no_pro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=NO_PRO, average="macro", zero_division=0)
    )
    bt = np.isin(y_true, MIT).astype(int)
    bp = np.isin(y_pred, MIT).astype(int)
    assert mv["bin_f1"] == pytest.approx(f1_score(bt, bp, zero_division=0))
    assert mv["mitotic_recall"] == pytest.approx(((bt == 1) & (bp == 1)).sum() / (bt == 1).sum())
    ii = CLASSES.index("interphase")
    fp = ((y_true == ii) & (y_pred != ii)).sum()
    tn = ((y_true == ii) & (y_pred == ii)).sum()
    assert mv["interphase_fpr"] == pytest.approx(fp / (fp + tn))


def test_bootstrap_ci_deterministic_and_brackets(drv):
    y_true, y_pred = _random_predictions(seed=3)
    a = drv.bootstrap_ci(y_true, y_pred, CLASSES, n_boot=200, seed=7)
    b = drv.bootstrap_ci(y_true, y_pred, CLASSES, n_boot=200, seed=7)
    mv = drv._metric_values(y_true, y_pred, CLASSES)
    assert a == b
    for key in drv.CI_METRICS:
        lo, hi = a[key]
        assert lo <= hi
        assert lo <= mv[key] <= hi


def test_bootstrap_ci_handles_missing_class(drv):
    rng = np.random.default_rng(1)
    y_true = rng.choice([0, 1, 2], size=120)
    y_pred = rng.choice([0, 1, 2], size=120)
    ci = drv.bootstrap_ci(y_true, y_pred, CLASSES, n_boot=50, seed=1)
    assert set(ci) == set(drv.CI_METRICS)
    assert all(np.isfinite([lo, hi]).all() for lo, hi in ci.values())


# ---------------------------------------------------------------- matrix / datasets


def _base():
    from allium_cepa_classifier.config.mitosis_phase_config import MitosisPhaseConfig

    return MitosisPhaseConfig(experiment_name="t")


def test_interphase_dataset_dir(drv, tmp_path):
    assert drv.interphase_dataset_dir(tmp_path, 500) == tmp_path / "datasets/crops/mitosis_phase5"
    assert (
        drv.interphase_dataset_dir(tmp_path, 4000)
        == tmp_path / "datasets/crops/mitosis_phase5_ic4000"
    )
    assert (
        drv.interphase_dataset_dir(tmp_path, None)
        == tmp_path / "datasets/crops/mitosis_phase5_icfull"
    )


def test_prepare_reports_missing_without_flag(drv, tmp_path):
    caps = [500, 4000, None]
    problems = drv.prepare_interphase_datasets(
        tmp_path, tmp_path / "vae", caps, CLASSES, prepare=False
    )
    # 500 maps to the standard dir too, so all three are missing here
    assert len(problems) == 3
    assert all("missing dataset dir" in p for p in problems)


def test_build_matrix_axes_and_seeds(drv, tmp_path):
    base = _base()
    cells = drv.build_matrix(
        base,
        tmp_path,
        grid_i=[500, 4000, None],
        grid_r=[25, None],
        grid_s=[0, 250],
        pools={"full": tmp_path / "synth"},
        out_root=tmp_path / "out",
        seeds=[43, 44],
        vary="both",
    )
    assert len(cells) == 3 * 2 * 2 * 2
    names = {c.name for c in cells}
    assert "i500_r25_s250_gfull" in names
    assert "ifull_rfull_s0_gfull" in names
    c = next(c for c in cells if c.name == "i4000_rfull_s250_gfull" and c.seed == 44)
    assert c.run_dir == tmp_path / "out" / "seed44" / "i4000_rfull_s250_gfull"
    assert c.cfg.data.crops_dir == tmp_path / "datasets/crops/mitosis_phase5_ic4000"
    assert c.cfg.data.seed == 44 and c.cfg.data.budget_seed == 44
    assert c.cfg.data.include_synthetic and c.cfg.data.synthetic_max_per_class == 250
    c0 = next(c for c in cells if c.name == "i500_r25_s0_gfull" and c.seed == 43)
    assert not c0.cfg.data.include_synthetic


def test_build_matrix_implicit_seed_keeps_base_budget_seed(drv, tmp_path):
    base = _base()
    base.data.seed = 42
    base.data.budget_seed = 1234
    cells = drv.build_matrix(
        base, tmp_path, [500], [None], [0], {"full": tmp_path / "s"}, tmp_path / "o", seeds=None
    )
    assert len(cells) == 1
    assert cells[0].seed == 42
    assert cells[0].run_dir == tmp_path / "o" / "seed42" / "i500_rfull_s0_gfull"
    assert cells[0].cfg.data.seed == 42
    assert cells[0].cfg.data.budget_seed == 1234


# ---------------------------------------------------------------- aggregation


def test_parse_cell_dir_roundtrip(drv, tmp_path):
    assert drv._parse_cell_dir(tmp_path / "i500_r25_sfull_gfull") == (500, 25, None, "full")
    assert drv._parse_cell_dir(tmp_path / "ifull_rfull_s0_ghalf") == (None, None, 0, "half")
    assert drv._parse_cell_dir(tmp_path / "plots") is None


def test_aggregate_without_predictions_is_safe(drv, tmp_path):
    out = tmp_path / "study"
    _write_cell(out / "seed42" / "i500_r25_s0_gfull", seed=42)
    rows = drv.aggregate(out)
    assert len(rows) == 1
    assert rows[0]["i"] == 500 and rows[0]["seed"] == 42
    assert rows[0]["test_mit_only_f1"] is None
    assert rows[0]["macro_f1_ci_lo"] is None
    drv.write_results(rows, out)
    header = (out / "results.csv").read_text().splitlines()[0]
    assert "i_label" in header and "test_mit_only_f1" in header
    assert len(json.loads((out / "results.json").read_text())) == 1


def test_aggregate_computes_metrics_from_npz(drv, tmp_path):
    out = tmp_path / "study"
    cell = out / "seed43" / "ifull_rfull_s250_gfull"
    cell.mkdir(parents=True)
    y_true, y_pred = _random_predictions(seed=11)
    np.savez_compressed(cell / "predictions.npz", y_true=y_true, y_pred=y_pred)
    mv = drv._metric_values(y_true, y_pred, CLASSES)
    (cell / "metrics.json").write_text(
        json.dumps(
            {
                "include_synthetic": True,
                "classes": CLASSES,
                "n_train_real": 1722,
                "n_train_synth": 1250,
                "test_macro_f1": mv["macro_f1"],
                "test_balanced_acc": mv["balanced_acc"],
            }
        )
    )
    (cell / "used_config.yaml").write_text("data:\n  seed: 43\n  budget_seed: 43\n")
    row = drv.aggregate(out)[0]
    assert row["i"] is None and row["seed"] == 43
    assert row["test_mit_only_f1"] == pytest.approx(mv["mit_only_f1"])
    assert row["test_bin_f1"] == pytest.approx(mv["bin_f1"])
    assert row["macro_f1_ci_lo"] <= row["test_macro_f1"] <= row["macro_f1_ci_hi"]


def test_seed_stats_mean_sd_and_paired_delta_with_i(drv):
    def row(i, s, seed, macro):
        return {
            "cell": f"i{i}_r25_s{s}_gfull",
            "g": "full",
            "i": i,
            "r": 25,
            "s": s,
            "seed": seed,
            "test_macro_f1": macro,
        }

    rows = [
        row(500, 0, 42, 0.40),
        row(500, 0, 43, 0.44),
        row(500, 250, 42, 0.43),
        row(500, 250, 43, 0.50),
        row(4000, 0, 42, 0.70),
    ]
    stats = {(st["i"], st["s"], st["metric"]): st for st in drv.seed_stats(rows)}
    cell = stats[(500, 250, "test_macro_f1")]
    assert cell["n"] == 2
    assert cell["mean"] == pytest.approx(0.465)
    assert cell["sd"] == pytest.approx(np.std([0.43, 0.50], ddof=1))
    assert cell["n_pairs"] == 2
    assert cell["delta_mean"] == pytest.approx(0.045)
    assert cell["delta_sd"] == pytest.approx(np.std([0.03, 0.06], ddof=1))
    base = stats[(4000, 0, "test_macro_f1")]
    assert base["n"] == 1 and base["sd"] is None
    assert base["n_pairs"] == 0 and base["delta_mean"] is None


def test_per_class_counts_from_log_fallback(drv, tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "  train[interphase] = 25\n  train[prophase] = 25\n"
        "  train[interphase] = 500\n  train[prophase] = 355\n"
    )
    assert drv._per_class_counts_from_log(log) == {"interphase": 500, "prophase": 355}
    assert drv._per_class_counts_from_log(tmp_path / "missing.log") is None
