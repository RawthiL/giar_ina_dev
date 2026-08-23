"""
Tests for the Optuna search driver.

The critical invariant is that `DISTRIBUTIONS` stays a faithful mirror of the space
`build_config` actually samples. They are declared separately — `build_config` for readability,
`DISTRIBUTIONS` because replaying a finished trial into a new study needs explicit distribution
objects — so nothing but a test stops them drifting. If they drift, `--reseed-from` silently
replays trials under the wrong space and poisons the sampler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import optuna
import pytest

_ROOT = Path(__file__).resolve().parent.parent
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


opt = _load_script("optimize_lora", _ROOT / "scripts" / "optimize_lora.py")


_DATASET_VERSION = "per_phase_nofill"


def _sample(n: int = 100):
    study = optuna.create_study(direction="maximize")
    for _ in range(n):
        trial = study.ask()
        cfg = opt.build_config(trial, "t", _DATASET_VERSION)
        study.tell(trial, 0.0)
        yield trial, cfg


def test_params_round_trip_through_config():
    """params -> config.yaml -> params must be lossless, or reseeding rewrites history."""
    for trial, cfg in _sample():
        recovered = opt.params_from_config(cfg)
        assert set(recovered) == set(trial.params)
        for key, value in trial.params.items():
            if isinstance(value, float):
                assert recovered[key] == pytest.approx(value, rel=1e-6)
            else:
                assert recovered[key] == value


def test_every_sampled_param_is_declared_and_in_range():
    for trial, _ in _sample():
        for key, value in trial.params.items():
            assert key in opt.DISTRIBUTIONS, f"{key} missing from DISTRIBUTIONS"
            dist = opt.DISTRIBUTIONS[key]
            assert dist._contains(dist.to_internal_repr(value)), f"{key}={value} out of range"


def test_dataset_version_is_threaded_into_every_config():
    """
    Guards the bug that invalidated the first study: `dataset_version` was hardcoded to
    `per_phase`, so all 21 trials silently trained on the dataset with the rotation-fill
    artifact even after `per_phase_nofill` existed.
    """
    for _, cfg in _sample(20):
        assert cfg["data"]["dataset_version"] == _DATASET_VERSION


def test_disabled_tricks_are_absent_from_the_config():
    """A trick that is off must be omitted, not written as 0.0 — kohya treats those differently."""
    seen_off = False
    for trial, cfg in _sample():
        if not trial.params["use_noise_offset"]:
            seen_off = True
            assert "noise_offset" not in cfg["training"]
    assert seen_off, "sampling never produced use_noise_offset=False"


def test_score_penalises_memorisation_below_a_weaker_but_honest_run():
    """A memoriser with better raw metrics must not outrank an honest run."""
    honest, _ = opt.score(
        {
            "phase_consistency": 0.75,
            "kid_classifier": 1.6,
            "coverage": 0.33,
            "memorization_excess_p95": -0.07,
        }
    )
    memoriser, _ = opt.score(
        {
            "phase_consistency": 0.90,
            "kid_classifier": 0.5,
            "coverage": 0.30,
            "memorization_excess_p95": 0.10,
        }
    )
    assert memoriser < honest


def test_score_is_minus_inf_when_metrics_are_missing():
    value, _ = opt.score({"phase_consistency": None, "kid_classifier": None, "coverage": None})
    assert value == float("-inf")
