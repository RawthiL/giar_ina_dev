"""
Unit tests for the LoRA generation metrics.

These exist because of a concrete failure. The first `kid()` implementation used the canonical
polynomial kernel `(x·y/d + 1)^3` on *raw* classifier features. Canonical KID assumes
Inception pool features, whose scale is well behaved; ours are not. On 2026-08-10 one degenerate
generated crop had a feature norm of 2566 against a median of 13.7, the cubic amplified it to
~1e11 in the k(y,y) block, and since the unbiased estimator strips the k(y,y) diagonal but has
no diagonal to strip from k(x,y), the outlier landed asymmetrically and produced MMD^2 = -35.
That trial then led the Optuna study with a score of 9.04 despite a phase_consistency of 0.41.

Every test below is a property that failure violated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


metrics = _load_script("lora_metrics", _ROOT / "scripts" / "utils" / "lora_metrics.py")


def _blob(rng: np.random.Generator, n: int, dim: int = 64, shift: float = 0.0) -> np.ndarray:
    """Non-negative features, like the post-activation pooled features the judge produces."""
    return np.abs(rng.normal(loc=1.0 + shift, scale=0.3, size=(n, dim)))


# ---------------------------------------------------------------------------
# kid
# ---------------------------------------------------------------------------


def test_kid_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    value, _ = metrics.kid(_blob(rng, 120), _blob(rng, 120))
    assert abs(value) < 0.05


def test_kid_detects_a_shifted_distribution():
    rng = np.random.default_rng(0)
    same, _ = metrics.kid(_blob(rng, 120), _blob(rng, 120))
    shifted, _ = metrics.kid(_blob(rng, 120), _blob(rng, 120, shift=1.5))
    assert shifted > same


def test_kid_is_non_negative_for_disjoint_distributions():
    """MMD^2 between genuinely different distributions must not come out negative."""
    rng = np.random.default_rng(1)
    value, _ = metrics.kid(_blob(rng, 100), _blob(rng, 100, shift=3.0))
    assert value > 0


def test_kid_survives_a_norm_outlier():
    """
    The regression itself: one sample with a 200x feature norm must not swing KID negative
    or dominate the estimate.
    """
    rng = np.random.default_rng(2)
    real = _blob(rng, 100)
    fake = _blob(rng, 100)
    clean, _ = metrics.kid(real, fake)

    fake_outlier = fake.copy()
    fake_outlier[0] *= 200.0
    dirty, _ = metrics.kid(real, fake_outlier)

    assert dirty > -0.05, f"outlier drove KID negative: {dirty}"
    assert abs(dirty - clean) < 1.0, f"one sample moved KID by {abs(dirty - clean)}"


def test_kid_too_few_samples_is_nan():
    rng = np.random.default_rng(3)
    value, std = metrics.kid(_blob(rng, 2), _blob(rng, 2))
    assert np.isnan(value) and np.isnan(std)


# ---------------------------------------------------------------------------
# coverage / density
# ---------------------------------------------------------------------------


def test_coverage_high_when_distributions_match():
    rng = np.random.default_rng(4)
    coverage, _ = metrics.coverage_density(_blob(rng, 150), _blob(rng, 150))
    assert coverage > 0.5


def test_coverage_low_under_mode_collapse():
    """All generated samples piled on one point should cover almost nothing."""
    rng = np.random.default_rng(5)
    real = _blob(rng, 150)
    collapsed = np.repeat(real[:1], 150, axis=0) + rng.normal(scale=1e-3, size=(150, real.shape[1]))
    coverage, _ = metrics.coverage_density(real, collapsed)
    assert coverage < 0.2


def test_coverage_is_invariant_to_reference_set_size():
    """
    Coverage must not depend on how many real crops a phase happens to have — otherwise it is
    incomparable across phases. Guards the sample-size equalisation.
    """
    rng = np.random.default_rng(6)
    fake = _blob(rng, 80)
    small, _ = metrics.coverage_density(_blob(rng, 80), fake)
    large, _ = metrics.coverage_density(_blob(rng, 800), fake)
    assert abs(small - large) < 0.25


# ---------------------------------------------------------------------------
# memorisation
# ---------------------------------------------------------------------------


def test_memorization_flags_exact_copies():
    rng = np.random.default_rng(7)
    train = _blob(rng, 100)
    copies = train[:20].copy()
    assert metrics.memorization_score(train, copies)["nn_p95"] > 0.999


def test_memorization_baseline_excludes_self_match():
    """Leave-one-out: an image must never be its own nearest neighbour, or every set scores 1.0."""
    rng = np.random.default_rng(8)
    assert metrics.real_to_real_baseline(_blob(rng, 100))["nn_max"] < 0.9999


@pytest.mark.parametrize("fn", ["memorization_score", "real_to_real_baseline"])
def test_memorization_handles_empty_input(fn):
    empty = np.empty((0, 64))
    result = (
        metrics.memorization_score(empty, empty)
        if fn == "memorization_score"
        else metrics.real_to_real_baseline(empty)
    )
    assert np.isnan(result["nn_p95"])


# ---------------------------------------------------------------------------
# VQGANFeatures input range — guards the [0,1] vs [-1,1] regression
# ---------------------------------------------------------------------------


def test_vqgan_batch_normalises_to_unit_interval(tmp_path):
    """
    `_batch` must emit [0,1], not the SD-standard [-1,1].

    The checkpoint under `vqgan/vqgan weights/vqmodel` was trained on [0,1]. Feeding it [-1,1]
    inflated reconstruction MSE ~200x (0.0022 vs 0.423 over 48 crops at 256px) and made
    `vqgan_recon_mse` meaningless as an artifact detector: the value was dominated by the
    normalisation offset. The bug was invisible in rankings because it applied to real and
    generated crops alike.
    """
    from PIL import Image

    img_path = tmp_path / "crop.png"
    Image.new("RGB", (64, 64), (0, 128, 255)).save(img_path)

    stub = metrics.VQGANFeatures.__new__(metrics.VQGANFeatures)
    stub.device = "cpu"
    stub.resolution = 64

    batch = metrics.VQGANFeatures._batch(stub, [img_path])

    assert batch.shape == (1, 3, 64, 64)
    assert batch.min() >= 0.0, "input dips below 0 — still on the [-1,1] scale"
    assert batch.max() <= 1.0
    # Channel means recover the source colour on a 0-1 scale.
    assert batch[0, 0].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert batch[0, 1].mean().item() == pytest.approx(128 / 255, abs=1e-3)
    assert batch[0, 2].mean().item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# mean_pairwise_similarity — the intra-set diversity gap found on 2026-08-16
# ---------------------------------------------------------------------------


def test_mean_pairwise_similarity_detects_templating_synthetically():
    """
    A set built from a few templates must score far more self-similar than a diverse one.

    This is the gap the other metrics could not see: every one of them compares generations to
    real or training crops, none to each other. trial_016 posted the best kid_classifier (0.607)
    and the highest coverage (0.550) ever measured while visual inspection found only 3-4
    recurring templates per phase.
    """
    rng = np.random.default_rng(0)
    diverse = _blob(rng, 200, dim=64)

    templates = _blob(rng, 4, dim=64)
    templated = np.repeat(templates, 50, axis=0) + rng.normal(scale=0.02, size=(200, 64))

    assert metrics.mean_pairwise_similarity(templated) > metrics.mean_pairwise_similarity(diverse)


def test_self_similarity_of_identical_set_is_one():
    rng = np.random.default_rng(1)
    identical = np.repeat(_blob(rng, 1, dim=64), 50, axis=0)
    assert metrics.mean_pairwise_similarity(identical) == pytest.approx(1.0, abs=1e-5)


def test_self_similarity_excludes_self_match():
    """
    Must average the off-diagonal only. Including the diagonal pulls every set toward 1.0 by
    n/n^2, which would make a diverse set look templated at small n.
    """
    rng = np.random.default_rng(2)
    x = _blob(rng, 8, dim=64)

    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = unit @ unit.T
    n = len(x)
    expected_off_diagonal = (sim.sum() - np.trace(sim)) / (n * (n - 1))

    assert metrics.mean_pairwise_similarity(x) == pytest.approx(expected_off_diagonal, abs=1e-6)
    assert metrics.mean_pairwise_similarity(x) < sim.mean()


def test_self_similarity_too_few_samples_is_nan():
    assert np.isnan(metrics.mean_pairwise_similarity(np.empty((1, 64))))


def test_nn_self_similarity_detects_templating():
    """
    The statistic that actually works on real data. Templating means every sample has a
    near-twin, which a *max* exposes and a mean dilutes — `mean_pairwise_similarity` returned
    0.912/0.910/0.918 across a templated run, a healthy one and a poor one, i.e. nothing.
    """
    rng = np.random.default_rng(0)
    diverse = _blob(rng, 200, dim=64)
    templates = _blob(rng, 4, dim=64)
    templated = np.repeat(templates, 50, axis=0) + rng.normal(scale=0.02, size=(200, 64))

    assert metrics.nn_self_similarity(templated) > metrics.nn_self_similarity(diverse)


def test_effective_rank_detects_mode_collapse():
    """A set spanning few modes must score a far lower effective rank than a diverse one."""
    rng = np.random.default_rng(1)
    diverse = _blob(rng, 200, dim=64)
    collapsed = np.repeat(_blob(rng, 2, dim=64), 100, axis=0) + rng.normal(
        scale=0.01, size=(200, 64)
    )

    assert metrics.effective_rank(collapsed) < metrics.effective_rank(diverse) / 2


def test_effective_rank_of_identical_set_is_near_one():
    rng = np.random.default_rng(2)
    identical = np.repeat(_blob(rng, 1, dim=64), 50, axis=0)
    assert metrics.effective_rank(identical) < 1.5


def test_nn_and_effective_rank_are_independent_signals():
    """
    Templating and mode collapse are different failures and must be separable: a set of many
    tight clusters has high nn-similarity but healthy effective rank, which is exactly what
    trial_016 showed (nn_ratio 1.053, er_ratio 1.090).
    """
    rng = np.random.default_rng(3)
    many_tight_clusters = np.repeat(_blob(rng, 40, dim=64), 5, axis=0) + rng.normal(
        scale=0.01, size=(200, 64)
    )
    diverse = _blob(rng, 200, dim=64)

    assert metrics.nn_self_similarity(many_tight_clusters) > metrics.nn_self_similarity(diverse)
    assert metrics.effective_rank(many_tight_clusters) > metrics.effective_rank(diverse) / 2
