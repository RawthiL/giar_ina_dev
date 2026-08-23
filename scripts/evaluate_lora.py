"""
Extensible LoRA evaluator that writes metrics.json into the experiment directory.

Metrics (all computed per mitotic phase, then aggregated):

  loss              — final/min/avg loss from the kohya TensorBoard event files.
                      **Not comparable across configs that change the loss function.**
                      `min_snr_gamma` multiplies the loss by min(SNR,y)/SNR <= 1 and
                      `ip_noise_gamma` perturbs the target, so both deflate this number
                      mechanically. Kept for monitoring only — never rank on it.
  phase_consistency — PRIMARY. Does a phase-conditioned generation get classified as that
                      phase? Realism is worthless if the labels are wrong: a generated
                      "metaphase" that depicts prophase is label noise in the minority class
                      the project exists to fix.
  kid_classifier    — PRIMARY distributional metric. KID in phase-classifier feature space.
  kid_vqgan         — secondary. KID in cell-domain VQGAN encoder space: texture-oriented,
                      catches artifacts that phase-semantic features are invariant to.
  vqgan_recon       — secondary. Encode->decode error through the cell-domain autoencoder.
  memorization      — GUARD. Nearest-neighbour similarity to the LoRA's own training images.
                      A memorising LoRA scores excellent KID; this is what catches it.
  coverage          — GUARD. Coverage/density; catches mode collapse.

Replaces the previous `fid` (ImageNet Inception features, near-blind to grayscale microscopy,
and pointed at `datasets/crops/binary_classifier/` which is absent from the DVC remote) and
`classifier_judge` (whose loader hand-assembled a timm head and could not have loaded a
`BackboneWithHead` checkpoint).

Usage:
    uv run python scripts/evaluate_lora.py --config experiments/lora/p3_per_phase/config.yaml
    uv run python scripts/evaluate_lora.py --config ... --metrics loss phase_consistency
    uv run python scripts/evaluate_lora.py --config ... --samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.lora_metrics import (  # noqa: E402
    VQGANFeatures,
    coverage_density,
    effective_rank,
    kid,
    memorization_score,
    nn_self_similarity,
    real_to_real_baseline,
)
from utils.lora_samples import PHASES, ensure_phase_samples  # noqa: E402
from utils.phase_judge import find_images, load_phase_judge  # noqa: E402

from allium_cepa_classifier.config.base_config import find_project_root  # noqa: E402
from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig  # noqa: E402

_ROOT = find_project_root()
_JUDGE_DIR = _ROOT / "experiments/phase_classifier/efficientnet_b2"
_PHASE_CROPS = _ROOT / "datasets/crops/phase_classifier"
_VQGAN_DIR = _ROOT / "vqgan/vqgan weights/vqmodel"

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

MetricFn = Callable[["EvalContext"], dict[str, Any]]
METRICS: dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        METRICS[name] = fn
        return fn

    return decorator


class EvalContext:
    """Lazily-built shared state, so each metric pays only for what it actually uses."""

    def __init__(self, cfg: LoRAExperimentConfig, run_dir: Path, samples: dict[str, list[Path]]):
        self.cfg = cfg
        self.run_dir = run_dir
        self.samples = samples
        self._judge = None
        self._vqgan = None

    @property
    def judge(self):
        if self._judge is None:
            self._judge = load_phase_judge(_JUDGE_DIR)
        return self._judge

    @property
    def vqgan(self) -> VQGANFeatures:
        if self._vqgan is None:
            self._vqgan = VQGANFeatures(_VQGAN_DIR)
        return self._vqgan

    def real_crops(self, phase: str) -> list[Path]:
        """
        All real crops of a phase, across splits.

        Deliberately not restricted to the held-out split: KID needs as large a reference set
        as possible, and copying is caught separately by `memorization` rather than by starving
        this reference.
        """
        return [
            p
            for split in ("train", "validation", "test")
            for p in find_images(_PHASE_CROPS / split / phase)
        ]

    def lora_train_images(self) -> list[Path]:
        """The exact images this LoRA was trained on — the memorisation reference set."""
        root = (
            self.cfg.data.dataset_dir / self.cfg.data.dataset_version / self.cfg.data.train_data_dir
        )
        return find_images(root)


def _aggregate(per_phase: dict[str, float], prefix: str) -> dict[str, Any]:
    values = [v for v in per_phase.values() if v is not None and not np.isnan(v)]
    return {
        f"{prefix}_per_phase": {
            k: (None if v is None or np.isnan(v) else round(v, 5)) for k, v in per_phase.items()
        },
        f"{prefix}": round(float(np.mean(values)), 5) if values else None,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@register("loss")
def loss_from_tb(ctx: EvalContext) -> dict[str, Any]:
    """Read final/min/avg loss from the kohya TensorBoard event files. Monitoring only."""
    log_dir = ctx.run_dir / "logs"
    event_dir = None
    for depth in range(4):
        pattern = "/".join(["*"] * depth) + ("/" if depth else "") + "events.out.*"
        hits = sorted(log_dir.glob(pattern))
        if hits:
            event_dir = hits[-1].parent
            break
    if event_dir is None:
        print(f"[evaluate_lora] WARNING: no TensorBoard events under {log_dir}.")
        return {"final_loss": None, "min_loss": None, "avg_loss": None}

    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    loss_tags = [t for t in ea.Tags().get("scalars", []) if "loss" in t.lower()]
    if not loss_tags:
        return {"final_loss": None, "min_loss": None, "avg_loss": None}

    tag = next((t for t in loss_tags if "current" in t), loss_tags[0])
    values = [e.value for e in ea.Scalars(tag)]
    if not values:
        return {"final_loss": None, "min_loss": None, "avg_loss": None}
    return {
        "final_loss": round(values[-1], 6),
        "min_loss": round(min(values), 6),
        "avg_loss": round(sum(values) / len(values), 6),
        "loss_tag": tag,
        "loss_comparable_across_configs": False,
    }


@register("phase_consistency")
def phase_consistency(ctx: EvalContext) -> dict[str, Any]:
    """Fraction of phase-conditioned generations the judge assigns to the prompted phase."""
    judge = ctx.judge
    idx = judge.class_to_idx
    matrix: dict[str, dict[str, float]] = {}
    diag: dict[str, float] = {}

    for phase in PHASES:
        paths = ctx.samples.get(phase, [])
        if not paths:
            diag[phase] = float("nan")
            continue
        preds = judge.probs(paths).argmax(axis=1)
        row = {p: float((preds == idx[p]).mean()) for p in judge.class_names}
        matrix[phase] = {k: round(v, 4) for k, v in row.items()}
        diag[phase] = row[phase]

    out = _aggregate(diag, "phase_consistency")
    out["phase_confusion"] = matrix
    return out


@register("kid_classifier")
def kid_classifier(ctx: EvalContext) -> dict[str, Any]:
    """KID between generated and real crops, per phase, in phase-classifier feature space."""
    judge = ctx.judge
    per_phase: dict[str, float] = {}
    stds: dict[str, float] = {}
    for phase in PHASES:
        fake_paths, real_paths = ctx.samples.get(phase, []), ctx.real_crops(phase)
        if not fake_paths or not real_paths:
            per_phase[phase] = float("nan")
            continue
        value, std = kid(judge.features(real_paths), judge.features(fake_paths))
        per_phase[phase], stds[phase] = value, std
    out = _aggregate(per_phase, "kid_classifier")
    out["kid_classifier_std_per_phase"] = {k: round(v, 5) for k, v in stds.items()}
    return out


@register("kid_vqgan")
def kid_vqgan(ctx: EvalContext) -> dict[str, Any]:
    """KID in cell-domain VQGAN encoder space — texture-oriented, secondary to kid_classifier."""
    if not _VQGAN_DIR.exists():
        print(f"[evaluate_lora] WARNING: no VQGAN weights at {_VQGAN_DIR} — skipping kid_vqgan.")
        return {"kid_vqgan": None, "kid_vqgan_per_phase": {}}
    vq = ctx.vqgan
    per_phase: dict[str, float] = {}
    for phase in PHASES:
        fake_paths, real_paths = ctx.samples.get(phase, []), ctx.real_crops(phase)
        if not fake_paths or not real_paths:
            per_phase[phase] = float("nan")
            continue
        value, _ = kid(vq.features(real_paths), vq.features(fake_paths))
        per_phase[phase] = value
    return _aggregate(per_phase, "kid_vqgan")


@register("vqgan_recon")
def vqgan_recon(ctx: EvalContext) -> dict[str, Any]:
    """
    Encode->decode error through the cell-domain autoencoder, relative to real crops.

    **Read the ratio as a texture-complexity comparison, not purely as an artifact count**
    (clarified 2026-08-15, when the [-1,1]/[0,1] input-range bug in `VQGANFeatures._batch` was
    fixed). On the corrected scale, p3_per_phase_nofill scores ~0.74: generated crops reconstruct
    *more easily* than real ones, because real microscopy carries grain, dust specks and focus
    noise the generator smooths away. Under the old wrong scale the same comparison read 1.22 and
    looked like the opposite conclusion.

    So both directions are informative and neither is simply "good":
      ratio > 1 — generations contain structure the cell-domain autoencoder cannot represent,
                  i.e. genuine artifacts.
      ratio < 1 — generations are smoother/simpler than real crops, i.e. missing fine texture.
    A value near 1 is the target.
    """
    if not _VQGAN_DIR.exists():
        return {"vqgan_recon_mse": None, "vqgan_recon_ratio": None}
    vq = ctx.vqgan
    fake_paths = [p for phase in PHASES for p in ctx.samples.get(phase, [])]
    real_paths = [p for phase in PHASES for p in ctx.real_crops(phase)[:100]]
    if not fake_paths:
        return {"vqgan_recon_mse": None, "vqgan_recon_ratio": None}

    fake = vq.reconstruction_error(fake_paths)
    real = vq.reconstruction_error(real_paths) if real_paths else {"recon_mse_mean": float("nan")}
    ratio = fake["recon_mse_mean"] / real["recon_mse_mean"] if real["recon_mse_mean"] else None
    return {
        "vqgan_recon_mse": round(fake["recon_mse_mean"], 6),
        "vqgan_recon_mse_p95": round(fake["recon_mse_p95"], 6),
        "vqgan_recon_mse_real": round(real["recon_mse_mean"], 6),
        # >1: generations hold structure the autoencoder cannot represent (artifacts).
        # <1: generations are smoother than real crops (missing fine texture). ~1 is the target.
        "vqgan_recon_ratio": round(ratio, 4) if ratio else None,
    }


@register("self_similarity")
def self_similarity(ctx: EvalContext) -> dict[str, Any]:
    """
    Intra-set diversity: how much the generations differ from *each other*.

    The gap this fills: every other metric compares generations to real crops or to training
    crops. None compares them to one another, so a generator emitting a few templates plus noise
    passes all of them — if the templates land in dense regions of the real distribution,
    coverage counts many real neighbourhoods as reached and KID only checks aggregate statistics.
    Found on 2026-08-16 by eye, on trial_016, which posted the best kid_classifier (0.607) and the
    highest coverage (0.550) in the study while producing ~4 recurring templates per phase.

    Two statistics, both as a ratio to the same statistic on real crops, catching two failures:

      nn_self_similarity  leave-one-out nearest-neighbour similarity.
                          ratio > 1 -> TEMPLATING: generations cluster tighter than real cells do.
      effective_rank      participation ratio of the covariance spectrum.
                          ratio << 1 -> MODE COLLAPSE: the set spans few independent modes.

    Measured trial_016 / p5_noaug / a poor run: nn_ratio 1.053 / 0.935 / 0.785 and
    er_ratio 1.090 / 0.825 / 0.267. trial_016 is templated (tight clusters, many of them); the
    poor run is collapsed (few modes). The two signatures are independent.

    **Judge feature space, not VQGAN.** The first attempt used VQGAN encoder features, reasoning
    that the judge is invariant to the staining and illumination variation the eye notices. The
    measurement refuted it: GAP-pooled VQGAN features have an effective rank of ~1.2 — a single
    dimension — and returned 0.998/0.997/0.992, no discrimination at all.

    REPORTED ONLY — deliberately not in `optimize_lora.score()`. Added mid-study, and folding a
    new term into the objective would make trials before and after incomparable, which is the
    mistake that voided the first study. It runs on cached samples, so it applies retroactively.
    """
    judge = ctx.judge
    nn_ratio: dict[str, float] = {}
    er_ratio: dict[str, float] = {}

    for phase in PHASES:
        fake_paths = ctx.samples.get(phase, [])
        real_paths = ctx.real_crops(phase)
        if not fake_paths or len(real_paths) < 2:
            nn_ratio[phase] = er_ratio[phase] = float("nan")
            continue
        fake_feat = judge.features(fake_paths)
        real_feat = judge.features(real_paths)

        nn_real = nn_self_similarity(real_feat)
        er_real = effective_rank(real_feat)
        nn_ratio[phase] = nn_self_similarity(fake_feat) / nn_real if nn_real else float("nan")
        er_ratio[phase] = effective_rank(fake_feat) / er_real if er_real else float("nan")

    return {
        **_aggregate(nn_ratio, "nn_self_similarity_ratio"),
        **_aggregate(er_ratio, "effective_rank_ratio"),
    }


@register("memorization")
def memorization(ctx: EvalContext) -> dict[str, Any]:
    """Nearest-neighbour similarity from generated images to the LoRA's own training set."""
    train_paths = ctx.lora_train_images()
    if not train_paths:
        print("[evaluate_lora] WARNING: LoRA training images not found — skipping memorization.")
        return {"memorization_nn_p95": None}
    judge = ctx.judge
    feat_train = judge.features(train_paths)
    fake_paths = [p for phase in PHASES for p in ctx.samples.get(phase, [])]
    if not fake_paths:
        return {"memorization_nn_p95": None}
    scores = memorization_score(feat_train, judge.features(fake_paths))
    control = real_to_real_baseline(feat_train)
    # The number that matters: how much closer to the training set a generated image sits than
    # two distinct real crops sit to each other. Near 0 is healthy; clearly positive is copying.
    excess = scores["nn_p95"] - control["nn_p95"]
    return {
        "memorization_nn_p95": round(scores["nn_p95"], 4),
        "memorization_nn_max": round(scores["nn_max"], 4),
        "memorization_nn_mean": round(scores["nn_mean"], 4),
        "memorization_real_baseline_p95": round(control["nn_p95"], 4),
        "memorization_excess_p95": round(excess, 4),
        "memorization_n_train": len(train_paths),
    }


@register("coverage")
def coverage(ctx: EvalContext) -> dict[str, Any]:
    """Coverage and density per phase — the mode-collapse guard."""
    judge = ctx.judge
    cov: dict[str, float] = {}
    den: dict[str, float] = {}
    for phase in PHASES:
        fake_paths, real_paths = ctx.samples.get(phase, []), ctx.real_crops(phase)
        if not fake_paths or not real_paths:
            cov[phase] = den[phase] = float("nan")
            continue
        c, d = coverage_density(judge.features(real_paths), judge.features(fake_paths))
        cov[phase], den[phase] = c, d
    return {**_aggregate(cov, "coverage"), **_aggregate(den, "density")}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Images per phase. Generated only if not already cached (default 100).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into the existing metrics.json instead of replacing it. Needed when "
        "computing a subset via --metrics, which would otherwise drop every metric not "
        "recomputed. Off by default because silently keeping stale values is how a metric on an "
        "old scale survives a fix (see the VQGAN [0,1] change).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent

    samples = ensure_phase_samples(cfg, run_dir, args.samples, seed=args.seed)
    n_by_phase = {p: len(v) for p, v in samples.items()}
    print(f"[evaluate_lora] samples per phase: {n_by_phase}")

    ctx = EvalContext(cfg, run_dir, samples)
    out: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "dataset_version": cfg.data.dataset_version,
        "n_samples_per_phase": n_by_phase,
    }
    for name in args.metrics:
        print(f"[evaluate_lora] {name} ...")
        out.update(METRICS[name](ctx))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = run_dir / "metrics.json"
    if args.merge and metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        existing.update(out)
        out = existing
    metrics_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\n[evaluate_lora] wrote {metrics_path}")
    for k, v in sorted(out.items()):
        if not isinstance(v, dict):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
