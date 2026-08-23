"""
Phase 4: does the synthetic data actually improve a mitotic-phase classifier?

This is the only question that justifies the LoRA workstream. Everything upstream —
phase_consistency, KID, coverage — is a proxy chosen because it is cheap enough to put inside an
optimisation loop. This measures the real thing: train the phase classifier on real + synthetic
crops at several mixing ratios and score it on **real held-out crops only**.

Two guards against fooling ourselves, both mandatory:

1. **A different architecture from the judge.** `evaluate_lora.py` scores generations with
   efficientnet_b2. If the downstream evaluator were also efficientnet_b2, we would be optimising
   generations to please a model and then reporting that the same model improved. Default here is
   resnet50, and no weights are shared.
2. **Real test crops only, never synthetic.** The test split is the 118 group-disjoint,
   deduplicated real crops from `datasets/crops/phase_classifier/test`.

**Statistical caveat, unavoidable at this dataset size.** That test split holds 118 crops
(prophase 44, metaphase 27, telophase 25, anaphase 22). A per-phase recall of ~0.9 on n=22 has a
95% Wilson interval roughly ±0.15. Only large effects are detectable. Hence `--seeds`: each
configuration is trained several times and reported as mean +/- std across seeds, so run-to-run
variance is visible rather than mistaken for signal.

Usage:
    uv run python scripts/validate_synthetic_downstream.py \
        --configs experiments/lora/_studies/lora_phase_v2/trial_007/config.yaml
    uv run python scripts/validate_synthetic_downstream.py --configs ... --ratios 0 0.5 1.0 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.lora_samples import PHASES, ensure_phase_samples  # noqa: E402
from utils.phase_judge import find_images  # noqa: E402

from allium_cepa_classifier.config.base_config import find_project_root  # noqa: E402
from allium_cepa_classifier.config.experiment_config import ExperimentConfig  # noqa: E402
from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig  # noqa: E402

_ROOT = find_project_root()
_PHASE_CROPS = _ROOT / "datasets/crops/phase_classifier"
_OUT_ROOT = _ROOT / "experiments/phase_classifier/_downstream"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — correct at small n, unlike the normal approximation."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def filter_by_judge(
    synthetic: dict[str, list[Path]], min_confidence: float
) -> dict[str, list[Path]]:
    """
    Drop generations the judge does not assign to their intended phase confidently.

    `p3_per_phase` sits at 0.752 phase consistency, so roughly a quarter of its output carries a
    wrong label — and anaphase, the scarcest class, is the worst at 0.52. Feeding that in
    unfiltered injects label noise exactly where there is least real data to absorb it.

    The judge stays *outside* the downstream evaluator's architecture (see the --arch guard), so
    filtering here does not close the circularity loop: the evaluator never shares weights with
    the model doing the filtering. It does, however, make the surviving set a judge-shaped subset,
    which is why this is opt-in and why the unfiltered arm must also be reported.
    """
    from utils.phase_judge import load_phase_judge  # local: keeps the torch import off dry paths

    judge = load_phase_judge(_ROOT / "experiments/phase_classifier/efficientnet_b2")
    class_to_idx = judge.class_to_idx

    kept: dict[str, list[Path]] = {}
    print(f"\nJudge filter (min confidence {min_confidence:g}):")
    for phase, paths in synthetic.items():
        if not paths:
            kept[phase] = []
            continue
        probs = judge.probs(paths)
        target = class_to_idx[phase]
        keep_mask = (probs.argmax(axis=1) == target) & (probs[:, target] >= min_confidence)
        kept[phase] = [p for p, k in zip(paths, keep_mask, strict=True) if k]
        print(
            f"  {phase:<11} {len(kept[phase]):>4} / {len(paths):<4} kept ({keep_mask.mean():.1%})"
        )
    return kept


def build_mixed_dataset(
    dest: Path, synthetic: dict[str, list[Path]], ratio: float
) -> dict[str, int]:
    """
    Assemble train = real train + `ratio` x synthetic, with validation/test left purely real.

    Symlinks rather than copies: at ratio 2.0 this is ~2900 images per configuration and we build
    one per (config, ratio, seed).
    """
    if dest.exists():
        shutil.rmtree(dest)

    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        for phase in PHASES:
            src_dir = _PHASE_CROPS / split / phase
            dst_dir = dest / split / phase
            dst_dir.mkdir(parents=True, exist_ok=True)

            real = find_images(src_dir)
            for path in real:
                (dst_dir / f"real_{path.name}").symlink_to(path.resolve())

            if split != "train" or ratio <= 0:
                continue

            want = int(round(len(real) * ratio))
            pool = synthetic.get(phase, [])
            if not pool:
                continue
            if want > len(pool):
                print(
                    f"  WARNING: {phase} wants {want} synthetic but only {len(pool)} generated; "
                    f"using all {len(pool)} (ratio effectively "
                    f"{len(pool) / max(1, len(real)):.2f})"
                )
                want = len(pool)
            for path in pool[:want]:
                (dst_dir / f"syn_{path.parent.parent.parent.parent.name}_{path.name}").symlink_to(
                    path.resolve()
                )
            counts[phase] = want

    return counts


def train_and_score(run_dir: Path, crops_dir: Path, arch: str, seed: int) -> dict[str, Any]:
    """Train one classifier on the mixed dataset and score it on the real test split."""
    from allium_cepa_classifier.training.trainer import run_training

    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        experiment_name=run_dir.name,
        model={"arch": arch, "pretrained": True, "freeze_stages": 3},
        training={
            "epochs": 40,
            "lr": 1e-4,
            "early_stopping_patience": 10,
            "class_weight_multipliers": {},
            "tensorboard": False,
        },
        data={"crops_dir": crops_dir, "image_size": (260, 260), "batch_size": 32, "seed": seed},
    )
    return run_training(cfg, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 downstream synthetic-data validation.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
        help="Synthetic images per real training image, per phase.",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Training repeats per configuration.")
    parser.add_argument(
        "--arch",
        default="resnet50",
        help="Evaluator backbone. MUST differ from the judge (efficientnet_b2) or the result "
        "is circular.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Synthetic images to generate per phase (must cover the largest ratio).",
    )
    parser.add_argument(
        "--judge-filter",
        type=float,
        default=None,
        metavar="MIN_CONF",
        help="Keep only generations the judge assigns to their intended phase with at least this "
        "confidence (e.g. 0.7). Trades volume for label correctness. Off by default: the "
        "unfiltered arm is the honest baseline, and filtering is itself a variable to test.",
    )
    args = parser.parse_args()

    if args.arch == "efficientnet_b2":
        sys.exit(
            "Refusing to run: --arch efficientnet_b2 is the judge architecture. Using it here "
            "makes the result circular — generations were optimised to please that model."
        )

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for cfg_path in args.configs:
        lora_cfg = LoRAExperimentConfig.from_yaml(cfg_path)
        lora_dir = cfg_path.resolve().parent
        tag = (
            f"{lora_dir.parent.name}_{lora_dir.name}"
            if lora_dir.name.startswith("trial")
            else lora_dir.name
        )

        print(
            f"\n{'=' * 70}\n{tag}: generating up to {args.samples} synthetic images per phase\n{'=' * 70}"
        )
        synthetic = ensure_phase_samples(lora_cfg, lora_dir, args.samples)
        if args.judge_filter is not None:
            synthetic = filter_by_judge(synthetic, args.judge_filter)

        for ratio in args.ratios:
            per_seed: list[dict[str, Any]] = []
            for seed in range(args.seeds):
                name = f"{tag}_r{ratio:g}_s{seed}"
                run_dir = _OUT_ROOT / name
                crops_dir = _OUT_ROOT / f"_data_{name}"

                added = build_mixed_dataset(crops_dir, synthetic, ratio)
                print(f"\n[{name}] synthetic added per phase: {added or 'none (real only)'}")
                metrics = train_and_score(run_dir, crops_dir, args.arch, seed)
                per_seed.append(metrics)
                print(
                    f"[{name}] macro_f1={metrics['macro_f1']:.4f}  "
                    f"test_acc={metrics['test_acc']:.4f}  {metrics['per_class_f1']}"
                )
                shutil.rmtree(crops_dir, ignore_errors=True)

            macro = [m["macro_f1"] for m in per_seed]
            row = {
                "experiment": tag,
                "ratio": ratio,
                "seeds": args.seeds,
                "macro_f1_mean": round(float(np.mean(macro)), 4),
                "macro_f1_std": round(float(np.std(macro)), 4),
                "per_class_f1_mean": {
                    phase: round(float(np.mean([m["per_class_f1"][phase] for m in per_seed])), 4)
                    for phase in sorted(per_seed[0]["per_class_f1"])
                },
            }
            results.append(row)
            print(
                f"\n>>> {tag} ratio={ratio:g}: macro_f1 {row['macro_f1_mean']} +/- {row['macro_f1_std']}"
            )

    out_path = _OUT_ROOT / "downstream_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print(
        f"\n\n{'=' * 90}\nDOWNSTREAM VALIDATION (evaluator={args.arch}, real test crops only)\n{'=' * 90}"
    )
    header = f"{'experiment':<28}{'ratio':>7}{'macro_f1':>18}  per-phase F1"
    print(header)
    print("-" * len(header))
    for row in results:
        cells = "  ".join(f"{k[:4]} {v:.3f}" for k, v in row["per_class_f1_mean"].items())
        print(
            f"{row['experiment']:<28}{row['ratio']:>7.2f}"
            f"{row['macro_f1_mean']:>11.4f} ±{row['macro_f1_std']:.3f}  {cells}"
        )
    print(f"\nBaseline is ratio=0.00 (real only). Wrote {out_path}")
    print(
        "Read differences against the seed spread: at n=118 test crops, anything inside "
        "~2x the std is noise."
    )


if __name__ == "__main__":
    main()
