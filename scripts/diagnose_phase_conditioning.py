"""
Phase 0 diagnostic: does the phase word in the prompt actually control what the LoRA generates?

Generates N images per mitotic phase from a trained LoRA, classifies them with the 4-class
phase judge, and reports the confusion matrix. The number that matters is the **mean diagonal**:

    ~0.25  -> chance. Phase conditioning FAILED; the LoRA learned one generic mitotic-cell mode
              and the phase token does nothing. Any metric or HPO built on top of this is
              optimizing the wrong thing.
    >0.60  -> conditioning works well enough to build on.

Writes `phase_conditioning.json` and `plots/phase_conditioning.png` into the LoRA run dir.

Usage:
    uv run python scripts/diagnose_phase_conditioning.py \
        --config experiments/lora/p2_full_tricks/config.yaml
    uv run python scripts/diagnose_phase_conditioning.py --config ... --samples 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

from allium_cepa_classifier.config.base_config import find_project_root
from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.lora_samples import ensure_phase_samples, existing_samples  # noqa: E402
from utils.phase_judge import load_phase_judge  # noqa: E402

_ROOT = find_project_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose LoRA phase conditioning.")
    parser.add_argument("--config", required=True, type=Path, help="LoRA experiment config.")
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=_ROOT / "experiments/phase_classifier/efficientnet_b2",
        help="Run dir of the trained 4-class phase classifier.",
    )
    parser.add_argument("--samples", type=int, default=100, help="Images per phase.")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip generation and classify whatever is already in plots/diagnostic/.",
    )
    args = parser.parse_args()

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent

    judge = load_phase_judge(args.judge_dir, device=args.device)
    phases = judge.class_names  # index order matches the judge's output columns
    print(f"Judge classes: {phases}")

    if args.reuse_existing:
        samples = existing_samples(run_dir, phases)
    else:
        samples = ensure_phase_samples(
            cfg,
            run_dir,
            args.samples,
            phases=phases,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            device=args.device,
        )

    # Classify: rows = prompted phase, cols = predicted phase
    matrix = np.zeros((len(phases), len(phases)), dtype=float)
    per_phase_n: dict[str, int] = {}
    for row, phase in enumerate(phases):
        paths = samples.get(phase, [])
        per_phase_n[phase] = len(paths)
        if not paths:
            print(f"WARNING: no generated images for {phase}")
            continue
        preds = judge.probs(paths).argmax(axis=1)
        for p in preds:
            matrix[row, p] += 1
        matrix[row] /= len(paths)

    diagonal = {phase: round(float(matrix[i, i]), 4) for i, phase in enumerate(phases)}
    mean_diagonal = round(float(np.mean(list(diagonal.values()))), 4)
    chance = 1.0 / len(phases)

    result = {
        "experiment_name": cfg.experiment_name,
        "samples_per_phase": per_phase_n,
        "phases": phases,
        "confusion_matrix": matrix.round(4).tolist(),
        "diagonal": diagonal,
        "mean_diagonal": mean_diagonal,
        "chance_level": chance,
        "verdict": (
            "FAILED — at or below chance"
            if mean_diagonal <= chance * 1.2
            else "WEAK — above chance but unreliable"
            if mean_diagonal < 0.60
            else "OK"
        ),
    }

    (run_dir / "phase_conditioning.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=phases,
        yticklabels=phases,
        ax=ax,
    )
    ax.set_title(
        f"Phase conditioning — {cfg.experiment_name}\n"
        f"mean diagonal {mean_diagonal:.1%} (chance {chance:.0%}) — {result['verdict']}"
    )
    ax.set_xlabel("Predicted by judge")
    ax.set_ylabel("Prompted phase")
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "phase_conditioning.png", dpi=150)
    plt.close(fig)

    print("\nConfusion matrix (rows = prompted, cols = predicted):")
    header = " " * 12 + "".join(f"{p:>12}" for p in phases)
    print(header)
    for i, phase in enumerate(phases):
        print(f"{phase:<12}" + "".join(f"{matrix[i, j]:>11.1%}" for j in range(len(phases))))
    print(f"\nDiagonal: {diagonal}")
    print(f"Mean diagonal: {mean_diagonal:.1%}  (chance = {chance:.0%})")
    print(f"Verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
