"""
Five-class mitosis classifier (4 mitotic phases + interphase), real vs real+synthetic.

This is the experiment the project actually deliverables on: a classifier that sees the real
class balance -- interphase vastly outnumbers every mitotic phase -- and must still resolve the
four phases. Synthetic crops from a LoRA generator are added to the *training* split only, for
the four mitotic phases only; interphase is already abundant and gets nothing.

Two things this script is careful about, both learned the hard way in this repo:

1. **Split integrity across two datasets.** The 4-phase crops were split group-aware (whole
   source micrographs assigned to one split) after 83% exact-duplicate test leakage was found in
   the upstream VAE splits. The interphase crops come from a *different* pipeline (COCO
   `attributes.division == 0`) with its own per-image splits. Naively unioning them would put
   interphase crops from micrograph X in test while phase crops from X sit in train. So
   interphase crops are re-assigned here: deduplicated by md5, grouped by source micrograph with
   the same `group_key` normalisation, and any group already claimed by the phase manifest
   inherits that split. Zero md5 and zero group overlap across splits is asserted, not hoped for.

2. **Circularity.** `validate_synthetic_downstream.py` refuses `--arch efficientnet_b2` because
   that is the phase judge, and generations were selected on a `phase_consistency` score that
   judge produced. Here b2 is used *deliberately*: it is the deliverable architecture and the
   thing we actually need an answer about. The bias is real and one-directional (b2 should
   benefit more from b2-pleasing generations than a neutral architecture would), so the resnet50
   numbers from Phase 4 remain the unbiased reference. The interphase class is uncontaminated
   either way -- the 4-class judge never saw it.

Usage:
    uv run python scripts/validate_synthetic_5class.py \
        --lora experiments/lora/_studies/lora_nofill_v1/trial_020/config.yaml \
        --ratios 0 0.5 max --seeds 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.phase_classifier_dataset import group_key  # noqa: E402

from allium_cepa_classifier.config.base_config import find_project_root  # noqa: E402
from allium_cepa_classifier.config.experiment_config import ExperimentConfig  # noqa: E402

_ROOT = find_project_root()
PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
CLASSES = [*PHASES, "interphase"]
SPLITS = ["train", "validation", "test"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

_PHASE_CROPS = _ROOT / "datasets" / "crops" / "phase_classifier"
_BINARY_CROPS = _ROOT / "datasets" / "crops" / "binary_classifier"
_BASE_5CLASS = _ROOT / "datasets" / "crops" / "phase5_classifier"
_OUT_ROOT = _ROOT / "experiments" / "phase_classifier" / "_downstream_5class"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def find_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


# --------------------------------------------------------------------------------------
# Building the real 5-class dataset
# --------------------------------------------------------------------------------------
def build_real_5class(
    ratios: tuple[float, float, float], seed: int, interphase_ratio: float
) -> dict[str, Any]:
    """
    Assemble {split}/{class}/ as symlinks: phase crops keep their audited split, interphase
    crops are deduplicated and assigned group-aware, inheriting the split of any group the
    phase manifest already claimed.
    """
    manifest_path = _PHASE_CROPS / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    group_split: dict[str, str] = {c["group"]: c["split"] for c in manifest["crops"]}
    phase_digests: set[str] = {c["md5"] for c in manifest["crops"]}
    inherited_groups = set(group_split)

    # Collect interphase crops from every binary split; the binary splits are discarded and
    # recomputed here, because they are per-crop and not group-aware.
    seen: set[str] = set()
    by_group: dict[str, list[Path]] = defaultdict(list)
    n_dupes = 0
    n_cross_class = 0
    for split in SPLITS:
        for path in find_images(_BINARY_CROPS / split / "no_mitosis"):
            digest = md5(path)
            if digest in seen:
                n_dupes += 1
                continue
            if digest in phase_digests:
                # Same bytes labelled both interphase and a mitotic phase: trust the phase label.
                n_cross_class += 1
                continue
            seen.add(digest)
            by_group[group_key(path.stem)].append(path)

    rng = random.Random(seed)
    groups = sorted(by_group)
    rng.shuffle(groups)

    new_groups = [g for g in groups if g not in inherited_groups]
    n_inherited = len(groups) - len(new_groups)

    # Greedy fill on crop counts so split proportions land near target despite uneven groups.
    assigned: dict[str, str] = {}
    counts = dict.fromkeys(SPLITS, 0)
    for g in groups:
        if g in inherited_groups:
            assigned[g] = group_split[g]
            counts[assigned[g]] += len(by_group[g])
    total_new = sum(len(by_group[g]) for g in new_groups)
    target = {
        s: r * (sum(counts.values()) + total_new) for s, r in zip(SPLITS, ratios, strict=True)
    }
    for g in sorted(new_groups, key=lambda g: -len(by_group[g])):
        split = min(SPLITS, key=lambda s: counts[s] - target[s])
        assigned[g] = split
        counts[split] += len(by_group[g])

    if _BASE_5CLASS.exists():
        shutil.rmtree(_BASE_5CLASS)

    stats: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    split_digests: dict[str, set[str]] = {s: set() for s in SPLITS}
    split_groups: dict[str, set[str]] = {s: set() for s in SPLITS}

    for split in SPLITS:
        for phase in PHASES:
            dst = _BASE_5CLASS / split / phase
            dst.mkdir(parents=True, exist_ok=True)
            srcs = find_images(_PHASE_CROPS / split / phase)
            for p in srcs:
                (dst / p.name).symlink_to(p.resolve())
                split_groups[split].add(group_key(p.stem))
            stats[split][phase] = len(srcs)

        dst = _BASE_5CLASS / split / "interphase"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        n_mitotic = sum(stats[split][ph] for ph in PHASES)
        cap = int(round(n_mitotic * interphase_ratio)) if interphase_ratio > 0 else 0
        members = sorted(
            (g for g in by_group if assigned[g] == split), key=lambda g: (-len(by_group[g]), g)
        )
        for g in members:
            if cap and n >= cap:
                break
            split_groups[split].add(g)
            for p in by_group[g]:
                (dst / p.name).symlink_to(p.resolve())
                split_digests[split].add(md5(p))
                n += 1
        stats[split]["interphase"] = n

    # Integrity: no content and no source micrograph may straddle two splits.
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            assert not (split_digests[a] & split_digests[b]), f"md5 overlap {a}/{b}"
            assert not (split_groups[a] & split_groups[b]), (
                f"group overlap {a}/{b}: {sorted(split_groups[a] & split_groups[b])[:5]}"
            )

    report = {
        "counts": stats,
        "interphase_duplicates_dropped": n_dupes,
        "interphase_dropped_as_phase_duplicate": n_cross_class,
        "interphase_groups_inheriting_phase_split": n_inherited,
        "interphase_groups_newly_assigned": len(new_groups),
        "interphase_per_mitotic": interphase_ratio,
    }
    (_BASE_5CLASS / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# --------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------
def build_arm(dest: Path, synthetic: dict[str, list[Path]], ratio: float | str) -> dict[str, int]:
    """Copy the real 5-class tree by symlink, adding synthetic to train's mitotic phases."""
    if dest.exists():
        shutil.rmtree(dest)
    added: dict[str, int] = {}
    for split in SPLITS:
        for cls in CLASSES:
            src_dir = _BASE_5CLASS / split / cls
            dst_dir = dest / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            reals = find_images(src_dir)
            for p in reals:
                (dst_dir / f"real_{p.name}").symlink_to(p.resolve())
            if split != "train" or cls == "interphase" or ratio == 0:
                continue
            pool = synthetic.get(cls, [])
            want = len(pool) if ratio == "max" else int(round(len(reals) * float(ratio)))
            want = min(want, len(pool))
            for p in pool[:want]:
                (dst_dir / f"syn_{p.name}").symlink_to(p.resolve())
            added[cls] = want
    return added


def train_and_score(run_dir: Path, crops_dir: Path, seed: int) -> dict[str, Any]:
    from allium_cepa_classifier.training.trainer import run_training

    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(
        experiment_name=run_dir.name,
        model={
            "arch": "efficientnet_b2",
            "pretrained": True,
            "freeze_stages": 3,
            "head": {
                "hidden_dims": [512, 256, 128],
                "dropouts": [0.3, 0.2, 0.0],
                "activation": "leaky_relu",
            },
        },
        training={
            "epochs": 40,
            "lr": 1e-4,
            "early_stopping_patience": 10,
            "class_weight_multipliers": {},
            "lr_scheduler": {"factor": 0.2, "patience": 5, "min_lr": 1e-6},
            "augmentation": ["hflip", "vflip", "color_jitter"],
            "tensorboard": False,
        },
        data={"crops_dir": crops_dir, "image_size": (260, 260), "batch_size": 32, "seed": seed},
    )
    return run_training(cfg, run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path, required=True, help="LoRA experiment config.yaml")
    ap.add_argument("--ratios", nargs="+", default=["0", "0.5", "max"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split-ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument(
        "--interphase-ratio",
        type=float,
        default=8.0,
        help="Interphase crops per mitotic crop, applied identically in every split "
        "(0 = keep all 34:1). Same prior everywhere keeps the arm comparison honest.",
    )
    ap.add_argument("--rebuild", action="store_true", help="Rebuild the real 5-class base.")
    args = ap.parse_args()

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.rebuild or not _BASE_5CLASS.exists():
        print("Building real 5-class dataset ...")
        report = build_real_5class(tuple(args.split_ratios), args.split_seed, args.interphase_ratio)
        print(json.dumps(report, indent=2))
    else:
        report = json.loads((_BASE_5CLASS / "split_report.json").read_text())
        print("Reusing existing 5-class base:")
        print(json.dumps(report["counts"], indent=2))

    lora_dir = args.lora.resolve().parent
    synthetic = {
        phase: sorted((lora_dir / "plots" / "diagnostic" / phase).glob("*.png")) for phase in PHASES
    }
    print("\nsynthetic available: " + ", ".join(f"{k}={len(v)}" for k, v in synthetic.items()))
    if not all(synthetic.values()):
        sys.exit("No cached synthetic samples found; run evaluate_lora.py first.")

    tag = f"{lora_dir.parent.name}_{lora_dir.name}"
    results: list[dict[str, Any]] = []
    for ratio in args.ratios:
        r: float | str = "max" if ratio == "max" else float(ratio)
        per_seed = []
        for seed in range(args.seeds):
            name = f"{tag}_5c_r{ratio}_s{seed}"
            run_dir = _OUT_ROOT / name
            crops_dir = _OUT_ROOT / f"_data_{name}"
            # Resume: a finished run already wrote metrics.json. Reruns of an 8-hour grid after
            # a reboot should not repeat work that is already on disk.
            done = run_dir / "metrics.json"
            if done.exists():
                m = json.loads(done.read_text())
                per_seed.append(m)
                print(f"\n[{name}] REUSED existing run: macro_f1={m['macro_f1']:.4f}")
                continue
            added = build_arm(crops_dir, synthetic, r)
            print(f"\n[{name}] synthetic added: {added or 'none (real only)'}")
            m = train_and_score(run_dir, crops_dir, seed)
            per_seed.append(m)
            print(
                f"[{name}] macro_f1={m['macro_f1']:.4f} acc={m['test_acc']:.4f} {m['per_class_f1']}"
            )
            shutil.rmtree(crops_dir, ignore_errors=True)

        macro = [m["macro_f1"] for m in per_seed]
        results.append(
            {
                "experiment": tag,
                "ratio": ratio,
                "seeds": args.seeds,
                "macro_f1_mean": round(float(np.mean(macro)), 4),
                "macro_f1_std": round(float(np.std(macro)), 4),
                "test_acc_mean": round(float(np.mean([m["test_acc"] for m in per_seed])), 4),
                "per_class_f1_mean": {
                    c: round(float(np.mean([m["per_class_f1"][c] for m in per_seed])), 4)
                    for c in sorted(per_seed[0]["per_class_f1"])
                },
            }
        )
        print(
            f"\n>>> ratio={ratio}: macro_f1 {results[-1]['macro_f1_mean']} "
            f"+/- {results[-1]['macro_f1_std']}"
        )

    out = _OUT_ROOT / "results_5class.json"
    out.write_text(json.dumps(results, indent=2) + "\n")

    print("\n" + "=" * 96)
    print("FIVE-CLASS DOWNSTREAM (efficientnet_b2 = judge architecture; real test crops only)")
    print("=" * 96)
    print(f"{'ratio':>8}{'macro_f1':>18}{'acc':>8}  per-class F1")
    print("-" * 96)
    for row in results:
        pc = "  ".join(f"{k[:4]} {v:.3f}" for k, v in sorted(row["per_class_f1_mean"].items()))
        print(
            f"{row['ratio']:>8}{row['macro_f1_mean']:>11.4f} ±{row['macro_f1_std']:.3f}"
            f"{row['test_acc_mean']:>8.3f}  {pc}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
