"""
Build the 4-class mitotic-phase classifier dataset from the tagged VAE crops.

The upstream VAE splits cannot be used as-is. Two defects, both measured 2026-08-09:

  1. **Exact-duplicate leakage.** 200 of the 242 test crops (83%) are byte-identical to
     training crops, and 42 test crops also appear in validation. Of 1441 files only
     1194 are unique. Any metric computed on that test set is mostly memorisation.
  2. **Source-micrograph leakage.** Crops were split per-crop, so cells cropped from the
     same field of view land in different splits and share staining, illumination, focus
     and background cues.

This script fixes both:

  * deduplicates by content hash (md5), keeping the first occurrence in a deterministic order;
  * groups crops by source micrograph, normalising away Roboflow re-upload hashes so the
     same underlying image re-uploaded twice maps to one group;
  * assigns whole groups to splits with a greedy stratified fill, so every crop from one
     micrograph lands in exactly one split while per-phase proportions stay near target.

Pre-existing `*_aug*` copies are dropped: they only ever covered the old train split, so
after re-splitting their coverage would be arbitrary. The trainer applies hflip/vflip/
color-jitter at runtime instead.

A `split_manifest.json` recording every group assignment and content hash is written
alongside the dataset so the split is auditable and reproducible.

Output layout (what `trainer.py` expects):
    {train,validation,test}/{prophase,metaphase,anaphase,telophase}/*.png

Usage:
    uv run python scripts/utils/phase_classifier_dataset.py
    uv run python scripts/utils/phase_classifier_dataset.py --ratios 0.8 0.1 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from allium_cepa_classifier.config.base_config import find_project_root

_ROOT = find_project_root()

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
SPLITS = ["train", "validation", "test"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Source dirs, relative to --vae-dir. The test split has no `tagged/` level.
SOURCE_DIRS = ["train/tagged", "val/tagged", "test"]


@dataclass(frozen=True)
class Crop:
    path: Path
    phase: str
    group: str
    digest: str


def group_key(stem: str) -> str:
    """
    Normalise a crop filename to the source micrograph it came from.

    `004_00099_2`                                  -> `004_00099`
    `IMG_1598_JPG.rf.4a36ab39...._0`               -> `img_1598`
    `bdddec82-0wU3Y_png_jpg.rf.1e3da6fb...._9`     -> `bdddec82-0wu3y`
    `bdddec82-0wU3Y_png.rf.9148092c...._9`         -> `bdddec82-0wu3y`   (same image, re-uploaded)
    """
    s = re.sub(r"_(\d+)$", "", stem)  # trailing annotation index
    s = re.sub(r"\.rf\.[0-9a-f]+.*$", "", s)  # Roboflow re-upload hash
    s = re.sub(r"[_.](jpg|jpeg|png)$", "", s, flags=re.IGNORECASE)
    return s.lower()


def collect_crops(vae_dir: Path) -> list[Crop]:
    """Every unique original crop across all upstream splits, deduplicated by content."""
    seen: set[str] = set()
    crops: list[Crop] = []
    duplicates = 0

    # Sorted iteration keeps which-copy-we-keep deterministic across runs.
    for source in SOURCE_DIRS:
        for phase in PHASES:
            phase_dir = vae_dir / source / phase
            if not phase_dir.is_dir():
                print(f"WARNING: missing source dir {phase_dir}")
                continue
            for path in sorted(phase_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES or "_aug" in path.stem:
                    continue
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                crops.append(Crop(path, phase, group_key(path.stem), digest))

    print(f"Collected {len(crops)} unique crops ({duplicates} exact duplicates dropped).")
    return crops


def assign_groups(crops: list[Crop], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """
    Greedy stratified group assignment: whole micrographs only, per-phase counts near target.

    Groups are placed largest-first so the biggest ones (here 70 and 55 crops) settle before
    the small ones fine-tune the balance. Each group goes to whichever split it leaves least
    over-filled, measured as the worst per-phase fill ratio.
    """
    by_group: dict[str, list[Crop]] = defaultdict(list)
    for crop in crops:
        by_group[crop.group].append(crop)

    phase_totals = {p: sum(1 for c in crops if c.phase == p) for p in PHASES}
    targets = {
        s: {p: phase_totals[p] * r for p, r in ((p, ratios[s]) for p in PHASES)} for s in SPLITS
    }
    counts: dict[str, dict[str, int]] = {s: dict.fromkeys(PHASES, 0) for s in SPLITS}

    # Largest first; hash of the group name breaks ties deterministically but seed-dependently.
    ordered = sorted(
        by_group.items(),
        key=lambda kv: (-len(kv[1]), hashlib.md5(f"{seed}:{kv[0]}".encode()).hexdigest()),
    )

    def worst_fill(split: str, group_phases: dict[str, int]) -> float:
        """How over-filled `split` would be, per phase, if this group were added to it."""
        return max(
            (counts[split][p] + group_phases[p]) / targets[split][p]
            for p in PHASES
            if targets[split][p] > 0
        )

    assignment: dict[str, str] = {}
    for group, members in ordered:
        group_phases = {p: sum(1 for c in members if c.phase == p) for p in PHASES}
        best = min(SPLITS, key=lambda s: worst_fill(s, group_phases))  # noqa: B023
        assignment[group] = best
        for p in PHASES:
            counts[best][p] += group_phases[p]

    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the mitotic-phase classifier dataset.")
    parser.add_argument("--vae-dir", type=Path, default=_ROOT / "datasets/crops/vae")
    parser.add_argument("--out", type=Path, default=_ROOT / "datasets/crops/phase_classifier")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = dict(zip(SPLITS, args.ratios, strict=True))

    crops = collect_crops(args.vae_dir)
    if not crops:
        raise SystemExit(f"No crops found under {args.vae_dir}")

    assignment = assign_groups(crops, ratios, args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)

    counts: dict[str, dict[str, int]] = {s: dict.fromkeys(PHASES, 0) for s in SPLITS}
    hashes: dict[str, set[str]] = {s: set() for s in SPLITS}
    manifest: list[dict[str, str]] = []

    for crop in crops:
        split = assignment[crop.group]
        dst_dir = args.out / split / crop.phase
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(crop.path, dst_dir / crop.path.name)

        counts[split][crop.phase] += 1
        hashes[split].add(crop.digest)
        manifest.append(
            {
                "file": crop.path.name,
                "phase": crop.phase,
                "group": crop.group,
                "split": split,
                "md5": crop.digest,
            }
        )

    (args.out / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "ratios": ratios,
                "n_crops": len(crops),
                "n_groups": len(assignment),
                "crops": manifest,
            },
            indent=2,
        )
        + "\n"
    )

    # Hard guarantees — a silent regression here invalidates every downstream metric.
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                overlap = hashes[a] & hashes[b]
                assert not overlap, f"content overlap {a}/{b}: {len(overlap)} crops"
    groups_per_split: dict[str, set[str]] = defaultdict(set)
    for group, split in assignment.items():
        groups_per_split[split].add(group)
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                shared = groups_per_split[a] & groups_per_split[b]
                assert not shared, f"group overlap {a}/{b}: {sorted(shared)[:5]}"

    print(f"\nWrote {args.out}")
    header = f"{'split':<12}" + "".join(f"{p:>12}" for p in PHASES) + f"{'total':>9}{'groups':>9}"
    print(header)
    print("-" * len(header))
    for split in SPLITS:
        row = counts[split]
        line = f"{split:<12}" + "".join(f"{row[p]:>12}" for p in PHASES)
        print(line + f"{sum(row.values()):>9}{len(groups_per_split[split]):>9}")
    total = sum(sum(r.values()) for r in counts.values())
    print(f"{'TOTAL':<12}" + " " * (12 * len(PHASES)) + f"{total:>9}{len(assignment):>9}")
    print("\nVerified: no content-hash overlap and no source-micrograph overlap across splits.")


if __name__ == "__main__":
    main()
