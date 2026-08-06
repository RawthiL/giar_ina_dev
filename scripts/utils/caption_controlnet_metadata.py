"""Populate the 'text' field in controlnet metadata.jsonl using VAE phase labels.

Phase-labeled cells get:   "micrograph of allium cepa root tip mitotic cell in {phase} phase"
All other cells get:       "micrograph of allium cepa root tip cell in mitosis"

Looks up stems in (in priority order):
  1. datasets/crops/vae/train/tagged/{phase}/
  2. datasets/crops/vae/test/{phase}/
  3. Fallback generic caption
"""

import argparse
import json
import shutil
from pathlib import Path

from allium_cepa_classifier.config.base_config import find_project_root

PHASE_CAPTION = "micrograph of allium cepa root tip mitotic cell in {phase} phase"
GENERIC_CAPTION = "micrograph of allium cepa root tip cell in mitosis"


def build_phase_map(root: Path) -> dict[str, str]:
    phase_map: dict[str, str] = {}
    for phase_dir in root.iterdir():
        if not phase_dir.is_dir():
            continue
        for img in phase_dir.iterdir():
            stem = img.stem.replace("_aug", "")
            phase_map[stem] = phase_dir.name
    return phase_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption controlnet metadata.jsonl")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.jsonl (default: datasets/crops/controlnet/train/metadata.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and sample captions without writing anything",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    meta_path: Path = (
        args.metadata or project_root / "datasets/crops/controlnet/train/metadata.jsonl"
    )

    vae_root = project_root / "datasets/crops/vae"
    phase_map: dict[str, str] = {}
    for split in ("train", "test"):
        tagged_dir = vae_root / split / "tagged"
        if tagged_dir.exists():
            phase_map.update(build_phase_map(tagged_dir))

    print(f"Phase map entries: {len(phase_map)}")

    rows = []
    stats = {"phased": 0, "generic": 0}

    with open(meta_path) as f:
        for line in f:
            row = json.loads(line)
            stem = Path(row["file_name"]).stem.replace("_sharp_512x512", "")
            phase = phase_map.get(stem)
            if phase:
                row["text"] = PHASE_CAPTION.format(phase=phase)
                stats["phased"] += 1
            else:
                row["text"] = GENERIC_CAPTION
                stats["generic"] += 1
            rows.append(row)

    print(f"Phase-captioned entries : {stats['phased']}")
    print(f"Generic-captioned entries: {stats['generic']}")
    print(f"Total                   : {len(rows)}")

    if args.dry_run:
        print("\nSample captions (first 5):")
        for row in rows[:5]:
            print(f"  {row['file_name']!r:55s} → {row['text']!r}")
        return

    backup = meta_path.with_suffix(".jsonl.bak")
    shutil.copy2(meta_path, backup)
    print(f"\nBackup written to {backup}")

    with open(meta_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Updated {meta_path}")


if __name__ == "__main__":
    main()
