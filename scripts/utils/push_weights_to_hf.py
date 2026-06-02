"""
Upload DVC-tracked weight files to the HuggingFace model repo.

DVC's HTTP remote is read-only, so this script replicates what `dvc push`
would do: it reads each .dvc pointer file, finds the corresponding file in
the local DVC cache, and uploads it to the HF repo at the path DVC expects
(files/md5/<2>/<rest>).

Usage:
    uv run python scripts/utils/push_weights_to_hf.py [--dry-run]
"""

import argparse
import configparser
from pathlib import Path

import yaml
from huggingface_hub import HfApi

REPO_ID = "GIAR-UTN/allium-cepa-automation"
CACHE_DIR = Path(".dvc/cache/files/md5")

DVC_POINTERS = [
    "src/allium_cepa_classifier/weights/object_detection.pt.dvc",
    "src/allium_cepa_classifier/weights/classifier_calibrated.pt.dvc",
    "src/allium_cepa_classifier/weights/yolo_isotonic_calibrator.pkl.dvc",
]


def read_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(".dvc/config.local")
    try:
        return cfg["'remote \"weights\"'"]["password"]
    except KeyError:
        raise RuntimeError(
            "No 'weights' remote token found in .dvc/config.local. "
            "Run: dvc remote modify --local weights password <your_hf_token>"
        )


def push(dry_run: bool = False) -> None:
    token = read_token()
    api = HfApi(token=token)

    for dvc_path in DVC_POINTERS:
        with open(dvc_path) as f:
            info = yaml.safe_load(f)
        out = info["outs"][0]
        md5: str = out["md5"]
        size: int = out["size"]
        name = Path(dvc_path).stem

        local = CACHE_DIR / md5[:2] / md5[2:]
        remote = f"files/md5/{md5[:2]}/{md5[2:]}"

        if not local.exists():
            print(f"  MISSING from cache: {name} ({local})")
            print(f"  Run: dvc add {Path(dvc_path).with_suffix('')}")
            continue

        size_mb = size / 1024 / 1024
        print(f"  {name}  ({size_mb:.1f} MB)  →  {remote}")

        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print("    uploaded.")
        else:
            print("    (dry-run, skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push DVC weight files to HuggingFace.")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading.")
    args = parser.parse_args()

    print(f"Target repo: {REPO_ID}")
    push(dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
