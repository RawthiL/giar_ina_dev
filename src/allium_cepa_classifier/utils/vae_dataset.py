import argparse
import shutil

from allium_cepa_classifier.config.training_config import TrainingConfig


def collect_mitosis_split(split: str, cfg: TrainingConfig) -> int:
    """Copy all mitosis crops from one split into the mitosis_crops directory."""
    src_dir = cfg.binary_classifier_crops_dir / split / "mitosis"

    if not src_dir.exists():
        print(f"Skipping {split}: {src_dir} not found.")
        return 0

    cfg.vae_crops_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in src_dir.iterdir():
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        dest = cfg.vae_crops_dir / f"{image_path.name}"
        shutil.copy2(image_path, dest)
        copied += 1

    print(f"  {split}: {copied} images copied.")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect all mitosis crops from every split into a single directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Uses TrainingConfig defaults if omitted.",
    )
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config, key="training") if args.config else TrainingConfig()

    print(f"Source:      {cfg.binary_classifier_crops_dir}")
    print(f"Destination: {cfg.vae_crops_dir}\n")

    total = 0
    for split in cfg.splits:
        total += collect_mitosis_split(split, cfg)

    print(f"\nDone. {total} mitosis images collected into {cfg.vae_crops_dir}")


if __name__ == "__main__":
    main()
