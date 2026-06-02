"""
Thin wrapper that drives the vendored diffusers ControlNet trainer from a
Pydantic YAML config, mirroring scripts/train_vae.py.

It translates `experiments/controlnet/<name>/config.yaml` into an
`accelerate launch scripts/vendor/train_controlnet.py <args>` subprocess and
writes outputs into the experiment dir:

    experiments/controlnet/<name>/
    ├── weights/   ← ControlNet (config.json + diffusion_pytorch_model.safetensors)
    ├── logs/      ← tensorboard event files (tensorboard --logdir <this>)
    └── train.log  ← captured stdout/stderr of the run

ControlNet is a standalone synthetic-data generator; it is NOT loaded at
inference time by AlliumCepaModel.

Usage:
    uv run python scripts/train_controlnet.py --config experiments/controlnet/sd15_baseline/config.yaml
    uv run python scripts/train_controlnet.py --config experiments/controlnet/sd15_baseline/config.yaml --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

from allium_cepa_classifier.config.controlnet_config import (
    ControlNetExperimentConfig,
)

VENDORED = Path(__file__).resolve().parent / "vendor" / "train_controlnet.py"


def build_cmd(cfg: ControlNetExperimentConfig, run_dir: Path) -> list[str]:
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    train_dir = (cfg.data.dataset_dir / cfg.data.train_split).resolve()
    validation_image = (train_dir / cfg.validation.image).resolve()

    args = [
        "accelerate",
        "launch",
        str(VENDORED),
        f"--pretrained_model_name_or_path={cfg.model.pretrained_model_name_or_path}",
        f"--output_dir={weights_dir}",
        # logging_dir is joined onto output_dir by the trainer; an absolute path
        # escapes output_dir so tensorboard logs land in <run_dir>/logs, not weights/.
        f"--logging_dir={logs_dir}",
        f"--dataset_name={train_dir}",
        f"--image_column={cfg.data.image_column}",
        f"--conditioning_image_column={cfg.data.conditioning_image_column}",
        f"--caption_column={cfg.data.caption_column}",
        f"--resolution={cfg.model.resolution}",
        f"--train_batch_size={cfg.training.train_batch_size}",
        f"--num_train_epochs={cfg.training.num_train_epochs}",
        f"--learning_rate={cfg.training.learning_rate}",
        f"--lr_scheduler={cfg.training.lr_scheduler}",
        f"--lr_warmup_steps={cfg.training.lr_warmup_steps}",
        f"--seed={cfg.training.seed}",
        f"--mixed_precision={cfg.training.mixed_precision}",
        f"--validation_prompt={cfg.validation.prompt}",
        f"--validation_image={validation_image}",
        f"--validation_steps={cfg.training.validation_steps}",
        f"--checkpointing_steps={cfg.training.checkpointing_steps}",
        f"--report_to={cfg.training.report_to}",
    ]
    if cfg.training.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if cfg.training.enable_xformers:
        args.append("--enable_xformers_memory_efficient_attention")
    if cfg.training.max_train_steps is not None:
        args.append(f"--max_train_steps={cfg.training.max_train_steps}")
    return args


def run(cmd: list[str], log_path: Path) -> int:
    """Run cmd, streaming stdout/stderr live to console and to log_path."""
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        ) as proc,
    ):
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + presence of the vendored script and dataset, print the "
        "assembled command, and exit without launching training.",
    )
    args = parser.parse_args()

    cfg = ControlNetExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    cmd = build_cmd(cfg, run_dir)
    train_dir = cfg.data.dataset_dir / cfg.data.train_split

    if args.dry_run:
        if not VENDORED.exists():
            sys.exit(f"Vendored trainer not found: {VENDORED}")
        if not train_dir.exists():
            sys.exit(
                f"Dataset not prepared: {train_dir} is missing. "
                "Run `dvc repro prepare_controlnet_dataset` first."
            )
        validation_image = train_dir / cfg.validation.image
        if not validation_image.exists():
            sys.exit(f"Validation image not found: {validation_image}")
        print("Dry run OK. Command:\n  " + " ".join(cmd))
        return

    returncode = run(cmd, run_dir / "train.log")
    if returncode != 0:
        sys.exit(returncode)
    print(f"\nDone. Weights in {run_dir / 'weights'} | tensorboard --logdir {run_dir / 'logs'}")


if __name__ == "__main__":
    main()
