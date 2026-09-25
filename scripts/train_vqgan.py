"""
Thin wrapper that drives the vendored diffusers VQGAN trainer from a
Pydantic YAML config, mirroring scripts/train_controlnet.py.

It translates `experiments/vqgan/<name>/config.yaml` into an
`accelerate launch scripts/vendor/train_vqgan.py <args>` subprocess and
writes outputs into the experiment dir:

    experiments/vqgan/<name>/
    ├── weights/   ← vqmodel + discriminator (config.json + diffusion_pytorch_model.safetensors)
    ├── logs/      ← tensorboard event files (tensorboard --logdir <this>)
    └── train.log  ← captured stdout/stderr of the run

VQGAN is a standalone compressor; it is NOT loaded at inference time by
AlliumCepaModel.

Usage:
    uv run python scripts/train_vqgan.py --config experiments/vqgan/vqgan_baseline/config.yaml
    uv run python scripts/train_vqgan.py --config experiments/vqgan/vqgan_baseline/config.yaml --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from allium_cepa_classifier.config.vqgan_config import VQGANExperimentConfig

VENDORED = Path(__file__).resolve().parent / "vendor" / "vqgan"
VENDORED_TRAINER = VENDORED / "train_vqgan.py"


def build_cmd(cfg: VQGANExperimentConfig, run_dir: Path) -> list[str]:
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    train_dir = cfg.data.dataset_dir.resolve()
    validation_images = [
        str((cfg.data.dataset_dir.parent / img).resolve()) for img in cfg.validation.images
    ]

    args = [
        "accelerate",
        "launch",
        str(VENDORED_TRAINER),
        f"--train_data_dir={train_dir}",
        f"--image_column={cfg.data.image_column}",
        f"--output_dir={weights_dir}",
        f"--logging_dir={logs_dir}",
        f"--resolution={cfg.model.resolution}",
        f"--train_batch_size={cfg.training.train_batch_size}",
        f"--num_train_epochs={cfg.training.num_train_epochs}",
        f"--gradient_accumulation_steps={cfg.training.gradient_accumulation_steps}",
        f"--learning_rate={cfg.training.learning_rate}",
        f"--discr_learning_rate={cfg.training.discr_learning_rate}",
        f"--lr_scheduler={cfg.training.lr_scheduler}",
        f"--discr_lr_scheduler={cfg.training.discr_lr_scheduler}",
        f"--lr_warmup_steps={cfg.training.lr_warmup_steps}",
        f"--adam_beta1={cfg.training.adam_beta1}",
        f"--adam_beta2={cfg.training.adam_beta2}",
        f"--adam_weight_decay={cfg.training.adam_weight_decay}",
        f"--adam_epsilon={cfg.training.adam_epsilon}",
        f"--max_grad_norm={cfg.training.max_grad_norm}",
        f"--seed={cfg.training.seed}",
        f"--mixed_precision={cfg.training.mixed_precision}",
        f"--vae_loss={cfg.model.vae_loss}",
        f"--timm_model_backend={cfg.model.timm_model_backend}",
        f"--timm_model_layers={cfg.model.timm_model_layers}",
        f"--timm_model_offset={cfg.model.timm_model_offset}",
        f"--validation_steps={cfg.training.validation_steps}",
        f"--checkpointing_steps={cfg.training.checkpointing_steps}",
        f"--log_steps={cfg.training.log_steps}",
        f"--log_grad_norm_steps={cfg.training.log_grad_norm_steps}",
        f"--report_to={cfg.training.report_to}",
        f"--dataloader_num_workers={cfg.training.dataloader_num_workers}",
        *["--validation_images", *validation_images],
    ]

    if cfg.model.pretrained_model_name_or_path is not None:
        args.append(f"--pretrained_model_name_or_path={cfg.model.pretrained_model_name_or_path}")
    if cfg.model.model_config_name_or_path is not None:
        args.append(f"--model_config_name_or_path={cfg.model.model_config_name_or_path}")

    if cfg.training.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if cfg.training.enable_xformers:
        args.append("--enable_xformers_memory_efficient_attention")
    if cfg.training.use_ema:
        args.append("--use_ema")
    if cfg.training.use_8bit_adam:
        args.append("--use_8bit_adam")
    if cfg.training.allow_tf32:
        args.append("--allow_tf32")
    if cfg.training.scale_lr:
        args.append("--scale_lr")
    if cfg.model.center_crop:
        args.append("--center_crop")
    if cfg.model.random_flip:
        args.append("--random_flip")
    if cfg.training.max_train_steps is not None:
        args.append(f"--max_train_steps={cfg.training.max_train_steps}")
    if cfg.training.max_train_samples is not None:
        args.append(f"--max_train_samples={cfg.training.max_train_samples}")
    if cfg.training.checkpoints_total_limit is not None:
        args.append(f"--checkpoints_total_limit={cfg.training.checkpoints_total_limit}")

    return args


def run(cmd: list[str], log_path: Path) -> int:
    """Run cmd, streaming stdout/stderr live to console and to log_path."""
    vendor_pythonpath = str(VENDORED)
    env = {**os.environ, "PYTHONPATH": f"{vendor_pythonpath}:{os.environ.get('PYTHONPATH', '')}"}
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            cmd,
            env=env,
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

    cfg = VQGANExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    if not torch.cuda.is_available() and cfg.training.mixed_precision != "no":
        print("WARNING: CUDA not available, forcing mixed_precision=no (fp16 requires GPU)")
        cfg.training.mixed_precision = "no"

    cmd = build_cmd(cfg, run_dir)
    train_dir = cfg.data.dataset_dir

    if args.dry_run:
        if not VENDORED_TRAINER.exists():
            sys.exit(f"Vendored trainer not found: {VENDORED_TRAINER}")
        if not (VENDORED / "discriminator.py").exists():
            sys.exit(f"Vendored discriminator not found: {VENDORED / 'discriminator.py'}")
        if not train_dir.exists():
            sys.exit(f"Dataset not found: {train_dir}")
        for img_rel in cfg.validation.images:
            img_path = (cfg.data.dataset_dir.parent / img_rel).resolve()
            if not img_path.exists():
                sys.exit(f"Validation image not found: {img_path}")
        print("Dry run OK. Command:\n  " + " ".join(cmd))
        return

    print(f"Training — log: {run_dir / 'train.log'}")
    returncode = run(cmd, run_dir / "train.log")
    if returncode != 0:
        sys.exit(returncode)
    print(f"\nDone. Weights in {run_dir / 'weights'} | tensorboard --logdir {run_dir / 'logs'}")


if __name__ == "__main__":
    main()
