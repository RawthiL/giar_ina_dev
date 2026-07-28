"""
Thin wrapper that drives the kohya sd-scripts LoRA trainer from a Pydantic YAML config,
mirroring scripts/train_controlnet.py.

It translates `experiments/lora/<name>/config.yaml` into an
`accelerate launch scripts/vendor/sd-scripts/<entrypoint> <args>` subprocess and
writes outputs into the experiment dir:

    experiments/lora/<name>/
    ├── weights/   ← trained LoRA safetensors
    ├── logs/      ← tensorboard event files (tensorboard --logdir <this>)
    └── train.log  ← captured stdout/stderr of the run

LoRA is a standalone synthetic-data capability; it is NOT loaded at inference
time by AlliumCepaModel.

Usage:
    uv run python scripts/train_lora.py --config experiments/lora/sd15_rank16/config.yaml
    uv run python scripts/train_lora.py --config experiments/lora/sd15_rank16/config.yaml --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

SD_SCRIPTS = Path(__file__).resolve().parent / "vendor" / "sd-scripts"

ENTRYPOINTS = {
    "sd15": "train_network.py",
    "sd2": "train_network.py",
    "sdxl": "sdxl_train_network.py",
    "sd3": "sd3_train_network.py",
}


def build_cmd(cfg: LoRAExperimentConfig, run_dir: Path) -> list[str]:
    entry = SD_SCRIPTS / ENTRYPOINTS[cfg.model.model_family]
    out_dir = run_dir / "weights"
    log_dir = run_dir / "logs"
    train_dir = (
        cfg.data.dataset_dir / cfg.data.dataset_version / cfg.data.train_data_dir
    ).resolve()

    args = [
        "accelerate",
        "launch",
        str(entry),
        f"--pretrained_model_name_or_path={cfg.model.pretrained_model_name_or_path}",
        f"--output_dir={out_dir}",
        f"--output_name={cfg.experiment_name}",
        f"--logging_dir={log_dir}",
        f"--resolution={cfg.model.resolution}",
        f"--network_module={cfg.network.network_module}",
        f"--network_dim={cfg.network.network_dim}",
        f"--network_alpha={cfg.network.network_alpha}",
        f"--train_batch_size={cfg.training.train_batch_size}",
        f"--learning_rate={cfg.training.learning_rate}",
        f"--lr_scheduler={cfg.training.lr_scheduler}",
        f"--lr_warmup_steps={cfg.training.lr_warmup_steps}",
        f"--optimizer_type={cfg.training.optimizer_type}",
        f"--mixed_precision={cfg.training.mixed_precision}",
        f"--save_model_as={cfg.training.save_model_as}",
        f"--seed={cfg.training.seed}",
        f"--gradient_accumulation_steps={cfg.training.gradient_accumulation_steps}",
    ]

    # SD3 uses --dataset_config (TOML) instead of --train_data_dir
    if cfg.model.model_family == "sd3":
        if cfg.data.dataset_config is None:
            sys.exit(
                "SD3 training requires data.dataset_config (path to a TOML dataset config). "
                "Set it in your config.yaml."
            )
        args.append(f"--dataset_config={cfg.data.dataset_config.resolve()}")
    else:
        args += [
            f"--train_data_dir={train_dir}",
            f"--caption_extension={cfg.data.caption_extension}",
        ]

    # Optional flags
    if cfg.training.logging:
        args.append("--log_with=tensorboard")
    if cfg.training.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if cfg.training.enable_bucket and cfg.model.model_family != "sd3":
        args += [
            "--enable_bucket",
            f"--min_bucket_reso={cfg.training.min_bucket_reso}",
            f"--max_bucket_reso={cfg.training.max_bucket_reso}",
        ]
    if cfg.training.cache_latents:
        args.append("--cache_latents")
    if cfg.training.cache_latents_to_disk:
        args.append("--cache_latents_to_disk")
    if cfg.training.max_train_epochs is not None:
        args.append(f"--max_train_epochs={cfg.training.max_train_epochs}")
    if cfg.training.max_train_steps is not None:
        args.append(f"--max_train_steps={cfg.training.max_train_steps}")
    if cfg.training.noise_offset is not None:
        args.append(f"--noise_offset={cfg.training.noise_offset}")
    if cfg.training.min_snr_gamma is not None:
        args.append(f"--min_snr_gamma={cfg.training.min_snr_gamma}")
    if cfg.training.ip_noise_gamma is not None:
        args.append(f"--ip_noise_gamma={cfg.training.ip_noise_gamma}")
    if cfg.training.unet_lr is not None:
        args.append(f"--unet_lr={cfg.training.unet_lr}")
    if cfg.training.text_encoder_lr is not None:
        args.append(f"--text_encoder_lr={cfg.training.text_encoder_lr}")
    if cfg.training.save_every_n_epochs is not None:
        args.append(f"--save_every_n_epochs={cfg.training.save_every_n_epochs}")
    if cfg.network.network_dropout is not None:
        args.append(f"--network_dropout={cfg.network.network_dropout}")
    if cfg.network.network_train_unet_only:
        args.append("--network_train_unet_only")
    if cfg.network.network_train_text_encoder_only:
        args.append("--network_train_text_encoder_only")
    if cfg.network.network_args:
        # nargs="*" in kohya: --network_args key=val key=val  (separate list elements)
        args.append("--network_args")
        args.extend(cfg.network.network_args)

    # SD1.x / SD2.x specific
    if cfg.model.v2:
        args.append("--v2")
    if cfg.model.v_parameterization:
        args.append("--v_parameterization")
    if cfg.model.clip_skip is not None:
        args.append(f"--clip_skip={cfg.model.clip_skip}")

    # SD3 specific
    if cfg.model.model_family == "sd3":
        if cfg.model.clip_l is not None:
            args.append(f"--clip_l={cfg.model.clip_l}")
        if cfg.model.clip_g is not None:
            args.append(f"--clip_g={cfg.model.clip_g}")
        if cfg.model.t5xxl is not None:
            args.append(f"--t5xxl={cfg.model.t5xxl}")
        if cfg.model.vae is not None:
            args.append(f"--vae={cfg.model.vae}")
        if cfg.training.sdpa:
            args.append("--sdpa")
        if cfg.training.weighting_scheme is not None:
            args.append(f"--weighting_scheme={cfg.training.weighting_scheme}")
        if cfg.training.blocks_to_swap is not None:
            args.append(f"--blocks_to_swap={cfg.training.blocks_to_swap}")
        if cfg.training.cache_text_encoder_outputs:
            args.append("--cache_text_encoder_outputs")

    # Sampling during training (kohya writes PNGs to weights/sample/)
    if cfg.sampling.enabled:
        prompts_file = run_dir / "sample_prompts.txt"
        prompts_file.write_text("\n".join(cfg.sampling.prompts))
        args += [
            f"--sample_prompts={prompts_file}",
            f"--sample_sampler={cfg.sampling.sampler}",
        ]
        if cfg.sampling.at_first:
            args.append("--sample_at_first")
        if cfg.sampling.every_n_steps is not None:
            args.append(f"--sample_every_n_steps={cfg.sampling.every_n_steps}")
        if cfg.sampling.every_n_epochs is not None:
            args.append(f"--sample_every_n_epochs={cfg.sampling.every_n_epochs}")

    return args


def run(cmd: list[str], log_path: Path, extra_env: dict[str, str] | None = None) -> int:
    """Run cmd, streaming stdout/stderr live to console and to log_path."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env=env,
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
        help="Validate config + presence of sd-scripts and dataset, print the assembled "
        "command, and exit without launching training.",
    )
    args = parser.parse_args()

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    cmd = build_cmd(cfg, run_dir)
    train_dir = cfg.data.dataset_dir / cfg.data.dataset_version / cfg.data.train_data_dir

    if args.dry_run:
        entry = SD_SCRIPTS / ENTRYPOINTS[cfg.model.model_family]
        if not entry.exists():
            sys.exit(
                f"kohya entrypoint not found: {entry}\nRun: git submodule update --init --recursive"
            )
        if cfg.model.model_family != "sd3" and not train_dir.exists():
            sys.exit(
                f"Dataset not prepared: {train_dir} is missing. "
                "Run `dvc repro prepare_lora_dataset` first."
            )
        print("Dry run OK. Command:\n  " + " ".join(cmd))
        return

    # Add sd-scripts to PYTHONPATH so kohya's library/ package is importable
    extra_env = {"PYTHONPATH": str(SD_SCRIPTS)}
    returncode = run(cmd, run_dir / "train.log", extra_env=extra_env)
    if returncode != 0:
        sys.exit(returncode)

    # Bridge kohya sample PNGs into TensorBoard IMAGES tab
    if cfg.sampling.enabled:
        bridge_script = Path(__file__).resolve().parent / "utils" / "lora_tb_bridge.py"
        subprocess.run([sys.executable, str(bridge_script), str(run_dir)], check=False)

    print(f"\nDone. Weights in {run_dir / 'weights'} | tensorboard --logdir {run_dir / 'logs'}")


if __name__ == "__main__":
    main()
