from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class LoRAModelConfig(BaseModel):
    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # Selects the kohya entrypoint: train_network.py / sdxl_train_network.py / sd3_train_network.py
    model_family: Literal["sd15", "sd2", "sdxl", "sd3"] = "sd15"
    resolution: int = 512
    # SD2.x options (ignored for other families)
    v2: bool = False
    v_parameterization: bool = False
    clip_skip: int | None = None
    # SD3/3.5 encoder paths — optional when using a unified .safetensors file
    clip_l: Path | None = None
    clip_g: Path | None = None
    t5xxl: Path | None = None
    vae: Path | None = None


class LoRANetworkConfig(BaseModel):
    network_module: str = "networks.lora"  # or "lycoris.kohya"
    network_dim: int = 16  # LoRA rank
    network_alpha: float = 8.0
    network_dropout: float | None = None
    network_args: list[str] = []  # e.g. ["algo=locon", "conv_dim=8", "conv_alpha=4"]
    network_train_unet_only: bool = False
    network_train_text_encoder_only: bool = False


class LoRATrainingConfig(BaseModel):
    train_batch_size: int = 2
    # NOTE: do NOT set both max_train_epochs and max_train_steps at the same time.
    # kohya unconditionally overwrites max_train_steps with the epoch-based value when
    # max_train_epochs is present, so the step cap is silently ignored.
    # Prefer max_train_steps for time-bounded sweeps; use max_train_epochs only when
    # you want full-epoch training without a step cap.
    max_train_epochs: int | None = None
    max_train_steps: int | None = None
    learning_rate: float = 1e-4
    unet_lr: float | None = None
    text_encoder_lr: float | None = None
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    optimizer_type: str = "AdamW8bit"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    mixed_precision: str = "fp16"
    seed: int = 42
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    save_every_n_epochs: int | None = None
    save_model_as: str = "safetensors"
    logging: bool = True  # --log_with tensorboard
    # Noise & loss shaping (SD1.x / SD2.x)
    noise_offset: float | None = None  # ~0.05-0.1; shifts noise to allow dark/bright generation
    min_snr_gamma: float | None = None  # 5 recommended; down-weights high-loss low-SNR timesteps
    ip_noise_gamma: float | None = None  # ~0.1; adds perturbation to clean latent (regularization)
    # Caption regularisation. Every crop of a phase carries an identical caption, and generation
    # uses that one prompt with only the seed varying, which structurally caps output diversity —
    # the `coverage` metric has been the weakest term throughout. Dropping the caption on a
    # fraction of steps forces the UNet to rely less on the fixed text embedding.
    # Incompatible with cache_text_encoder_outputs (kohya asserts).
    caption_dropout_rate: float | None = None  # 0.0-1.0
    shuffle_caption: bool = False
    # SD3/3.5 specific
    sdpa: bool = False  # use scaled dot-product attention (recommended for SD3)
    weighting_scheme: str | None = None  # e.g. "uniform" for SD3
    blocks_to_swap: int | None = None  # swap N transformer blocks to CPU (SD3 VRAM reduction)
    cache_text_encoder_outputs: bool = False  # SD3: cache 3 text-encoder outputs


class LoRADataConfig(BaseModel):
    dataset_dir: Path = _ROOT / "datasets/crops/lora"
    dataset_version: str = "baseline"  # named subfolder under dataset_dir
    train_data_dir: str = "img"  # kohya scans img/<repeats>_<concept>/
    caption_extension: str = ".txt"
    # SD3/3.5 requires a TOML dataset config instead of --train_data_dir.
    # When set, overrides train_data_dir in the wrapper command.
    dataset_config: Path | None = None


class LoRASamplingConfig(BaseModel):
    enabled: bool = True
    every_n_steps: int | None = None
    every_n_epochs: int | None = 1
    sampler: str = "euler_a"
    at_first: bool = True
    prompts: list[str] = [
        "micrograph of allium cepa root tip mitotic cell in prophase phase --w 512 --h 512 --s 25",
        "micrograph of allium cepa root tip mitotic cell in metaphase phase --w 512 --h 512 --s 25",
        "micrograph of allium cepa root tip mitotic cell in anaphase phase --w 512 --h 512 --s 25",
        "micrograph of allium cepa root tip mitotic cell in telophase phase --w 512 --h 512 --s 25",
    ]


class LoRAExperimentConfig(BaseConfig):
    experiment_name: str
    model: LoRAModelConfig = LoRAModelConfig()
    network: LoRANetworkConfig = LoRANetworkConfig()
    training: LoRATrainingConfig = LoRATrainingConfig()
    data: LoRADataConfig = LoRADataConfig()
    sampling: LoRASamplingConfig = LoRASamplingConfig()
