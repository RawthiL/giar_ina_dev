"""
Shared per-phase sample generation for a trained LoRA.

Both `diagnose_phase_conditioning.py` and `evaluate_lora.py` need the same thing: N images
per mitotic phase, generated from the phase-conditioned prompt, cached on disk so repeated
scoring is free. They share this module so a metric and a diagnostic are never computed over
differently-generated images.

Samples live in `<run_dir>/plots/diagnostic/<phase>/NNNN.png`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch

from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"
NEGATIVE_PROMPT = "blurry, low quality, deformed, malformed, text, watermark, jpeg artifacts"

PIPELINE_CLASSES = {
    "sd15": ("diffusers", "StableDiffusionPipeline"),
    "sd2": ("diffusers", "StableDiffusionPipeline"),
    "sdxl": ("diffusers", "StableDiffusionXLPipeline"),
    "sd3": ("diffusers", "StableDiffusion3Pipeline"),
}


def load_pipeline(cfg: LoRAExperimentConfig, lora_path: Path, device: str, dtype: torch.dtype):
    module_name, class_name = PIPELINE_CLASSES[cfg.model.model_family]
    PipelineClass = getattr(importlib.import_module(module_name), class_name)
    pipe = PipelineClass.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.load_lora_weights(str(lora_path))
    return pipe


def samples_dir(run_dir: Path) -> Path:
    return run_dir / "plots" / "diagnostic"


def existing_samples(run_dir: Path, phases: list[str] | None = None) -> dict[str, list[Path]]:
    root = samples_dir(run_dir)
    return {p: sorted((root / p).glob("*.png")) for p in (phases or PHASES)}


def ensure_phase_samples(
    cfg: LoRAExperimentConfig,
    run_dir: Path,
    n_per_phase: int,
    phases: list[str] | None = None,
    num_inference_steps: int = 25,
    seed: int = 1234,
    device: str | None = None,
) -> dict[str, list[Path]]:
    """
    Guarantee at least `n_per_phase` cached samples per phase, generating only the shortfall.

    Seeds are `seed + index`, so extending an existing set keeps the images already on disk
    reproducible rather than regenerating them under different noise.
    """
    phases = phases or PHASES
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = samples_dir(run_dir)

    current = existing_samples(run_dir, phases)
    missing = {p: n_per_phase - len(paths) for p, paths in current.items()}
    if all(n <= 0 for n in missing.values()):
        return {p: paths[:n_per_phase] for p, paths in current.items()}

    lora_path = run_dir / "weights" / f"{cfg.experiment_name}.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {cfg.model.model_family} pipeline + LoRA ...")
    pipe = load_pipeline(cfg, lora_path, device, dtype)
    pipe.set_progress_bar_config(disable=True)

    try:
        for phase in phases:
            need = missing[phase]
            if need <= 0:
                continue
            phase_dir = root / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            start = len(current[phase])
            print(f"Generating {need} × {phase} (have {start}) ...")
            prompt = CAPTION_TEMPLATE.format(phase=phase)
            for i in range(start, start + need):
                generator = torch.manual_seed(seed + i)
                img = pipe(
                    prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
                img.save(phase_dir / f"{i:04d}.png")
    finally:
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    return {p: paths[:n_per_phase] for p, paths in existing_samples(run_dir, phases).items()}
