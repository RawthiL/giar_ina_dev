"""
Generate sample images from a trained LoRA checkpoint.

Loads the trained LoRA weights into the base SD pipeline (family-aware),
generates 3 images per mitotic phase (prophase, metaphase, anaphase, telophase),
and saves a 4×3 grid (phase columns × sample rows) to
experiments/lora/<name>/plots/lora_samples.png.

Usage:
    uv run python scripts/generate_lora_samples.py --config experiments/lora/sd15_rank16/config.yaml
    uv run python scripts/generate_lora_samples.py --config ... --samples 5 --seed 99
"""

import argparse
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"
NEGATIVE_PROMPT = "blurry, low quality, deformed, malformed, text, watermark, jpeg artifacts"

# Pipeline class per model family
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=3, help="Images per phase (default 3).")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    lora_path = run_dir / "weights" / f"{cfg.experiment_name}.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}. Run training first.")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(
        f"Loading {cfg.model.model_family} pipeline from {cfg.model.pretrained_model_name_or_path} ..."
    )
    pipe = load_pipeline(cfg, lora_path, args.device, dtype)

    n_phases = len(PHASES)
    fig, axes = plt.subplots(
        nrows=args.samples,
        ncols=n_phases,
        figsize=(4 * n_phases, 4 * args.samples),
        squeeze=False,
    )
    fig.suptitle(f"LoRA samples — {cfg.experiment_name}", fontsize=14)

    for col, phase in enumerate(PHASES):
        prompt = CAPTION_TEMPLATE.format(phase=phase)
        print(f"Phase: {phase}")
        for row in range(args.samples):
            # Different seed per row for variation; consistent across phases for comparison
            generator = torch.manual_seed(args.seed + row)
            img = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(phase, fontsize=11, fontweight="bold")

    fig.tight_layout()
    out_path = plots_dir / "lora_samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
