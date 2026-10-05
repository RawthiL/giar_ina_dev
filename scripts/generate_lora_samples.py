"""
Generate sample images from a trained LoRA checkpoint.

Loads the trained LoRA weights into the base SD pipeline (family-aware),
generates `--samples` images per mitotic phase (prophase, metaphase, anaphase,
telophase) and saves them to experiments/lora/<name>/plots/generated/{phase}/.

Also writes a 4×min(samples,4) grid to plots/lora_samples.png for quick visual
inspection.

Default of 13 per phase (52 total) provides enough images for FID and classifier
judge metrics in evaluate_lora.py.

When model.vae is set in the config, that custom (fine-tuned) VAE replaces the base
model's and images are generated at model.resolution (e.g. 256px), matching the latent
space the LoRA was trained on. Pass --add_type_cue to append a structural image-type cue
("and type {annotated|numbered|web|scraped|camera}") to every prompt, cycling the cue
across samples — mirrors the caption format used when training on the 'typecue' dataset.

Usage:
    uv run python scripts/generate_lora_samples.py --config experiments/lora/sd15_rank16/config.yaml
    uv run python scripts/generate_lora_samples.py --config ... --samples 5 --seed 99
    uv run python scripts/generate_lora_samples.py --config ... --add_type_cue
"""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from name_types import CAPTION_TEMPLATE, CAPTION_TEMPLATE_TYPE, TYPE_CUES  # noqa: E402

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
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
    kwargs = {"torch_dtype": dtype, "safety_checker": None}
    # Custom fine-tuned VAE (e.g. the 256px one): replaces the base model's VAE so the
    # decode step matches the latent space the LoRA was trained against.
    if cfg.model.vae is not None:
        from diffusers import AutoencoderKL

        kwargs["vae"] = AutoencoderKL.from_pretrained(str(cfg.model.vae), torch_dtype=dtype)
    pipe = PipelineClass.from_pretrained(cfg.model.pretrained_model_name_or_path, **kwargs).to(
        device
    )
    pipe.load_lora_weights(str(lora_path))
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--samples",
        type=int,
        default=13,
        help="Images per phase (default 13, giving 52 total for FID/classifier metrics).",
    )
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--add_type_cue",
        action="store_true",
        help="Append a structural image-type cue ('and type <cue>') to each prompt, "
        "cycling the cue set across samples. Cues mirror the training captions "
        "(annotated, numbered, web, scraped, camera).",
    )
    args = parser.parse_args()

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    lora_path = run_dir / "weights" / f"{cfg.experiment_name}.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}. Run training first.")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    generated_dir = plots_dir / "generated"
    generated_dir.mkdir(exist_ok=True)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"Loading {cfg.model.model_family} pipeline ...")
    pipe = load_pipeline(cfg, lora_path, args.device, dtype)

    # Generate and save all images, collecting the first grid_rows per phase for the grid
    res = cfg.model.resolution
    grid_rows = min(args.samples, 4)
    grid_images: dict[str, list[Image.Image]] = {phase: [] for phase in PHASES}
    grid_labels: dict[str, list[str]] = {phase: [] for phase in PHASES}

    for phase in PHASES:
        phase_dir = generated_dir / phase
        phase_dir.mkdir(exist_ok=True)
        print(f"Generating {args.samples} images for phase: {phase}")
        for i in range(args.samples):
            if args.add_type_cue:
                cue = TYPE_CUES[i % len(TYPE_CUES)]
                prompt = CAPTION_TEMPLATE_TYPE.format(phase=phase, type=cue)
                fname = f"{i:03d}_{cue}.png"
            else:
                cue = ""
                prompt = CAPTION_TEMPLATE.format(phase=phase)
                fname = f"{i:03d}.png"
            generator = torch.manual_seed(args.seed + i)
            img = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=args.num_inference_steps,
                height=res,
                width=res,
                generator=generator,
            ).images[0]
            img.save(phase_dir / fname)
            if i < grid_rows:
                grid_images[phase].append(img)
                grid_labels[phase].append(cue)

    total = args.samples * len(PHASES)
    cue_note = f" | type cue: {'/'.join(TYPE_CUES)}" if args.add_type_cue else ""
    print(f"\nSaved {total} images @ {res}px to {generated_dir}{cue_note}")

    # Build grid: rows = first grid_rows samples, cols = phases
    fig, axes = plt.subplots(
        nrows=grid_rows,
        ncols=len(PHASES),
        figsize=(4 * len(PHASES), 4 * grid_rows),
        squeeze=False,
    )
    fig.suptitle(f"LoRA samples — {cfg.experiment_name} @ {res}px", fontsize=14)
    for col, phase in enumerate(PHASES):
        for row, img in enumerate(grid_images[phase]):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(phase, fontsize=11, fontweight="bold")
            if args.add_type_cue and grid_labels[phase][row]:
                axes[row, col].text(
                    0.5,
                    -0.03,
                    grid_labels[phase][row],
                    transform=axes[row, col].transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="dimgray",
                )

    fig.tight_layout()
    grid_path = plots_dir / "lora_samples.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid saved to {grid_path}")


if __name__ == "__main__":
    main()
