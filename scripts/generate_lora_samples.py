"""
Generate sample images from a trained LoRA checkpoint.

Loads the trained LoRA weights into the base SD pipeline (family-aware),
generates `--samples` images per cell class (prophase, metaphase, anaphase,
telophase, interphase) and saves them to experiments/lora/<name>/plots/generated/{phase}/.

Also writes a 5×min(samples,4) grid to plots/lora_samples.png for quick visual
inspection.

Default of 13 per phase (65 total) provides enough images for FID and classifier
judge metrics in evaluate_lora.py.

When model.vae is set in the config, that custom (fine-tuned) VAE replaces the base
model's and images are generated at model.resolution (e.g. 256px), matching the latent
space the LoRA was trained on. Prompts combine the phase with observed train VLM tag
tuples; the VLM ``detailed_mitotic_stage`` is ignored.

``--tag-tuples-json`` overrides the phase loops and generates exactly the requested
combinations. Each item must be::

    {"phase": "prophase", "tags": {"focus_quality": "sharp_high_definition"}, "repeat": 1}

Manual tags are validated strictly: unknown fields, values outside the fixed
dictionary, and phase-specific fields used with the wrong phase raise an error.

Usage:
    uv run python scripts/generate_lora_samples.py --config experiments/lora/vae256_tagcue/config.yaml
    uv run python scripts/generate_lora_samples.py --config ... --samples 5 --seed 99
    uv run python scripts/generate_lora_samples.py --config ... --tag-tuples-json combos.json
"""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from name_types import PHASES  # noqa: E402
from vlm_tags import (  # noqa: E402
    TagValidationError,
    build_train_tag_index,
    prompt_for,
    sample_tag_tuple,
)

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
        help="Images per phase (default 13, giving 65 total for FID/classifier metrics).",
    )
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crops-dir", type=Path, default=Path("datasets/crops/vae_256"))
    parser.add_argument("--tags-dir", type=Path, default=Path("datasets/crops/vae_256_vlm-tags"))
    parser.add_argument(
        "--tag-tuples-json",
        type=Path,
        default=None,
        help="Optional JSON list of {phase, tags, repeat?} items. Generates exactly those "
        "combinations instead of the per-phase sample loop.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Default repeats per --tag-tuples-json item when the item has no 'repeat'.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

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

    manual_jobs = None
    if args.tag_tuples_json is not None:
        try:
            raw = json.loads(Path(args.tag_tuples_json).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot read --tag-tuples-json: {exc}") from exc
        if not isinstance(raw, list) or not raw:
            raise SystemExit("--tag-tuples-json must contain a non-empty JSON list")
        manual_jobs = []
        for item_idx, item in enumerate(raw):
            if not isinstance(item, dict) or "phase" not in item or "tags" not in item:
                raise SystemExit(f"Invalid tag tuple #{item_idx}: expected 'phase' and 'tags' keys")
            phase = item["phase"]
            tags = item["tags"]
            try:
                prompt_for(phase, tags)
            except TagValidationError as exc:
                raise SystemExit(f"Invalid tag tuple #{item_idx}: {exc}") from exc
            repeat = int(item.get("repeat", args.repeat))
            if repeat < 1:
                raise SystemExit(f"Invalid repeat for tag tuple #{item_idx}: {repeat}")
            for _ in range(repeat):
                manual_jobs.append({"phase": phase, "tags": tags})

    # Generate and save all images, collecting up to 4 samples per phase for the grid
    res = cfg.model.resolution
    grid_images: dict[str, list[Image.Image]] = {phase: [] for phase in PHASES}

    def _generate(prompt: str, seed: int, phase: str, fname: str) -> Image.Image:
        phase_dir = generated_dir / phase
        phase_dir.mkdir(exist_ok=True)
        generator = torch.manual_seed(seed)
        img = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.num_inference_steps,
            height=res,
            width=res,
            generator=generator,
        ).images[0]
        img.save(phase_dir / fname)
        if len(grid_images[phase]) < 4:
            grid_images[phase].append(img)
        return img

    if manual_jobs is not None:
        print(f"Generating {len(manual_jobs)} manual tag-tuple samples ...")
        for idx, job in enumerate(manual_jobs):
            phase = job["phase"]
            _generate(
                prompt_for(phase, job["tags"]),
                args.seed + idx,
                phase,
                f"{idx:03d}.png",
            )
        total = len(manual_jobs)
    else:
        tag_index = build_train_tag_index(args.crops_dir, args.tags_dir, PHASES)
        tag_rng = random.Random(args.seed ^ 0x5EED)
        for phase in PHASES:
            print(f"Generating {args.samples} images for phase: {phase}")
            for i in range(args.samples):
                tags = sample_tag_tuple(phase, tag_index, tag_rng)
                _generate(
                    prompt_for(phase, tags),
                    args.seed + i,
                    phase,
                    f"{i:03d}.png",
                )
        total = args.samples * len(PHASES)

    print(f"\nSaved {total} images @ {res}px to {generated_dir}")

    # Build grid: rows = first up-to-4 samples, cols = phases with images
    phases_to_plot = [phase for phase in PHASES if grid_images[phase]]
    if not phases_to_plot:
        print("No images generated; skipping grid.")
        return
    grid_rows = min(max(len(grid_images[phase]) for phase in phases_to_plot), 4)
    fig, axes = plt.subplots(
        nrows=grid_rows,
        ncols=len(phases_to_plot),
        figsize=(4 * len(phases_to_plot), 4 * grid_rows),
        squeeze=False,
    )
    fig.suptitle(f"LoRA samples — {cfg.experiment_name} @ {res}px", fontsize=14)
    for col, phase in enumerate(phases_to_plot):
        for row, img in enumerate(grid_images[phase]):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
        for row in range(len(grid_images[phase]), grid_rows):
            axes[row, col].axis("off")
        axes[0, col].set_title(phase, fontsize=11, fontweight="bold")

    fig.tight_layout()
    grid_path = plots_dir / "lora_samples.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid saved to {grid_path}")


if __name__ == "__main__":
    main()
