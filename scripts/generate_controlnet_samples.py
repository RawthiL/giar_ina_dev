"""
Generate sample images from a trained ControlNet checkpoint.

Standalone replacement for the notebook's testing cells: loads the trained
ControlNet + an SD1.5 pipeline, runs it over a few (conditioning image, prompt)
pairs from the test split, and saves a control-vs-generated grid to
`experiments/controlnet/<name>/plots/controlnet_samples.png`.

ControlNet is a standalone synthetic-data generator; this script is not part of
the AlliumCepaModel inference pipeline.

Usage:
    uv run python scripts/generate_controlnet_samples.py --config experiments/controlnet/sd15_baseline/config.yaml
    uv run python scripts/generate_controlnet_samples.py --config <cfg> --images a.png b.png --prompts "cell in prophase" "cell in metaphase"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

from allium_cepa_classifier.config.controlnet_config import (
    ControlNetExperimentConfig,
)

# The mitotic phases the model was conditioned to reproduce (notebook defaults).
DEFAULT_PROMPTS = [
    "cell in prophase",
    "cell in metaphase",
    "cell in anaphase",
    "cell in telophase",
]
NEGATIVE_PROMPT = "blurry, low quality, bad, deformed, malformed, text, watermark, jpeg artifacts"


def resolve_conditioning_images(cfg: ControlNetExperimentConfig, names: list[str]) -> list[Path]:
    """Resolve conditioning image names against the test split's blurred dir.

    If no names are given, fall back to the first N available test images so the
    script keeps working even as the dataset is re-split/re-packed.
    """
    blurred_dir = cfg.data.dataset_dir / cfg.data.test_split / "blurred_upscaled"
    if names:
        paths = [blurred_dir / n for n in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Conditioning image(s) not found: {missing}")
        return paths
    available = sorted(blurred_dir.glob("*.png"))
    if not available:
        raise FileNotFoundError(f"No test conditioning images in {blurred_dir}")
    return available[: len(DEFAULT_PROMPTS)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Conditioning image filenames under test/blurred_upscaled/ "
        "(default: first few available).",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help=f"Prompts, one per image (default: {DEFAULT_PROMPTS}).",
    )
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = ControlNetExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    weights_dir = run_dir / "weights"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    conditioning_paths = resolve_conditioning_images(cfg, args.images)
    prompts = args.prompts if args.prompts is not None else DEFAULT_PROMPTS
    prompts = prompts[: len(conditioning_paths)]
    conditioning_paths = conditioning_paths[: len(prompts)]
    if len(conditioning_paths) != len(prompts):
        raise ValueError("Number of conditioning images and prompts must match.")

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"Loading ControlNet from {weights_dir} on {args.device} ({dtype}) ...")
    controlnet = ControlNetModel.from_pretrained(weights_dir, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(args.device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    res = cfg.model.resolution
    controls, generated = [], []
    for i, (img_path, prompt) in enumerate(zip(conditioning_paths, prompts, strict=True)):
        print(f"[{i + 1}/{len(prompts)}] {img_path.name}  prompt='{prompt}'")
        control_image = load_image(str(img_path)).resize((res, res))
        controls.append(control_image)
        generator = torch.manual_seed(args.seed)
        out = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=control_image,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
        generated.append(out)

    n = len(generated)
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(4 * n, 8), squeeze=False)
    fig.suptitle("ControlNet Generations", fontsize=16)
    for i in range(n):
        axes[0, i].imshow(controls[i])
        axes[0, i].set_title(f"Control {i + 1}", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(generated[i])
        axes[1, i].set_title(f"Generated {i + 1}\n('{prompts[i]}')", fontsize=10, wrap=True)
        axes[1, i].axis("off")
    fig.tight_layout()

    out_path = plots_dir / "controlnet_samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
