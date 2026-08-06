"""
Merge a kohya-format LoRA safetensors into a base SD1.5 diffusers model directory.

After merging, point pretrained_model_name_or_path in your ControlNet config at
the output directory — the vendored train_controlnet.py will load it as a normal
diffusers model with the LoRA concept already baked in.

Usage:
    uv run python scripts/utils/merge_lora_into_diffusers.py \\
        --lora experiments/lora/p2_min_snr/weights/p2_min_snr.safetensors \\
        --output experiments/controlnet/sd15_lora_p2_min_snr/base_model \\
        [--base stable-diffusion-v1-5/stable-diffusion-v1-5] \\
        [--ratio 1.0]
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge kohya LoRA into a diffusers model dir.")
    parser.add_argument("--lora", required=True, type=Path, help="Path to kohya .safetensors LoRA.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination diffusers model directory."
    )
    parser.add_argument(
        "--base",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Base model: HF model ID or local diffusers dir (default: SD1.5).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="LoRA merge strength 0–1 (default: 1.0 = fully merged).",
    )
    args = parser.parse_args()

    if not args.lora.exists():
        raise FileNotFoundError(f"LoRA file not found: {args.lora}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model from: {args.base}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )

    print(f"Loading LoRA weights from: {args.lora}  (ratio={args.ratio})")
    pipe.load_lora_weights(str(args.lora))
    pipe.fuse_lora(lora_scale=args.ratio)
    pipe.unload_lora_weights()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {args.output}")
    pipe.save_pretrained(str(args.output))

    print("\nDone. Update your ControlNet config:")
    print(f"  pretrained_model_name_or_path: {args.output}")


if __name__ == "__main__":
    main()
