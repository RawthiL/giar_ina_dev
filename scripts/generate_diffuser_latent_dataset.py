"""
Generate a diffuser latent dataset for SD-VAE adversarial fine-tuning.

Produces a balanced set of latents from a LoRA-fine-tuned SD pipeline.
The primary output is a Parquet file with raw (pre-scaling) denoised latents.
Optional PNG/npy exports mirror the original generate_cells.py behaviour.

Resolution and VAE come from the training config when --config is passed (so a LoRA
trained on a fine-tuned 256px VAE yields matching (4,32,32) latents decoded with the same
VAE); explicit --resolution/--vae override it. Without --config it defaults to 512px / the
base VAE for backward compatibility with the existing adversarial flow.

Usage:
    uv run python scripts/generate_diffuser_latent_dataset.py --config experiments/lora/vae256_typecue/config.yaml --n-total 400 --out datasets/latents/diffuser_latents_256 --seed 42
    uv run python scripts/generate_diffuser_latent_dataset.py --config ... --n-total 400 --save-images --save-npy --add_type_cue
    uv run python scripts/generate_diffuser_latent_dataset.py --lora untracked/lora_cell_generator/trial_020.safetensors --base-model stable-diffusion-v1-5/stable-diffusion-v1-5 --n-total 400 --out datasets/latents/diffuser_latents
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

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


def _load_pipeline(base_model, lora_path, model_family, device, dtype, vae_path=None):
    import importlib

    module_name, class_name = PIPELINE_CLASSES[model_family]
    PipelineClass = getattr(importlib.import_module(module_name), class_name)
    kwargs = {"torch_dtype": dtype, "safety_checker": None}
    if vae_path:
        from diffusers import AutoencoderKL

        kwargs["vae"] = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=dtype)
    pipe = PipelineClass.from_pretrained(base_model, **kwargs).to(device)
    if lora_path:
        pipe.load_lora_weights(str(lora_path))
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, help="LoRAExperimentConfig yaml")
    ap.add_argument("--lora", type=Path, help="LoRA safetensors path")
    ap.add_argument("--base-model", type=str, default=None, help="Override config model base")
    ap.add_argument("--model-family", type=str, default=None, choices=list(PIPELINE_CLASSES.keys()))
    ap.add_argument("--vae", type=Path, default=None, help="Override config model.vae (custom VAE)")
    ap.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override config model.resolution (pixel/latent size); default 512",
    )
    ap.add_argument("--n-total", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--save-npy", action="store_true")
    ap.add_argument("--keep-noise", action="store_true")
    ap.add_argument(
        "--add_type_cue",
        action="store_true",
        help="Append a cycling structural image-type cue to prompts (matches cued LoRAs)",
    )
    args = ap.parse_args()

    if args.n_total % len(PHASES) != 0:
        raise SystemExit(f"--n-total must be divisible by {len(PHASES)}")

    # Defaults (used when no --config). Config values override these; explicit CLI args
    # override config, so a --config resolution of 256 yields matching (4,32,32) latents.
    base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    model_family = "sd15"
    vae_path = None
    resolution = None

    if args.config:
        from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

        cfg = LoRAExperimentConfig.from_yaml(args.config)
        base_model = cfg.model.pretrained_model_name_or_path
        model_family = cfg.model.model_family
        vae_path = cfg.model.vae
        resolution = cfg.model.resolution
        lora_path = args.lora or (
            Path(args.config).parent / "weights" / f"{cfg.experiment_name}.safetensors"
        )
    else:
        if not args.lora:
            raise SystemExit("Either --config or --lora must be provided")
        lora_path = args.lora

    # Explicit CLI overrides beat config
    if args.base_model:
        base_model = args.base_model
    if args.model_family:
        model_family = args.model_family
    if args.vae is not None:
        vae_path = args.vae
    if args.resolution is not None:
        resolution = args.resolution
    if resolution is None:
        resolution = 512

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    pipe = _load_pipeline(base_model, lora_path, model_family, args.device, dtype, vae_path)
    lat_h = lat_w = resolution // 8
    lat_shape = (1, pipe.unet.config.in_channels, lat_h, lat_w)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**32 - 1, size=args.n_total, dtype=np.int64)

    jobs = []
    per_phase = args.n_total // len(PHASES)
    for k in range(per_phase):
        for p, phase in enumerate(PHASES):
            idx = k * len(PHASES) + p
            jobs.append({"idx": idx, "phase": phase, "seed": int(seeds[idx])})

    latents_list = []
    phase_ids = []
    phase_names = []

    # Prepare optional outputs
    if args.save_images or args.save_npy:
        img_dir = out_dir / "images"
        lat_dir = out_dir / "latents"
        img_dir.mkdir(exist_ok=True)
        lat_dir.mkdir(exist_ok=True)

    def _prompt_for(job):
        if args.add_type_cue:
            cue = TYPE_CUES[job["idx"] % len(TYPE_CUES)]
            return CAPTION_TEMPLATE_TYPE.format(phase=job["phase"], type=cue)
        return CAPTION_TEMPLATE.format(phase=job["phase"])

    print(f"Generating {args.n_total} samples ({per_phase} per phase)...")
    for b in tqdm(range(0, len(jobs), args.batch_size), desc="Generating batches", unit="batch"):
        batch = jobs[b : b + args.batch_size]
        prompts = [_prompt_for(j) for j in batch]
        # Build deterministic noise per seed
        noise_tensors = []
        for j in batch:
            gen = torch.Generator("cpu").manual_seed(j["seed"])
            noise_tensors.append(torch.randn(lat_shape, generator=gen, dtype=torch.float32))
        noise = torch.cat(noise_tensors).to(args.device, dtype=dtype)

        out = pipe(
            prompts,
            negative_prompt=[NEGATIVE_PROMPT] * len(batch),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            latents=noise,
            output_type="latent",
        )
        final_latents = out.images  # (B,4,H,W), UNet-scaled latents
        final_latents_cpu = final_latents.float().cpu()

        # Decode the whole batch once: per-sample rows are unbatched (C,H,W) and feeding
        # them to vae.decode() trips group_norm (channels misread as the batch dim).
        pil_imgs = None
        if args.save_images:
            with torch.no_grad():
                decoded = pipe.vae.decode(final_latents / pipe.vae.config.scaling_factor).sample
            pil_imgs = pipe.image_processor.postprocess(decoded, output_type="pil")

        for pos, (j, z, init_noise) in enumerate(
            zip(batch, final_latents_cpu, noise_tensors, strict=True)
        ):
            latents_list.append(z.numpy())
            phase_ids.append(PHASES.index(j["phase"]))
            phase_names.append(j["phase"])
            stem = f"{j['idx']:05d}_{j['phase']}_seed{j['seed']}"

            if args.save_npy:
                np.save(lat_dir / f"{stem}_final_latent.npy", z.numpy())
                if args.keep_noise:
                    np.save(lat_dir / f"{stem}_init_noise.npy", init_noise.numpy())

            if pil_imgs is not None:
                pil_imgs[pos].save(out_dir / "images" / f"{stem}.png")

    latents_arr = np.stack(latents_list, axis=0).astype(np.float32)
    n = latents_arr.shape[0]
    # Build Parquet with FixedSizeList
    flat = pa.array(latents_arr.reshape(-1), type=pa.float32())
    list_size = 4 * lat_h * lat_w
    latent_col = pa.FixedSizeListArray.from_arrays(flat, list_size=list_size)
    table = pa.table(
        {
            "index": pa.array(np.arange(n, dtype=np.int64)),
            "seed": pa.array(seeds, type=pa.int64()),
            "phase_id": pa.array(phase_ids, type=pa.int64()),
            "latent": latent_col,
        }
    )
    parquet_path = out_dir / "diffuser_latents.parquet"
    pq.write_table(table, parquet_path)

    config = {
        "latent_shape": [4, lat_h, lat_w],
        "dtype": "float32",
        "phase_names": PHASES,
        "base_model": base_model,
        "lora": str(lora_path),
        "vae": str(vae_path) if vae_path else None,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "resolution": resolution,
        "add_type_cue": bool(args.add_type_cue),
        "seed": args.seed,
        "latent_space": "pre-scaling-factor",
        "n_samples": int(n),
    }
    (out_dir / "diffuser_latents.config.json").write_text(json.dumps(config, indent=2))

    print(f"Saved parquet → {parquet_path}")
    print(f"Saved config  → {out_dir / 'diffuser_latents.config.json'}")
    if args.save_images:
        print(f"Images → {out_dir / 'images'}")
    if args.save_npy:
        print(f"NPY   → {out_dir / 'latents'}")


if __name__ == "__main__":
    main()
