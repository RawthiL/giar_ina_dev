"""
Generate a diffuser latent dataset for SD-VAE adversarial fine-tuning.

Produces a balanced set of latents from a LoRA-fine-tuned SD pipeline.
The primary output is a Parquet file with raw (pre-scaling) denoised latents.
Optional PNG/npy exports mirror the original generate_cells.py behaviour.

Resolution and VAE come from the training config when --config is passed (so a LoRA
trained on a fine-tuned 256px VAE yields matching (4,32,32) latents decoded with the same
VAE); explicit --resolution/--vae override it. Without --config/--resolution it falls back
to 512px / the base VAE (legacy raw-LoRA flows only; the vae_256 chain always passes --config).

Prompts combine the folder phase with observed train VLM tag tuples.  The VLM
``detailed_mitotic_stage`` is ignored, and no provenance source cue is used.

``--tag-tuples-json`` overrides train-tuple sampling and generates exactly the
requested combinations.  Each item must be::

    {"phase": "prophase", "tags": {"focus_quality": "sharp_high_definition"}, "repeat": 1}

Manual tags are validated strictly: unknown fields, values outside the fixed
dictionary, and phase-specific fields used with the wrong phase raise an error.

Usage:
    uv run python scripts/generate_diffuser_latent_dataset.py --config experiments/lora/vae256_tagcue/config.yaml --n-total 400 --out datasets/latents/diffuser_latents_256_tagcue --seed 42
    uv run python scripts/generate_diffuser_latent_dataset.py --config ... --n-total 400 --save-images --save-npy
    uv run python scripts/generate_diffuser_latent_dataset.py --config ... --tag-tuples-json combos.json --save-images --out ...
    uv run python scripts/generate_diffuser_latent_dataset.py --lora untracked/lora_cell_generator/trial_020.safetensors --base-model stable-diffusion-v1-5/stable-diffusion-v1-5 --n-total 400 --out datasets/latents/diffuser_latents
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

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
    ap.add_argument(
        "--n-total",
        type=int,
        default=None,
        help="Balanced sample count. Required unless --tag-tuples-json is used.",
    )
    ap.add_argument(
        "--tag-tuples-json",
        type=Path,
        default=None,
        help="Optional JSON list of {phase, tags, repeat?} items. Bypasses train-tuple "
        "sampling and generates exactly those combinations.",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Default repeats per --tag-tuples-json item when the item has no 'repeat'.",
    )
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
        "--crops-dir",
        type=Path,
        default=Path("datasets/crops/vae_256"),
        help="Source crop tree used to mirror VLM tag JSON paths.",
    )
    ap.add_argument(
        "--tags-dir",
        type=Path,
        default=Path("datasets/crops/vae_256_vlm-tags"),
        help="VLM tag tree mirroring --crops-dir.",
    )
    ap.add_argument(
        "--phases",
        nargs="+",
        choices=PHASES,
        default=PHASES,
        metavar="PHASE",
        help="Cell classes to generate, balanced. Default: all 5 (4 mitotic phases + interphase). "
        "Pass the 4 mitotic phases for the mitosis-classifier synthetic set.",
    )
    args = ap.parse_args()

    phases = list(args.phases)
    if args.tag_tuples_json is None:
        if args.n_total is None:
            raise SystemExit("--n-total is required unless --tag-tuples-json is provided")
        if args.n_total % len(phases) != 0:
            raise SystemExit(f"--n-total must be divisible by the number of phases ({len(phases)})")
    elif args.n_total is not None:
        raise SystemExit("Use either --n-total or --tag-tuples-json, not both")
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

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
        n_total = len(manual_jobs)
    else:
        n_total = args.n_total

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_total, dtype=np.int64)

    jobs = []
    if manual_jobs is not None:
        for idx, job in enumerate(manual_jobs):
            jobs.append(
                {
                    "idx": idx,
                    "phase": job["phase"],
                    "seed": int(seeds[idx]),
                    "tags": job["tags"],
                }
            )
    else:
        tag_index = build_train_tag_index(args.crops_dir, args.tags_dir, phases)
        tag_rng = random.Random(args.seed ^ 0x5EED)
        missing_phase_tags = [p for p in phases if not tag_index.get(p)]
        if missing_phase_tags:
            print(
                f"Warning: no train VLM tags for phases {missing_phase_tags}; "
                "using phase-only prompts"
            )
        per_phase = n_total // len(phases)
        for k in range(per_phase):
            for p, phase in enumerate(phases):
                idx = k * len(phases) + p
                tags = sample_tag_tuple(phase, tag_index, tag_rng)
                jobs.append({"idx": idx, "phase": phase, "seed": int(seeds[idx]), "tags": tags})

    latents_list = []
    phase_ids = []
    phase_names = []
    prompts_list = []
    tags_json_list = []

    # Prepare optional outputs
    if args.save_images or args.save_npy:
        img_dir = out_dir / "images"
        lat_dir = out_dir / "latents"
        img_dir.mkdir(exist_ok=True)
        lat_dir.mkdir(exist_ok=True)

    def _prompt_for(job):
        return prompt_for(job["phase"], job["tags"])

    if manual_jobs is not None:
        print(f"Generating {n_total} manual tag-tuple samples...")
    else:
        print(f"Generating {n_total} samples ({per_phase} per phase)...")
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
            prompts_list.append(prompts[pos])
            tags_json_list.append(json.dumps(j["tags"], sort_keys=True))
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
            "prompt": pa.array(prompts_list, type=pa.string()),
            "tags_json": pa.array(tags_json_list, type=pa.string()),
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
        "tag_profile": "core5+phase2",
        "tag_tuples_json": str(args.tag_tuples_json) if args.tag_tuples_json else None,
        "crops_dir": str(args.crops_dir),
        "tags_dir": str(args.tags_dir),
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
