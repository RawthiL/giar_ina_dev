"""
Generate a same-cell cell-division sequence with the existing tagcue LoRA.

The script keeps the cell identity from one anchor image and changes only the phase
prompt. Three modes are available:

  seed    - text2img for every state with the same seed (weak identity baseline)
  anchor  - one text2img anchor; every state is an img2img edit of that anchor
  chain   - text2img for the first state; each next state is img2img from the previous

The requested states come from ``--tag-tuples-json``::

    [
      {"phase": "interphase", "tags": {...}},
      {"phase": "prophase", "tags": {...}, "strength": 0.35},
      {"phase": "metaphase", "tags": {...}},
      {"phase": "anaphase", "tags": {...}},
      {"phase": "telophase", "tags": {...}}
    ]

Tags are validated with ``vlm_tags.prompt_for``: unknown fields, values outside the
fixed dictionary, and phase-specific fields used with the wrong phase raise an error.
Core cell cues must be identical across states (unless ``--allow-common-mismatch``),
and the anaphase/telophase division axis is kept consistent.

Smooth animation is produced by interpolating between keyframes either in VAE latent
space (default) or by crossfading. No model training is required.

Usage:
    uv run python scripts/generate_division_sequence.py \
      --config experiments/lora/vae256_tagcue/config.yaml \
      --vae experiments/sd_vae_adversarial/vae_256_tagcue/weights \
      --tag-tuples-json /tmp/division_sequence.json \
      --mode anchor --strength 0.4 --steps 30 --guidance-scale 6.0 \
      --seed 1234 --frames-per-transition 8 --fps 8 \
      --out /tmp/division_sequence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from name_types import PHASES  # noqa: E402
from vlm_tags import (  # noqa: E402
    CORE_TAG_FIELDS,
    TagValidationError,
    prompt_for,
)

NEGATIVE_PROMPT = "blurry, low quality, deformed, malformed, text, watermark, jpeg artifacts"
STATE_ORDER = ["interphase", *PHASES[:-1]]


def _load_states(path: Path, default_strength: float, allow_common_mismatch: bool) -> list[dict]:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read --tag-tuples-json: {exc}") from exc
    if not isinstance(raw, list) or len(raw) < 2:
        raise SystemExit("--tag-tuples-json must contain a list with at least two states")

    states: list[dict] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict) or "phase" not in item or "tags" not in item:
            raise SystemExit(f"Invalid state #{idx}: expected 'phase' and 'tags' keys")
        phase = item["phase"]
        tags = dict(item["tags"])
        try:
            prompt_for(phase, tags)
        except TagValidationError as exc:
            raise SystemExit(f"Invalid state #{idx}: {exc}") from exc
        strength = float(item.get("strength", default_strength))
        if not 0.0 < strength <= 1.0:
            raise SystemExit(f"Invalid strength for state #{idx}: {strength}")
        states.append({"phase": phase, "tags": tags, "strength": strength})

    phases = [state["phase"] for state in states]
    if len(set(phases)) != len(phases):
        raise SystemExit(f"Duplicate phases are not allowed: {phases}")
    states.sort(key=lambda state: STATE_ORDER.index(state["phase"]))

    by_phase = {state["phase"]: state for state in states}
    if "anaphase" in by_phase and "telophase" in by_phase:
        axes = {
            phase: by_phase[phase]["tags"].get("axis_of_division")
            for phase in ("anaphase", "telophase")
        }
        specified = {axis for axis in axes.values() if axis}
        if len(specified) > 1:
            raise SystemExit(f"Anaphase/telophase axis mismatch: {axes}")
        if len(specified) == 1:
            axis = next(iter(specified))
            for phase in ("anaphase", "telophase"):
                if not by_phase[phase]["tags"].get("axis_of_division"):
                    by_phase[phase]["tags"]["axis_of_division"] = axis
                    print(f"Warning: copied axis_of_division={axis!r} to {phase}")

    for field in CORE_TAG_FIELDS:
        values = {state["tags"].get(field) for state in states}
        if len(values) > 1:
            message = (
                f"Core tag {field!r} differs across states: {sorted(map(str, values))}. "
                "Same-cell sequences should keep core cues identical."
            )
            if allow_common_mismatch:
                print(f"Warning: {message}")
            else:
                raise SystemExit(message)

    for state in states:
        prompt_for(state["phase"], state["tags"])
    return states


def _load_pipelines(cfg: LoRAExperimentConfig, lora_path: Path, vae_path: Path | None, device: str):
    if cfg.model.model_family != "sd15":
        raise SystemExit(
            f"generate_division_sequence currently supports sd15 only, got "
            f"{cfg.model.model_family!r}"
        )
    from diffusers import (
        AutoencoderKL,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
    )

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    kwargs = {"torch_dtype": dtype, "safety_checker": None}
    if vae_path is not None:
        kwargs["vae"] = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=dtype)
    text_pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path, **kwargs
    ).to(device)
    text_pipe.load_lora_weights(str(lora_path))
    img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(text_pipe).to(device)
    return text_pipe, img_pipe


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _text2img(pipe, prompt: str, seed: int, steps: int, guidance: float, resolution: int):
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=resolution,
        width=resolution,
        generator=_make_generator(seed),
    ).images[0]


def _img2img(
    pipe,
    prompt: str,
    image: Image.Image,
    strength: float,
    seed: int,
    steps: int,
    guidance: float,
):
    return pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=_make_generator(seed),
    ).images[0]


def _encode(pipe, image: Image.Image) -> torch.Tensor:
    """Encode a generated keyframe to the VAE's deterministic latent mean.

    The image is converted to the [-1, 1] range expected by the VAE, then encoded
    with the frozen encoder.  ``latent_dist.mean`` is used instead of a random
    sample so interpolation is reproducible.
    """
    vae = pipe.vae
    x = pipe.image_processor.preprocess(image).to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        return vae.encode(x).latent_dist.mean


def _decode(pipe, latent: torch.Tensor) -> Image.Image:
    """Decode an interpolated VAE latent back to a PIL image."""
    vae = pipe.vae
    with torch.no_grad():
        image = vae.decode(latent.to(vae.device, dtype=vae.dtype)).sample
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _interpolate(
    pipe,
    keyframes: list[Image.Image],
    frames_per_transition: int,
    mode: str,
) -> list[Image.Image]:
    """Create smooth intermediate frames between adjacent phase keyframes.

    Keyframes are generated first (seed/text2img, anchor/img2img, or chain/img2img).
    This function does not create new phase states; it only fills the gaps between
    existing keyframes.

    Modes:
      - ``latent`` (default): encode each keyframe with the VAE, linearly interpolate
        the latent vectors, and decode the interpolated latents.  This usually avoids
        the double-exposure ghosting of pixel blending, though decoder artifacts are
        still possible.
      - ``crossfade``: blend the two keyframe pixels directly with ``Image.blend``.
      - ``none``: return only the keyframes.
    """
    if frames_per_transition <= 0 or mode == "none" or len(keyframes) < 2:
        return list(keyframes)

    frames = [keyframes[0]]
    if mode == "crossfade":
        # Pixel-space interpolation: cheap, but can look like a double exposure when
        # the chromosome arrangement changes substantially.
        for idx in range(len(keyframes) - 1):
            for step in range(1, frames_per_transition + 1):
                alpha = step / (frames_per_transition + 1)
                frames.append(Image.blend(keyframes[idx], keyframes[idx + 1], alpha))
            frames.append(keyframes[idx + 1])
        return frames

    # Latent-space interpolation: z(alpha) = (1-alpha) z0 + alpha z1.  We skip the
    # endpoints here because they are the original keyframes, which are appended
    # separately.
    latents = [_encode(pipe, image).float() for image in keyframes]
    for idx in range(len(latents) - 1):
        z0, z1 = latents[idx], latents[idx + 1]
        for step in range(1, frames_per_transition + 1):
            alpha = step / (frames_per_transition + 1)
            frames.append(_decode(pipe, (1.0 - alpha) * z0 + alpha * z1))
        frames.append(keyframes[idx + 1])
    return frames


def _save_keyframe_grid(states: list[dict], keyframes: list[Image.Image], path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(states), figsize=(4 * len(states), 4), squeeze=False, constrained_layout=True
    )
    for col, (state, image) in enumerate(zip(states, keyframes, strict=True)):
        axes[0, col].imshow(image)
        axes[0, col].axis("off")
        axes[0, col].set_title(state["phase"], fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lora", type=Path, default=None, help="Override LoRA safetensors path")
    parser.add_argument("--vae", type=Path, default=None, help="Override config VAE path")
    parser.add_argument("--tag-tuples-json", required=True, type=Path)
    parser.add_argument("--mode", choices=["seed", "anchor", "chain"], default="anchor")
    parser.add_argument(
        "--anchor-phase",
        default="interphase",
        choices=PHASES,
        help="State used for the text2img anchor in --mode anchor (default: interphase).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.4,
        help="Default img2img strength; per-state 'strength' in the JSON overrides it.",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--interpolation", choices=["latent", "crossfade", "none"], default="latent"
    )
    parser.add_argument("--frames-per-transition", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--allow-common-mismatch", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.frames_per_transition < 0:
        raise SystemExit("--frames-per-transition must be >= 0")
    if args.fps < 1:
        raise SystemExit("--fps must be >= 1")

    states = _load_states(args.tag_tuples_json, args.strength, args.allow_common_mismatch)

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    lora_path = args.lora or (run_dir / "weights" / f"{cfg.experiment_name}.safetensors")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}. Run training first.")
    vae_path = args.vae if args.vae is not None else cfg.model.vae

    out_dir = Path(args.out)
    keyframe_dir = out_dir / "keyframes"
    frame_dir = out_dir / "frames"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for directory in (keyframe_dir, frame_dir):
        for stale in directory.iterdir():
            if stale.is_file():
                stale.unlink()

    print(f"Loading {cfg.model.model_family} pipelines ...")
    text_pipe, img_pipe = _load_pipelines(cfg, lora_path, vae_path, args.device)
    resolution = cfg.model.resolution

    # Build one keyframe per requested phase.  These are the phase states; the
    # intermediate transition frames are created afterwards by _interpolate().
    keyframes: list[Image.Image] = []
    keyframe_paths: list[str] = []
    prompts = [prompt_for(state["phase"], state["tags"]) for state in states]
    anchor_phase = None
    anchor_image = None
    if args.mode == "anchor":
        # Generate one anchor cell, then edit it with img2img for every other phase.
        # This is the strongest identity-preserving mode because every state starts
        # from the same image.
        phases = [state["phase"] for state in states]
        anchor_phase = args.anchor_phase if args.anchor_phase in phases else phases[0]
        anchor_idx = phases.index(anchor_phase)
        anchor_image = _text2img(
            text_pipe,
            prompts[anchor_idx],
            args.seed,
            args.steps,
            args.guidance_scale,
            resolution,
        )
        print(f"Anchor: {anchor_phase}")
    elif args.mode == "chain":
        anchor_phase = states[0]["phase"]

    for idx, (state, prompt) in enumerate(zip(states, prompts, strict=True)):
        phase = state["phase"]
        if args.mode == "seed":
            image = _text2img(
                text_pipe, prompt, args.seed, args.steps, args.guidance_scale, resolution
            )
        elif args.mode == "anchor":
            if phase == anchor_phase:
                image = anchor_image
            else:
                image = _img2img(
                    img_pipe,
                    prompt,
                    anchor_image,
                    state["strength"],
                    args.seed,
                    args.steps,
                    args.guidance_scale,
                )
        else:
            if idx == 0:
                image = _text2img(
                    text_pipe, prompt, args.seed, args.steps, args.guidance_scale, resolution
                )
            else:
                image = _img2img(
                    img_pipe,
                    prompt,
                    keyframes[idx - 1],
                    state["strength"],
                    args.seed,
                    args.steps,
                    args.guidance_scale,
                )

        keyframes.append(image)
        keyframe_path = keyframe_dir / f"{idx:02d}_{phase}.png"
        image.save(keyframe_path)
        keyframe_paths.append(str(keyframe_path))
        print(f"Keyframe {idx + 1}/{len(states)}: {phase} -> {keyframe_path}")

    # Fill the gaps between phase keyframes (latent interpolation by default).
    frames = _interpolate(img_pipe, keyframes, args.frames_per_transition, args.interpolation)
    frame_paths: list[str] = []
    for idx, frame in enumerate(frames):
        frame_path = frame_dir / f"frame_{idx:04d}.png"
        frame.save(frame_path)
        frame_paths.append(str(frame_path))

    grid_path = out_dir / "keyframes_grid.png"
    _save_keyframe_grid(states, keyframes, grid_path)
    gif_path = out_dir / "animation.gif"
    if len(frames) >= 2:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(1000 / args.fps)),
            loop=0,
        )

    metadata = {
        "config": str(args.config),
        "lora": str(lora_path),
        "vae": str(vae_path) if vae_path is not None else None,
        "mode": args.mode,
        "anchor_phase": anchor_phase,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "default_strength": args.strength,
        "interpolation": args.interpolation,
        "frames_per_transition": args.frames_per_transition,
        "fps": args.fps,
        "states": [
            {
                "phase": state["phase"],
                "tags": state["tags"],
                "prompt": prompt,
                "strength": state["strength"],
                "keyframe": keyframe_path,
            }
            for state, prompt, keyframe_path in zip(states, prompts, keyframe_paths, strict=True)
        ],
        "keyframes_grid": str(grid_path),
        "gif": str(gif_path) if len(frames) >= 2 else None,
        "frames": frame_paths,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nKeyframes: {keyframe_dir}")
    print(f"Frames:    {frame_dir}")
    if len(frames) >= 2:
        print(f"Animation: {gif_path}")
    print(f"Metadata:  {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
