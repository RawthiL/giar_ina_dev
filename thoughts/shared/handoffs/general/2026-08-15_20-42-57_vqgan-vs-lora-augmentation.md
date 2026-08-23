---
date: 2026-08-15T20:42:57-03:00
researcher: Nicolas
git_commit: 4a1302e766b056006d4dab6ce9331652ae5f35b5
branch: lora-fine-tune
repository: giar_ina_dev
topic: "VQGAN vs SD-VAE vs LoRA as synthetic-data sources for the phase classifier"
tags: [implementation, strategy, vqgan, lora, synthetic-data, phase-classifier, lora-metrics]
status: complete
last_updated: 2026-08-15
last_updated_by: Nicolas
type: implementation_strategy
---

# Handoff: VQGAN vs LoRA as augmentation sources + two silent bug fixes

## Task(s)

Session started as "repeat the VAE latent walk using the VQGAN weights" and became an
evaluation of which generator should supply synthetic mitotic-phase crops. Phase 4 of
`thoughts/shared/plans/2026-08-09-phase-conditional-lora-hpo.md` is the destination.

1. **VQGAN latent walk** — COMPLETE. Works, but only via a specific recipe (see Learnings).
2. **Diagnose VQGAN codebook collapse** — COMPLETE. Total collapse; not worth fixing.
3. **Compare LoRA vs SD-VAE vs VQGAN as augmentation sources** — COMPLETE, measured.
4. **Fix rotation-fill bug in LoRA training data** — COMPLETE, shipped + tested.
5. **Add judge filtering to downstream validator** — COMPLETE, shipped + measured.
6. **Retrain LoRA on fixed dataset** — COMPLETE. `p3_per_phase_nofill`, phase_consistency
   0.752 -> 0.838.
7. **Fix VQGAN `[0,1]` normalisation bug** — COMPLETE, shipped + tested. **A re-evaluation of
   both LoRA runs was still IN FLIGHT when this handoff was written** (see Action Items).
8. **Phase 4 downstream validation** — NOT STARTED. Blocked on the user choosing arms/ratios;
   it is several hours of GPU.

## Critical References

- `CLAUDE.md` — the "Four rules that are easy to violate silently" section. Rules 1 (never rank
  by kohya loss), 2 (pin the judge), 3 (KID scale), 4 (evaluator != judge architecture) all bit
  or nearly bit during this session. Updated this session.
- `thoughts/shared/plans/2026-08-09-phase-conditional-lora-hpo.md` — the plan this work serves.

## Recent changes

- `scripts/utils/lora_dataset.py:74-93` — new `rotate_without_fill()`; `augment_mild` and
  `augment_heavy` now call it instead of `image.rotate(..., fillcolor=128)`.
- `scripts/utils/lora_dataset.py:70-73` — registered `per_phase_nofill` version.
- `dvc.yaml:55` — added `per_phase_nofill` to the `prepare_lora_dataset` foreach.
- `scripts/utils/lora_metrics.py:239` — `_batch` now normalises to `[0,1]`, was `/127.5 - 1.0`.
- `scripts/utils/lora_metrics.py:204-230` — `VQGANFeatures` docstring records the range fix and
  its blast radius.
- `scripts/evaluate_lora.py:235-252` — `vqgan_recon` docstring rewritten; the ratio's meaning
  inverted after the fix (see Learnings).
- `scripts/validate_synthetic_downstream.py:68-101` — new `filter_by_judge()`.
- `scripts/validate_synthetic_downstream.py:~186` — new `--judge-filter MIN_CONF` flag, applied
  right after `ensure_phase_samples`.
- `tests/test_lora.py` — 2 new tests guarding the rotation fix (8 parametrised cases).
- `tests/test_lora_metrics.py` — 1 new test pinning the `[0,1]` input range.
- `CLAUDE.md` — rotation-fill trap, VQGAN input range + metrics comparability warning, rule 3
  corollary, `--judge-filter` usage example.
- `experiments/lora/p3_per_phase_nofill/config.yaml` — NEW experiment (copy of p3_per_phase,
  only `dataset_version` differs).

Full suite: **48 passed**. Lint clean except a pre-existing B904 in
`scripts/utils/push_weights_to_hf.py` (untouched, not mine).

## Learnings

### The VQGAN is a dead end as a generator — do not invest further

- **Codebook collapse is total**: 6 distinct codes used out of 16384, *corpus-wide* across 120
  test crops (not 6 per image — 6 total). Relative quantisation error `‖q−h‖/‖h‖ = 0.998`;
  encoder output norm ~15.6 vs codebook norm ~0.145 (~100x mismatch).
- **Cause** is visible in `vqgan/vqgan-training/*/hparams.yml`: trained from scratch
  (`pretrained_model_name_or_path: null`) at `learning_rate: 4.5e-06` for ~49,750 steps. That LR
  is the diffusers *fine-tuning* default — far too low for a codebook to organise from random
  init. The training script is no longer in the repo, only tfevents.
- **Consequence**: the classic VQGAN generative path (transformer prior over code indices) is
  permanently closed. It can only re-render functions of real latents.
- **Reconstruction ceiling is 21 dB** vs SD1.5 VAE's **34 dB** on the same crops. It also
  retains only **79.5% saturation and 68.9% contrast** — a *systematic* domain shift, not noise.
  Per-crop mean/std matching to the parent fixes much of this (KID 0.982 -> 0.702) but it is
  still dominated by SD-VAE on every axis.

### Latent walk recipe (if ever needed again)

- Must interpolate **post-quantisation** and decode with `force_not_quantize=True`. Feeding raw
  continuous latents gives garbage (MSE 1706 vs 0.002) because the decoder only ever saw
  codebook-scale inputs.
- Phase **centroids do not work** — the VQGAN latent is spatial `(4,64,64)` at 256px, so
  averaging unregistered crops smears to blobs. The 32-D VAE worked because its latent is a
  global vector. Use single-exemplar interpolation instead.
- A slightly better bypass: `h/‖h‖ * mean‖q‖` decoded with `force_not_quantize=True` (21.17 dB
  vs 20.92 quantised, visibly sharper).

### "Temperature" cannot work on SD-VAE or VQGAN

Measured on SD-VAE latents, judged by the pinned classifier: sigma=0.25 -> 0.938 acc but the
image is indistinguishable from the original; sigma=0.5 -> 0.625; sigma=1.0 -> 0.062. Decoding
`N(0,I)` gives pure noise. **There is no band where images become different-but-valid cells** —
they go straight from "identical" to "confetti". SD1.5's VAE has KL weight ~1e-6, so it is an
autoencoder, not a generative VAE. Only `experiments/vae/latent32_beta2` (real learnable prior)
and the LoRA can actually be sampled.

### Source comparison (100 samples/phase, pinned efficientnet_b2 judge)

| source | phase_cons | KID_cls | coverage | density | mem_excess |
|---|---|---|---|---|---|
| LoRA p3_per_phase | 0.752 | 1.255 | 0.333 | 0.531 | **-0.073** |
| SD-VAE mixup | 0.963 | 0.255 | 0.817 | 0.656 | **+0.040** |
| SD-VAE mixup+stain match | 0.960 | 0.209 | 0.870 | 0.811 | **+0.045** |
| VQGAN mixup | 0.812 | 0.982 | 0.423 | 0.314 | -0.061 |
| VQGAN mixup+stain match | 0.795 | 0.702 | 0.553 | 0.504 | -0.043 |

**Do not read the SD-VAE row as a win.** It sweeps the quality metrics because it is returning
near-copies of real crops (35% blend toward a nearest neighbour through a 34 dB codec).
`mem_excess > 0` means generations sit *closer* to training crops than real crops sit to each
other — the memorisation guard firing. The LoRA at -0.073 is the only genuinely novel source.
The quality metrics reward proximity to real data, so they cannot arbitrate this; only the
Phase 4 downstream test can.

### Root cause of the LoRA's visible border artifact

`image.rotate(angle, fillcolor=128)` on an **RGB** image fills with **(128, 0, 0) dark red** —
PIL expands a scalar into the first channel only, so the intended neutral grey never happened.
With `copies=1` exactly half of every LoRA dataset carried red corner wedges, and the model
learned the tilted red-cornered canvas as part of the concept. Verified directly; there is a
regression test pinning the old behaviour so the trap stays documented.

**Retraining with the fix (`p3_per_phase_nofill`) moved exactly the phases that mattered:**

| metric | p3_per_phase | nofill | delta |
|---|---|---|---|
| phase_consistency | 0.7525 | **0.8375** | +0.085 |
| prophase / metaphase | 0.89 / 0.93 | 0.94 / 0.97 | +0.05 / +0.04 |
| **anaphase** | 0.52 | **0.65** | **+0.13** |
| **telophase** | 0.67 | **0.79** | **+0.12** |
| kid_vqgan (corrected scale) | 2.644 | 2.072 | -0.572 |
| vqgan_recon_ratio (corrected) | 0.737 | **0.444** | **-0.293 (worse)** |
| coverage / density | 0.333 / 0.531 | 0.340 / 0.579 | +0.008 / +0.047 |
| kid_classifier | 1.255 | 1.273 | +0.018 (worse) |
| memorisation excess | -0.073 | -0.043 | narrowed, still safe |

Caveats: on n=100 at p~0.5 the 95% binomial interval is ~±0.10, so anaphase/telophase gains sit
just outside it individually — but all four phases moved the same direction. `kid_classifier`
did *not* improve. Anaphase still loses 33% of samples to metaphase (was 43%): the border
artifact was *a* cause, not *the* cause, and metaphase/anaphase is a genuine biological
continuum.

**IMPORTANT — the nofill run is MORE oversmoothed, not less.** On the corrected scale its
`vqgan_recon_ratio` is **0.444** against p3_per_phase's 0.737, where ~1.0 is the target. So
while phase conditioning improved a lot, texture fidelity relative to real crops got
*noticeably worse*. This corroborates the unresolved oversaturation/texture defect (Action Item
3) and means "nofill is strictly better" is **not** a safe conclusion — it is better at
conditioning, worse at texture. Whether that trade helps a downstream classifier is exactly
what Phase 4 must decide.

### The `[0,1]` fix inverted a metric's meaning

`VQGANFeatures._batch` fed `[-1,1]`; the checkpoint was trained on `[0,1]`. Real-crop recon MSE
0.349 -> 0.008. Critically **`vqgan_recon_ratio` flipped 1.22 -> 0.74**: generated crops are
*easier* to reconstruct than real ones, because real microscopy has grain/dust/focus noise the
generator smooths away. So the metric measures texture complexity vs real, and **~1 is the
target in both directions** (>1 artifacts, <1 oversmoothing). The generator's real weakness on
this axis is oversmoothing.

Blast radius, checked before editing: `_batch` also feeds `features()`, so `kid_vqgan`'s feature
space shifts (rule 3). **Safe because `optimize_lora.py:162` weights `kid_classifier`, not
`kid_vqgan`** — the Optuna objective does not move and `W_KID` needs no re-derivation.

### Judge filter keep rates (400 cached p3_per_phase samples)

| min conf | prophase | metaphase | anaphase | telophase | total |
|---|---|---|---|---|---|
| argmax only | 89% | 93% | 52% | 67% | 301/400 |
| 0.5 | 86% | 90% | 47% | 56% | 279/400 |
| 0.7 | 82% | 74% | 36% | 41% | 233/400 |
| 0.9 | 61% | 54% | **7%** | **8%** | 130/400 |

**The filter bites hardest where data is scarcest.** At 0.9 you keep 7 anaphase images, which
inverts the imbalance you are trying to fix. Use **0.5**, with 0.7 as an aggressive arm. Raise
`--samples` to compensate for the smaller surviving pool.

## Artifacts

Code / config (all committed to working tree, **not** git-committed):
- `scripts/utils/lora_dataset.py`
- `scripts/utils/lora_metrics.py`
- `scripts/evaluate_lora.py`
- `scripts/validate_synthetic_downstream.py`
- `dvc.yaml`
- `CLAUDE.md`
- `tests/test_lora.py`, `tests/test_lora_metrics.py`
- `experiments/lora/p3_per_phase_nofill/config.yaml` (new)

Data / results:
- `datasets/crops/lora/per_phase_nofill/` — rebuilt dataset (1194 orig + 1194 aug, 11562/epoch)
- `experiments/lora/p3_per_phase_nofill/weights/p3_per_phase_nofill.safetensors` (604 MB)
- `experiments/lora/p3_per_phase_nofill/metrics.json`
- `experiments/lora/p3_per_phase_nofill/plots/diagnostic/{phase}/` — 100 cached samples/phase
- Old-scale metrics backed up in the session scratchpad as `p3_metrics_oldscale.json` and
  `nofill_metrics_oldscale.json` (scratchpad is ephemeral — re-create from git if needed)

## Action Items & Next Steps

1. ~~Check the in-flight re-evaluation.~~ **DONE.** Both `metrics.json` files are now on the
   corrected `[0,1]` scale: `vqgan_recon_mse_real` = **0.016447** in both (the stale old-scale
   marker was 0.349327). Corrected values are in the table above. If you ever see 0.349327
   again, that file predates the fix.
2. **Decide and run Phase 4 downstream validation** — the only thing that answers whether any
   synthetic source helps. Proposed arms: real-only / real+classical-aug / real+LoRA-filtered /
   real+SD-VAE-mixup, several ratios, >=3 seeds, `--arch resnet50` (never efficientnet_b2).
   **A classical-augmentation arm is mandatory** — without it a positive synthetic result is
   uninterpretable, since rotation/flip/elastic/stain jitter is free and strong on
   rotationally-symmetric microscopy crops. Several hours of GPU.
3. **Oversaturated chromosomes remain unfixed.** Separate defect from the border artifact; the
   new samples still show garish magenta/blue. Suspect `augment_mild`'s 0.7-1.3
   brightness/contrast jitter combined with rank 256 over only 1194 unique images. Worth its own
   experiment.
4. **Nothing is git-committed.** 35 files show in `git status` (many pre-date this session).
   Review before committing.
5. **Optional**: `scripts/generate_vqgan_variations.py` was designed but never written (dual
   backend, judge filter, stain matching). Given the measured verdict I would skip it and use
   SD-VAE mixup directly if a mixup arm is wanted.

## Other Notes

- **`humanlayer` CLI is not installed on this machine**, so `humanlayer thoughts sync` could not
  be run. The handoff exists only at the path above. `scripts/spec_metadata.sh` also does not
  exist; metadata was gathered manually via git.
- `thoughts/shared/handoffs/` did not exist before this session — created it.
- Everything upstream of the VQGAN (`phase_consistency`, `kid_classifier`, `coverage`,
  `density`, `memorization`) comes from the phase judge and is **unaffected** by the `[0,1]`
  fix. The 0.752 -> 0.838 headline result stands regardless.
- Judge lives at `experiments/phase_classifier/efficientnet_b2`; loaded via
  `scripts/utils/phase_judge.py:75 load_phase_judge()`.
- Two monitor bugs worth avoiding: (a) `tail -f` on a kohya tqdm bar floods the event channel —
  poll on an interval instead; (b) `pgrep -f "<script>.*<name>"` matches the monitor's *own*
  shell because the pattern is in its command line, so completion is never detected. Match on
  something narrower or check for the output file instead.
- GPU is a single 24 GB card; the LoRA train (~1h15m) and eval (~15m) fully occupy it. Batch
  VQGAN work in chunks of 8 at 512px or it OOMs alongside other jobs.
