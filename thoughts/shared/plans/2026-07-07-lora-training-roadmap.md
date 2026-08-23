# LoRA Training Roadmap — SD1.5 → SD2.x → SD3.x

## Overview

Progressive multi-phase ablation study for LoRA fine-tuning of Stable Diffusion models
on *Allium cepa* mitotic-phase crops. The goal is a synthetic data generator that produces
plausible micrograph-style images of each mitotic phase (prophase, metaphase, anaphase,
telophase) for downstream dataset augmentation.

**Carry-forward principle**: each phase fixes the winner from all previous phases and
varies only one group of hyperparameters. This keeps conclusions interpretable and avoids
a combinatorial explosion.

**Evaluation signal**: primary metric is `final_loss` from the kohya TensorBoard logs
(read by `scripts/evaluate_lora.py`). Secondary metric is visual quality of the 4×3
sample grid produced by `scripts/generate_lora_samples.py`. When both disagree, prefer
visual quality — loss is a proxy, not the goal.

**Winner tracking**: at the end of each phase, record the winning experiment name in the
`## Results` box of that phase section. Carry it forward as the `base` for the next phase.

---

## Infrastructure Prerequisites

Some phases require code or dataset versions that do not yet exist. Build these before
starting the corresponding phase.

### Before Phase 1
- [ ] Add `--copies N` flag to `scripts/utils/lora_dataset.py` (default 1, 0 = originals
  only) to control the number of augmented copies per original image.
- [ ] Add `--aug-strength {mild,heavy}` flag (or equivalent) to vary augmentation
  intensity: heavy uses ±45° rotation, wider color jitter, `RandomPerspective`.
- [ ] Build and verify the named dataset versions via DVC:
  ```bash
  dvc repro prepare_lora_dataset@no_aug      # --copies 0
  dvc repro prepare_lora_dataset@baseline    # --copies 1, mild (already done)
  dvc repro prepare_lora_dataset@aug2x       # --copies 2, mild
  dvc repro prepare_lora_dataset@heavy_aug   # --copies 1, heavy
  ```
- [ ] Update `dvc.yaml` `prepare_lora_dataset` foreach matrix to
  `[no_aug, baseline, aug2x, heavy_aug]` with the correct per-item flags.

### Before Phase 2
- [ ] Add `noise_offset`, `min_snr_gamma`, `timestep_sampling`, `ip_noise_gamma` fields
  to `LoRATrainingConfig` in `src/allium_cepa_classifier/config/lora_config.py`.
- [ ] Wire those fields into the `build_cmd` arg list in `scripts/train_lora.py`
  (conditional — emit flag only when non-default).

### Before Phase 3
- [ ] Add `lr_warmup_ratio` field to `LoRATrainingConfig` (alternative to
  `lr_warmup_steps` — kohya accepts both; ratio is more portable across step counts).
- [ ] Verify `optimizer_type: Prodigy` installs correctly (`prodigyopt` package).

### Before Phase 5 (SD2.x)
- [ ] Create `experiments/lora/sd2_rank<N>/config.yaml` configs with
  `model.v2: true`, `model.v_parameterization: true`, `model.resolution: 768`.
- [ ] Confirm the SD2.x checkpoint path / HF model id to use.

### Before Phase 6 (SD3.x)
- [ ] Verify `sd3_train_network.py` args for the model checkpoint and text encoders.
- [ ] Create `data.dataset_config` TOML files (SD3 requires this instead of
  `train_data_dir`) — see `scripts/vendor/sd-scripts/docs/sd3_train_network.md`.
- [ ] Create `experiments/lora/sd3_rank<N>/config.yaml` configs with
  `model.model_family: sd3`.

### Before Phase 7
- [ ] Implement additional `evaluate_lora.py` metrics (register via `@register`):
  - `fid` — FID between generated set and real crops per phase (evaluation only, no
    training change).
  - `classifier_judge` — run YOLO + EfficientNet on generated crops; report detection
    rate and phase classification accuracy.

---

## Phase 1 — SD1.5: Data × Architecture

**Question**: which dataset version and LoRA rank produce the best loss and visual quality?

**Fixed**: SD1.5 (`stable-diffusion-v1-5`), epsilon loss, AdamW8bit, cosine scheduler,
no noise tricks, `caption_dropout_rate=0`, standard `networks.lora` module.

**Varies**: `data.dataset_version` × `network.network_dim`

| experiment | dataset_version | network_dim | network_alpha |
|---|---|---|---|
| no_aug_r16 | no_aug | 16 | 8 |
| baseline_r16 | baseline | 16 | 8 |
| aug2x_r16 | aug2x | 16 | 8 |
| heavy_aug_r16 | heavy_aug | 16 | 8 |
| no_aug_r32 | no_aug | 32 | 16 |
| baseline_r32 | baseline | 32 | 16 |
| aug2x_r32 | aug2x | 32 | 16 |
| heavy_aug_r32 | heavy_aug | 32 | 16 |
| baseline_r64 | baseline | 64 | 32 |
| baseline_r128 | baseline | 128 | 64 |

> **Pruning note**: run all 4 dataset variants at rank 16 and 32. Promote only the
> best-performing dataset version to rank 64 and 128 — no need for a full 4×4 grid.

**Run**:
```bash
uv run python scripts/sweep_lora.py --configs experiments/lora/no_aug_r16/config.yaml \
  experiments/lora/baseline_r16/config.yaml experiments/lora/aug2x_r16/config.yaml \
  experiments/lora/heavy_aug_r16/config.yaml \
  experiments/lora/no_aug_r32/config.yaml experiments/lora/baseline_r32/config.yaml \
  experiments/lora/aug2x_r32/config.yaml experiments/lora/heavy_aug_r32/config.yaml \
  experiments/lora/baseline_r64/config.yaml experiments/lora/baseline_r128/config.yaml
tensorboard --logdir experiments/lora/_sweeps/<timestamp>
```

**Success criteria**:
- [ ] All experiments complete without error.
- [ ] HParams tab in TensorBoard shows all rows populated with `hparam/final_loss`.
- [ ] Visual inspection of `plots/lora_samples.png` for each experiment.
- [ ] Winner recorded below.

### Results
```
Winner dataset_version : _______________
Winner network_dim     : _______________
Winner experiment      : _______________   (carry forward as base for Phase 2)
```

---

## Phase 2 — SD1.5: Noise & Loss Shaping

**Question**: do standard SD1.5 training tricks improve quality, and which combination
is best?

**Fixed**: winner `(dataset_version*, network_dim*)` from Phase 1, all other settings
at Phase 1 defaults.

**Varies**: `noise_offset`, `min_snr_gamma`, `timestep_sampling`, `ip_noise_gamma`

| experiment | noise_offset | min_snr_gamma | timestep_sampling | ip_noise_gamma | notes |
|---|---|---|---|---|---|
| tricks_none | 0 | — | uniform | 0 | Phase 1 baseline |
| noise_off | 0.05 | — | uniform | 0 | offset only |
| min_snr | 0 | 5 | uniform | 0 | SNR weighting only |
| noise_min_snr | 0.05 | 5 | uniform | 0 | canonical combo |
| logit_normal | 0 | — | logit_normal | 0 | timestep reweighting |
| full_tricks | 0.05 | 5 | logit_normal | 0 | all three |
| ip_noise | 0.05 | 5 | logit_normal | 0.1 | add input perturbation |

> **Note**: `noise_offset` and `ip_noise_gamma` are both noise injection techniques.
> High values of both simultaneously can destabilize training — start with `ip_noise_gamma
> ≤ 0.1` if combining.

> **Note**: `noise_offset` is a UNet-specific patch for SD1.x; it may not apply or may
> need different tuning on SD2.x v-prediction models. Its effectiveness here informs
> whether to carry it into Phase 5.

**Success criteria**:
- [ ] All 7 experiments complete.
- [ ] `tricks_none` loss matches (±5%) the corresponding Phase 1 winner — confirms
  reproducibility.
- [ ] Winner recorded below.

### Results
```
Winner tricks config   : _______________
noise_offset           : _______________
min_snr_gamma          : _______________
timestep_sampling      : _______________
ip_noise_gamma         : _______________
```

---

## Phase 3 — SD1.5: Optimizer & Scheduler

**Question**: does a different optimizer or LR schedule improve convergence?

**Fixed**: `(dataset_version*, network_dim*, tricks*)` from Phases 1–2.

**Varies**: `optimizer_type`, `lr_scheduler`, `lr_warmup_ratio`

| experiment | optimizer_type | lr_scheduler | lr_warmup_ratio | notes |
|---|---|---|---|---|
| sched_baseline | AdamW8bit | cosine | 0.05 | Phase 1–2 baseline |
| cosine_restarts | AdamW8bit | cosine_with_restarts | 0.05 | cycle restarts |
| polynomial | AdamW8bit | polynomial | 0.10 | decaying LR |
| lion | Lion | cosine_with_restarts | 0.05 | sign-based optimizer |
| prodigy | Prodigy | constant | 0 | self-tuning LR |

> **Note**: `Prodigy` requires `prodigyopt` package. When using Prodigy, set
> `learning_rate: 1.0` (Prodigy interprets it as a multiplier, not a fixed LR).
> If it ties with `cosine_restarts`, prefer `AdamW8bit` for simplicity.

**Success criteria**:
- [ ] All 5 experiments complete.
- [ ] `sched_baseline` loss within ±5% of Phase 2 winner — confirms no regression.
- [ ] Winner recorded below.

### Results
```
Winner optimizer       : _______________
Winner scheduler       : _______________
lr_warmup_ratio        : _______________
```

---

## Phase 4 — SD1.5: Conditioning & Network Module

**Question**: does caption dropout improve inference-time CFG response? Does LoHa or LoKr
outperform standard LoRA at the same rank?

**Fixed**: full winner config from Phases 1–3.

**Varies**: `caption_dropout_rate` (first), then `network_module` at the best dropout.

| experiment | caption_dropout_rate | network_module | notes |
|---|---|---|---|
| cond_baseline | 0 | lora | Phase 1–3 baseline |
| dropout_5 | 0.05 | lora | mild dropout |
| dropout_10 | 0.10 | lora | moderate dropout |
| loha | dropout* | loha | Hadamard product adapter |
| lokr | dropout* | lokr | Kronecker adapter |

> **Note**: LoHa (`networks.loha`) and LoKr (`networks.lokr`) are more parameter-expressive
> than standard LoRA at the same rank — they may outperform especially at lower ranks
> (16, 32). Run them at the winning `dropout*` value from the first three rows.

> **Note**: verify that `networks.loha` and `networks.lokr` are available in the pinned
> kohya `sd3`-branch commit before scheduling these experiments.

**This phase completes the SD1.5 optimal config `C*`.**

**Success criteria**:
- [ ] All 5 experiments complete.
- [ ] Winner recorded below.
- [ ] `C*` is fully documented (all hyperparameters) and can be reproduced from a single
  config YAML.

### Results
```
Caption dropout_rate   : _______________
Network module         : _______________
Winner experiment      : _______________   (= C*, the optimal SD1.5 config)

C* full hyperparameters:
  dataset_version      : _______________
  network_dim          : _______________
  network_module       : _______________
  noise_offset         : _______________
  min_snr_gamma        : _______________
  timestep_sampling    : _______________
  optimizer_type       : _______________
  lr_scheduler         : _______________
  lr_warmup_ratio      : _______________
  caption_dropout_rate : _______________
```

---

## Phase 5 — SD2.x: Port & Validate

**Question**: does the optimal SD1.5 config `C*` transfer to SD2.x, or do key
hyperparameters (rank, noise tricks) shift under the v-prediction objective?

**Context**: SD2.x uses v-prediction natively at 768px. Key differences from SD1.5:
- `v_parameterization: true` is required.
- `noise_offset` is a patch for SD1.x epsilon models; its benefit may disappear or
  require re-tuning on v-pred.
- Higher resolution (768) increases VRAM usage — may require smaller batch or
  `gradient_accumulation_steps`.
- SD2.x is more sensitive to LR; the optimal `learning_rate` from `C*` may need
  scaling down.

**Fixed**: everything from `C*` except model settings.

**Varies**: base model, then re-validate rank and noise tricks.

| experiment | base_model | network_dim | noise_offset | min_snr_gamma | notes |
|---|---|---|---|---|---|
| sd2_port | SD2.1 | dim* | offset* | snr* | direct port of C* |
| sd2_rank_down | SD2.1 | dim*/2 | offset* | snr* | lower capacity |
| sd2_rank_up | SD2.1 | dim*×2 | offset* | snr* | higher capacity |
| sd2_no_offset | SD2.1 | dim* | 0 | snr* | offset without v-pred |
| sd2_no_snr | SD2.1 | dim* | offset* | — | SNR without v-pred |

> **SD2.1 model id**: confirm the exact HF model id before creating configs (e.g.
> `stabilityai/stable-diffusion-2-1` or a local checkpoint).

**Success criteria**:
- [ ] `sd2_port` trains without error with `v2: true, v_parameterization: true`.
- [ ] Loss curve shape is comparable to SD1.5 `C*` (expected to be higher in absolute
  terms due to different scale, but should converge).
- [ ] Visual quality of samples at least matches SD1.5 `C*` (primary criterion).
- [ ] Winner recorded below.

### Results
```
Optimal SD2.x rank     : _______________
noise_offset useful?   : _______________   (yes / no / different value)
min_snr_gamma useful?  : _______________
Winner experiment      : _______________   (= C2*)
```

---

## Phase 6 — SD3.x: Port & Validate

**Question**: does SD3.x produce meaningfully better image quality for this domain, and
do the training choices from `C2*` transfer?

**Context**: SD3/SD3.5 uses a flow-matching objective (MMDiT architecture). Key
differences:
- `loss_type` and `noise_offset` do not apply — flow-matching has its own formulation.
- `timestep_sampling: logit_normal` is the standard for flow matching (may already be
  default in kohya's SD3 entrypoint).
- Triple text encoders (CLIP-L + CLIP-G + T5-XXL) — T5 responds strongly to keyword
  density, so **caption quality** becomes more important than in SD1.5.
- Rank behavior in the MMDiT transformer differs from UNet — re-validating rank is
  essential.
- Requires `data.dataset_config` (TOML) instead of `train_data_dir` — different data
  pipeline setup.
- Compute cost is significantly higher than SD1.5/SD2.x.

**Fixed**: winning dataset version and conditioning settings from `C2*`, minus
SD1.x/SD2.x-specific tricks.

**Varies**: rank, timestep sampling strategy, caption style.

| experiment | model | network_dim | timestep_sampling | caption_style | notes |
|---|---|---|---|---|---|
| sd3_baseline | SD3.5M | dim* | logit_normal | minimal | baseline port |
| sd3_rank_down | SD3.5M | dim*/2 | logit_normal | minimal | MMDiT may need less |
| sd3_rank_up | SD3.5M | dim*×2 | logit_normal | minimal | or more |
| sd3_sigmoid | SD3.5M | dim* | sigmoid | minimal | alternative sampling |
| sd3_rich_captions | SD3.5M | rank* | logit_normal | descriptive | T5 keyword density |

> **Rich caption format** for `sd3_rich_captions`:
> ```
> micrograph of allium cepa onion root tip cell in {phase}, condensed chromosomes,
> mitosis, light microscopy, biological cell, high detail
> ```
> Both the training captions (in `.txt` files) and inference prompts must use the same
> format — update `generate_lora_samples.py` to read the caption template from config
> when this variant is used.

> **`dim*`** in the table refers to the winning rank from Phase 6's own sub-sweep
> (sd3_baseline, sd3_rank_down, sd3_rank_up), not necessarily the SD1.5 winner.

**Success criteria**:
- [ ] `sd3_baseline` trains without error using `sd3_train_network.py`.
- [ ] Loss converges (not diverges) within the first 10% of training.
- [ ] Visual quality comparison: does SD3.x produce noticeably better samples than
  SD2.x `C2*`? Document the verdict.
- [ ] Winner recorded below.

### Results
```
Optimal SD3.x rank     : _______________
Timestep sampling      : _______________
Rich captions better?  : _______________   (yes / no / marginal)
Winner experiment      : _______________   (= C3*)
Quality vs SD2.x       : _______________   (better / similar / worse — justify)
```

---

## Phase 7 — Custom / Conceptual Losses

**Question**: can task-specific evaluation signals guide training toward more biologically
plausible outputs?

**Base**: `C3*` (or `C2*` if SD3.x did not improve quality).

These experiments require modifying the training loop beyond standard kohya flags, or
adding post-generation evaluation metrics. Start with evaluation-only metrics (no
training change) to establish a baseline before committing to training-loop modifications.

| experiment | mechanism | training change | description |
|---|---|---|---|
| fid_baseline | evaluation only | none | FID between 100 generated crops and real crops per phase; establishes quality floor |
| classifier_judge_eval | evaluation only | none | YOLO + EfficientNet on 100 generated crops; report detection rate and phase accuracy |
| perceptual_loss | training modification | add LPIPS term to latent loss | penalize perceptual distance from real crops during denoising steps |
| phase_contrastive | training modification | contrastive objective | pull same-phase latents together, push cross-phase latents apart |

> **Implementation order**: run `fid_baseline` and `classifier_judge_eval` first (pure
> evaluation, zero training cost). If the classifier judge shows < 70% phase accuracy on
> generated images, this signals that the model is generating visually valid cells but
> not phase-specific ones — a contrastive or guide loss would help.

> **FID and classifier_judge** can be added to `scripts/evaluate_lora.py` via the
> `@register` decorator without touching `sweep_lora.py` — a good first step.

**Success criteria**:
- [ ] FID baseline established for `C3*` samples (one number per phase + overall).
- [ ] Classifier judge accuracy established (detection rate + phase accuracy).
- [ ] If training-loop modifications are pursued: loss curve remains stable (no
  divergence), and FID / classifier accuracy improve over the `fid_baseline`.
- [ ] Conclusions recorded below.

### Results
```
FID (C3* baseline)         : _______________   (per-phase breakdown in metrics.json)
Classifier judge accuracy  : _______________
Best custom loss experiment: _______________
Improvement over baseline  : _______________
```

---

## Carry-Forward Chain Summary

```
Phase 1: (dataset_version*, network_dim*)
    ↓
Phase 2: + (noise_offset*, min_snr_gamma*, timestep_sampling*)
    ↓
Phase 3: + (optimizer*, lr_scheduler*)
    ↓
Phase 4: + (caption_dropout*, network_module*)  →  C*  (optimal SD1.5)
    ↓
Phase 5: port C* to SD2.x, re-validate rank + noise tricks  →  C2*
    ↓
Phase 6: port to SD3.x, re-validate rank + captions  →  C3*
    ↓
Phase 7: custom evaluation + optional training modifications on C3*
```

**Cross-phase risk**: the network module chosen in Phase 4 (LoHa, LoKr) may not be
equally supported in the kohya SD3 entrypoint. Verify before committing to a non-standard
module for Phase 6 configs. If unsupported, fall back to `networks.lora` for SD3.x only.

---

## References

- LoRA kohya implementation plan: [2026-06-21-lora-finetuning-kohya.md](2026-06-21-lora-finetuning-kohya.md)
- LoRA testing suite (sweep infrastructure): [2026-06-22-lora-testing-suite.md](2026-06-22-lora-testing-suite.md)
- Dataset builder: [scripts/utils/lora_dataset.py](../../../scripts/utils/lora_dataset.py)
- Config model: [src/allium_cepa_classifier/config/lora_config.py](../../../src/allium_cepa_classifier/config/lora_config.py)
- Training wrapper: [scripts/train_lora.py](../../../scripts/train_lora.py)
- Sweep driver: [scripts/sweep_lora.py](../../../scripts/sweep_lora.py)
- Evaluator registry: [scripts/evaluate_lora.py](../../../scripts/evaluate_lora.py)
- Experiment configs: [experiments/lora/](../../../experiments/lora/)
- kohya sd3 branch: `scripts/vendor/sd-scripts/` (submodule, sd3 branch)
- kohya SD3 docs: `scripts/vendor/sd-scripts/docs/sd3_train_network.md`
