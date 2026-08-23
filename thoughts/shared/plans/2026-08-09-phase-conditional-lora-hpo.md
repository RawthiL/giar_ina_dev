# Phase-Conditional LoRA + Metric-Driven HPO

## Overview

Reorient the LoRA workstream from "generate plausible mitotic cells" to the actual project
goal: **generate phase-accurate synthetic crops to train a 4-class mitotic-phase classifier**
(prophase / metaphase / anaphase / telophase), where real data is scarce and imbalanced
relative to interphase.

This plan supersedes the evaluation strategy in
[2026-07-07-lora-training-roadmap.md](2026-07-07-lora-training-roadmap.md). The carry-forward
ablation structure there remains valid; what changes is the **evaluation signal** and the
**dataset layout**.

### Why the current approach needs correcting

Three findings from the state of the repo as of 2026-08-09:

1. **The current ranking metric is invalid.** The p2 sweep ranks experiments by `final_loss` /
   `avg_loss` from kohya's `loss/current`. But `min_snr_gamma` multiplies the loss by
   `min(SNR, γ)/SNR ≤ 1` — it changes the objective function, not just the model. The observed
   `p2_min_snr` avg_loss of 0.0747 vs `p2_baseline` 0.1148 is a −35% "improvement" that is
   largely a reweighting artifact. `ip_noise_gamma` has the same problem (it perturbs the
   target). **Loss is not comparable across configs that alter the loss.**
2. **FID is implemented but returns `null` in every run.** `datasets/crops/binary_classifier/`
   is not present locally and the classifier weights are `.dvc` pointers only, so
   `fid_n_real: 0` and `judge_n_images: 0`. A `dvc pull` unblocks both. Separately, n=52
   generated images against 2048 Inception features is a rank-deficient covariance — the
   estimate is meaningless (already noted in the `evaluate_lora.py` docstring).
3. **Phase conditioning is unverified and probably not working.** All four phases live in one
   kohya concept folder (`img/10_allium mitosis/`), distinguished only by a caption word.
   SD1.5's CLIP text encoder has no useful prior separating "anaphase" from "telophase", and
   `text_encoder_lr` is unset (inherits `learning_rate: 1e-5`, very low). The likely outcome is
   a single generic "mitotic cell" mode where the phase token does nothing.

Finding 3 is the crux: **if phase conditioning does not work, every metric and every HPO trial
optimizes the wrong thing.** It is tested first.

### Baseline facts

Unique original crops per phase (all splits, `_aug` copies excluded — the 959 `_aug` files
under `datasets/crops/vae/*/tagged/` are correctly filtered by `lora_dataset.py`; 1441 + 959 =
2400 checks out, no bug):

| phase | crops |
|---|---|
| prophase | 531 |
| metaphase | 334 |
| telophase | 308 |
| anaphase | 268 |
| **total** | **1441** |

Cost model: **1 training run ≈ 1h17m** (2000 steps, batch 1 × grad-accum 15, RTX 3090 24GB).
Budget: a few days, interruptible → assume **~60 GPU-hours**.

Determinism confirmed: `aug2x_r256` and `p2_baseline` are byte-identical configs and produced
identical metrics under seed 42. Single-seed A/B comparisons are reproducible (though still
single-seed).

### The VQGAN weights: verdict

`vqgan/vqgan weights/vqmodel/` is a diffusers **`VQModel`**, not an `AutoencoderKL`:

- `quantize.embedding.weight: [16384, 4]` — discrete codebook, not a continuous KL latent
- `block_out_channels: [128, 256, 512]` — 3 down blocks → **f=4**; SD1.5 is 4 blocks → **f=8**
- `norm_type: "spatial"` — LDM VQ-f4 convention, not SD's group norm
- `scaling_factor: 0.18215` is copied SD boilerplate and is **meaningless** for this model's
  latent statistics — do not read it as evidence the spaces are aligned

**It cannot be swapped into the SD1.5 pipeline.** Two independent blockers, either fatal alone:
the discrete codebook vs. continuous latent mismatch, and the spatial mismatch (at 512px this
encoder emits **128×128×4** where SD1.5's UNet expects **64×64×4** — it would not run, not merely
degrade).

It is still used in this plan, in two roles (Phase 2): as a **complementary KID feature space**
and as an **artifact detector**. The viable version of the original idea — fine-tuning SD1.5's
`AutoencoderKL` *decoder* on cell crops — is deferred to Phase 6 as a fidelity refinement,
because controllability, not reconstruction fidelity, is the current bottleneck.

---

## Phase 0 — Build the judge, run the diagnostic

**Goal:** produce the 4-class phase classifier (which serves three roles) and determine whether
phase conditioning works at all. **Cost:** ~half a day, minimal GPU.

- [ ] `dvc pull` — restores `datasets/crops/binary_classifier/` and the classifier/detector
      weights. This alone un-nulls `fid` and `classifier_judge` in `evaluate_lora.py`.
- [ ] Build a 4-class phase crop dataset at
      `datasets/crops/phase_classifier/{train,validation,test}/{prophase,metaphase,anaphase,telophase}/`
      from `datasets/crops/vae/*/tagged/`. Add as a DVC stage
      (`prepare_phase_crops`) mirroring `prepare_crops`.
      **Keep the existing split boundaries** — do not reshuffle, or the test set leaks.
- [ ] Train the phase classifier via the existing pipeline:
      ```bash
      uv run python scripts/train_classifier.py \
        --config experiments/phase_classifier/efficientnet_b2/config.yaml
      ```
      Reuse `ExperimentConfig`; only `num_classes` and the data path change. Calibration runs
      automatically. Record per-class F1 on the real test set — **this is the baseline the
      synthetic data must beat.**
- [ ] **Diagnostic:** load the existing `p2_full_tricks` LoRA, generate ~100 images per phase,
      run the phase classifier, print the 4×4 confusion matrix.

### Results box — run 2026-08-09

```
Dataset (datasets/crops/phase_classifier), no leakage of _aug copies:
  split        prophase  metaphase  anaphase  telophase   total
  train             708        444       356        410    1918   (959 orig + 959 _aug)
  validation         88         56        45         51     240   (originals only)
  test               89         56        45         52     242   (originals only)

Phase classifier baseline (efficientnet_b2, 14 epochs, early-stopped):
  test accuracy 0.9752   macro-F1 0.97
  per-class F1 — prophase 0.98  metaphase 0.99  anaphase 0.98  telophase 0.95
  ECE 0.0370 -> 0.0327 after vector scaling
  ** INFLATED — see the leakage finding below. Fine as a judge, invalid as a Phase 4 baseline. **

Phase-conditioning diagnostic on p2_full_tricks (100 samples/phase, judge = above):
  rows = prompted, cols = predicted
                anaphase   metaphase    prophase   telophase
  anaphase         51.0%       33.0%        5.0%       11.0%
  metaphase         9.0%       79.0%        8.0%        4.0%
  prophase          0.0%        0.0%       95.0%        5.0%
  telophase         2.0%        2.0%       39.0%       57.0%

  mean diagonal 70.5%  (chance 25%)   VERDICT: OK — conditioning works
```

**Gate result: PASSED.** The pre-plan hypothesis — that a single concept folder plus a caption
word would leave the phase token inert — was **wrong**. At 70.5% vs 25% chance, the phase token
demonstrably controls generation, and the four-separate-LoRAs fallback is not needed.

The residual error is structured, not random: it concentrates in **biologically adjacent,
visually similar pairs**, and tracks class scarcity.

| prompted | diagonal | leaks to | train crops |
|---|---|---|---|
| prophase | 95% | — | 531 |
| metaphase | 79% | prophase 8%, anaphase 9% | 334 |
| telophase | 57% | **prophase 39%** | 308 |
| anaphase | 51% | **metaphase 33%** | 268 |

The two weak phases are the two scarcest, and each collapses into its nearest neighbour
(telophase↔prophase share decondensed chromatin; anaphase↔metaphase differ only by chromatid
separation). This is precisely the failure mode the Phase 1 repeat-rebalancing targets, which
makes Phase 1 an improvement task rather than a rescue.

**Caveat on the numbers.** The judge has never been validated on generated images, so the matrix
conflates generator error with judge error under domain shift. It is reliable as a relative
signal but the per-phase percentages are not precise.

**SUPERSEDED — the 70.5% above was measured with the leaky-split judge.** After Phase 1 rebuilt
the judge on the clean split, the same p2_full_tricks images were re-scored
(`diagnose_phase_conditioning.py --reuse-existing`, no regeneration). The clean judge is stricter:

```
                anaphase   metaphase    prophase   telophase
anaphase           37.0%       49.0%        5.0%        9.0%
metaphase           4.0%       81.0%       13.0%        2.0%
prophase            0.0%        3.0%       94.0%        3.0%
telophase           2.0%        4.0%       47.0%       47.0%

mean diagonal 64.8%   (was 70.5% under the leaky judge)
```

| prompted | leaky judge | clean judge | leaks to |
|---|---|---|---|
| prophase | 95% | 94% | — |
| metaphase | 79% | 81% | prophase 13% |
| telophase | 57% | **47%** | prophase 47% |
| anaphase | 51% | **37%** | metaphase 49% |

**The baseline Phase 1 must beat is 64.8% mean / anaphase 37% / telophase 47%**, and every future
comparison must use the clean judge. The gate conclusion is unchanged (64.8% ≫ 25% chance), but
the two weak phases are weaker than first reported — anaphase is now closer to being *mistaken
for* metaphase (49%) than correctly generated (37%).

### Blockers and findings from the Phase 0 run

**1. `dvc pull` cannot restore the binary classifier assets — they are gone from the remote.**
`dvc pull` exits 0 but leaves `datasets/crops/binary_classifier/` absent and the three weights in
`src/allium_cepa_classifier/weights/` as bare `.dvc` pointers. `dvc status --cloud` reports four
`.dir` hashes and three file hashes *"do not exist neither locally nor on remote"*.

Not blocking this plan — the phase classifier is built from `datasets/crops/vae/` and trained from
scratch, and Phase 2's `kid_classifier` uses phase crops as its reference set rather than
`binary_classifier/test/mitosis`. But it does mean **the published inference weights
(`object_detection.pt`, `classifier_calibrated.pt`, `yolo_isotonic_calibrator.pkl`) are currently
unrecoverable from DVC** — a separate project-level problem worth addressing on its own.

**2. Leakage across the VAE crop splits — two distinct defects.**

*Exact-duplicate leakage (the severe one).* Held-out crops are byte-identical copies of training
crops, not merely similar:

```
unique content hashes   train=955  val=240  test=242   (from 1441 files)
test  crops also in train : 200 / 242  (83%)
val   crops also in train :   2 / 240  (1%)
val   ∩ test              :  42
total unique crops        : 1194, not 1441
```

*Source-micrograph leakage.* On top of that, crops were split per-crop, so cells from one field
of view land in different splits and share staining, illumination, focus and background cues.
After deduplication the 1194 unique crops come from only **508 source micrographs**.

The 97.5% test accuracy was therefore mostly memorisation of literally the same images.

The classifier can exploit field-level cues (staining, illumination, focus, background texture)
shared between a train crop and a test crop from the same slide. Consequences differ sharply by
use:

- **As the Phase 0/2 judge: acceptable.** The judge scores *generated* images, which belong to no
  split — there is nothing to leak across. Its 97.5% should simply not be quoted as its true
  accuracy.
- **As the Phase 4 downstream baseline: invalid.** A leaky test set puts the baseline at 0.97
  macro-F1, leaving essentially no headroom for synthetic data to demonstrate anything. Phase 4
  cannot produce a meaningful result on this split.

**Added to Phase 1 (see below): a group-aware re-split at the source-micrograph level.**

**3. Latent bug in the existing `classifier_judge` metric.** `evaluate_lora.py::_load_classifier`
hand-assembles a timm model and assigns `model.classifier = nn.Sequential(...)`, but the trainer
saves a `BackboneWithHead` whose state dict uses `backbone.*` / `head.*` keys. That loader would
raise on any real checkpoint — it has never run, because the weights were absent. Phase 2 replaces
it with `scripts/utils/phase_judge.py`, which rebuilds via `build_model()` and matches the saved
keys. The dead code path in `evaluate_lora.py` should be deleted when Phase 2 lands.

---

## Phase 1 — Fix conditioning and imbalance in the dataset layout

**Goal:** make the phase token actually control generation, and balance exposure across phases.
**Cost:** ~half a day, no GPU.

- [x] **Dedup + group-aware re-split (from the Phase 0 leakage finding).**
      `scripts/utils/phase_classifier_dataset.py` rewritten: deduplicates by md5, groups crops by
      source micrograph (normalising away Roboflow re-upload hashes so one image uploaded twice
      maps to one group), and assigns whole groups via greedy stratified fill. Asserts zero
      content-hash overlap and zero group overlap across splits, and writes an auditable
      `split_manifest.json`.

      ```
      split           prophase   metaphase    anaphase   telophase    total   groups
      train                354         222         175         207      958      413
      validation            44          27          22          25      118       47
      test                  44          27          22          25      118       48
      TOTAL                                                            1194      508
      ```

- [x] **Honest phase-classifier baseline: macro-F1 0.91** (was an inflated 0.97), accuracy 0.9068,
      ECE 0.0982 → 0.0725. Per-class F1: prophase 0.90, metaphase 0.93, anaphase 0.90,
      telophase 0.88. **This is the number Phase 4 must beat.**

      *Caveat for Phase 4:* the clean test set is only **118 crops** (anaphase 22, metaphase 27,
      telophase 25, prophase 44). At 0.91 accuracy the 95% CI is roughly ±5 points, so only fairly
      large effects will be statistically detectable. Phase 4 should therefore report per-phase
      recall with CIs and average over several seeds rather than reading a single run.

- [x] **Per-phase concept folders.** `scripts/utils/lora_dataset.py` gained `--layout per-phase`,
      `--dedup`, and `--target-per-phase`; repeats are computed per phase to equalise
      samples-per-epoch rather than hardcoded. New `per_phase` version registered in the
      `prepare_lora_dataset` DVC foreach.

      ```
      phase           orig     aug   images  repeats  per epoch
      telophase        257     257      514        6       3084
      anaphase         219     219      438        7       3066
      metaphase        276     276      552        5       2760
      prophase         442     442      884        3       2652
      TOTAL           1194    1194     2388               11562
      ```

      Balanced within 14%, versus the 2.4× prophase-to-anaphase skew before. Verified in
      `train.log` that kohya reads all four concepts with repeats 3/5/6/7.

- [x] **`text_encoder_lr: 5.0e-5`** (5× the UNet LR) in `experiments/lora/p3_per_phase/config.yaml`.
      Previously unset, so the text encoder inherited `learning_rate: 1e-5`.

- [x] **Re-ran the diagnostic on `p3_per_phase` (clean judge, 100 samples/phase): 75.2% mean,
      up +10.4 points from the corrected 64.8% baseline.**

      ```
                      anaphase   metaphase    prophase   telophase
      anaphase           52.0%       43.0%        0.0%        5.0%
      metaphase           1.0%       93.0%        6.0%        0.0%
      prophase            1.0%        8.0%       89.0%        2.0%
      telophase          10.0%        1.0%       22.0%       67.0%
      ```

      | prompted | p2 baseline | p3_per_phase | Δ | judge ceiling |
      |---|---|---|---|---|
      | metaphase | 81% | **93%** | +12 | 100% |
      | prophase | 94% | 89% | −5 | 86% |
      | telophase | 47% | **67%** | +20 | 92% |
      | anaphase | 37% | **52%** | +15 | 86% |
      | mean | 64.8% | **75.2%** | +10.4 | 91% |

      The two targeted phases gained most (+20 telophase, +15 anaphase), and telophase's
      confusion with prophase more than halved (47% → 22%). Prophase's −5 is the expected cost of
      rebalancing: its repeats dropped to 3, the lowest of the four, after being the
      over-represented phase.

### Reading the numbers against the judge ceiling

The diagnostic cannot score a phase above the judge's own recall on *real* crops of that phase, so
per-phase recall from the clean-split test set is the practical ceiling (an approximation — the
judge faces domain shift on generated images — but far better than assuming 100%):

| phase | p3 diagonal | judge recall on real | remaining gap |
|---|---|---|---|
| metaphase | 93% | 100% | 7 |
| prophase | 89% | 86% | **0 — at ceiling** |
| telophase | 67% | 92% | 25 |
| anaphase | 52% | 86% | **34** |

Prophase is already as good as this instrument can measure; its −5 is noise against the ceiling,
not a regression. **Anaphase is the one real outstanding problem** — still mistaken for metaphase
(43%) nearly as often as generated correctly (52%), and holding the largest gap to ceiling.

### Phase 1 verdict

Conditioning is good enough to build on. The changes that produced +10.4 (dedup + per-phase
folders + `text_encoder_lr`) remain bundled, so Phase 3 should carry `p3_per_phase` forward as its
base and put `text_encoder_lr` and the per-phase balance tilt into the search space rather than
fixing them by hand — the direction of travel suggests more of both may still help.

If Phase 3 leaves anaphase short of ~70%, the fallback is anaphase-specific rather than global: a
distinctive rare caption token (`sks_anaphase`) so CLIP cannot blend it with its weak "anaphase"
prior, or a dedicated LoRA for that phase alone.

**Confound to state plainly:** `p3_per_phase` changes three things at once relative to
`p2_full_tricks` — dedup, per-phase folders, and `text_encoder_lr`. If it improves, we will not
know from this run alone which change caused it. That is an acceptable trade at Phase 1, whose
question is only *"is conditioning good enough to build on?"*; Phase 3's Optuna study is where
the individual contributions get separated.

**Fallback if conditioning still fails:** four separate LoRAs, one per phase. Reliable
conditioning at 4× training cost and no shared learning across phases. A cheaper intermediate is
rare-token captions (`sks_prophase` rather than `prophase`) so CLIP's weak priors for the real
words don't blend the concepts.

### Results box

```
Post-fix conditioning diagonal:  ___  (vs Phase 0: ___)
Fallback triggered? ___
```

---

## Phase 2 — The metric module

**Goal:** replace `final_loss` with a metric stack that measures what the project actually needs.
**Cost:** ~half a day, no GPU beyond re-scoring.

All metrics register into the existing `@register` registry in
[evaluate_lora.py](scripts/evaluate_lora.py) and are computed **per phase**, not aggregate.

### 2a. `phase_consistency` — the primary objective

Generate N per phase, classify with the Phase 0 judge, report per-phase recall and the macro
mean. **Realism is worthless without this.** A generated image labelled `metaphase` that depicts
prophase is worse than no data — it is label noise injected directly into the minority classes
the project exists to fix.

### 2b. `kid_classifier` — the primary distributional metric

- **KID, not FID.** KID (an unbiased MMD estimator) is usable at n≈500; FID is badly biased at
  small n and its covariance term is rank-deficient below the feature dimension.
- **Classifier features, not Inception.** InceptionV3 is ImageNet-trained and near-blind to
  grayscale microscopy. Use the phase classifier's penultimate features.
- Generate **≥512 images per phase** (~10–15 min on the 3090, acceptable against 77 min of
  training) against a fixed real reference set.

### 2c. `kid_vqgan` — complementary feature space (uses the VQGAN weights)

Load `vqgan/vqgan weights/vqmodel/` as a diffusers `VQModel` and use its **encoder** as a second
feature extractor. This is a genuinely different view: classifier features are *phase-semantic*
and largely texture-invariant; VQGAN encoder features are *reconstructive* and encode texture,
noise floor, contrast, and local structure. A sample with plausible chromosome arrangement but
SD-typical artifacts (checkerboarding, wrong grain, oversaturated background) passes
`kid_classifier` and is caught here.

Implementation constraints — the tap point matters:

- Use **pre-quantization continuous** activations. Post-quantize, every vector is snapped to one
  of 16384 codes in 4-D; almost all information is destroyed.
- Tap the **512-channel activation immediately before `encoder.conv_out`**, then global-average-
  pool over spatial → a **512-dim** descriptor. Do *not* use `conv_out`'s output (only 4
  channels) or the raw spatial map (128×128×4 = 65,536 dims, hopeless at n≈500).
- Verify the input normalization convention (diffusers `VQModel` expects `[-1, 1]`) and the
  resolution the model was trained at before trusting the numbers.

**Reported as a diagnostic and tiebreaker, never as the optimization target.** Reconstructive
features are not organized around phase semantics, so minimizing them directly rewards texture
matching over correct biology.

### 2d. `vqgan_recon_error` — artifact detector (uses the VQGAN weights)

Encode→decode each generated image through the cell-domain `VQModel` and report reconstruction
error. High error means the image contains structure a cell-trained autoencoder never learned —
i.e. generation artifacts. One extra forward pass over images already generated for 2b/2c.

### 2e. `memorization` — the guard

95th-percentile max cosine similarity from each generated image to the training set, in
classifier feature space. **At rank 256 on ~300 images per phase this is a live risk**, and a
memorizing model scores *excellent* on every distributional metric. Must be measured, not assumed.

### 2f. `coverage` — the diversity guard

Coverage / density (Naeem et al.). Catches mode collapse, which memorization metrics alone will
not — a model can produce novel-but-identical outputs.

### Composite objective

```
minimize   kid_classifier (macro over phases)
subject to phase_consistency ≥ floor
           memorization     ≤ ceiling
           coverage         ≥ floor
```

Violations are penalized rather than hard-rejected, so the optimizer still gets gradient
information from bad regions. Set the thresholds from the Phase 1 run, not a priori.

### Tasks

- [x] Implemented 2a–2f as registered metrics in `scripts/evaluate_lora.py`, backed by
      `scripts/utils/lora_metrics.py`. Sample generation factored into
      `scripts/utils/lora_samples.py` and shared with `diagnose_phase_conditioning.py`, so a
      metric and a diagnostic can never be computed over differently-generated images.
- [x] Removed the two dead metrics: `fid` (ImageNet Inception features; also pointed at
      `datasets/crops/binary_classifier/`, absent from the DVC remote) and `classifier_judge`
      (loader could not have loaded a `BackboneWithHead` checkpoint).
- [x] Verified the VQGAN tap point empirically: `encoder.conv_out` receives `[B, 512, 128, 128]`,
      GAP → 512-dim. Also confirms f=4 (512px → 128px) against SD1.5's f=8, the second reason it
      cannot be swapped into the pipeline.
- [ ] **Re-score the 8 p2 runs + p3_per_phase** at 100 samples/phase (in progress).

### Two corrections made during implementation

Both were metrics that ran without error and produced numbers that meant something other than
what they appeared to.

**Coverage was biased by unequal sample sizes.** Scoring 100 generated against up to 442 real
crops, a perfect generator still cannot light up every real neighbourhood, and the bias varies by
phase according to how many real crops that phase happens to have. `coverage_density` now
subsamples the larger set. For `p3_per_phase` this moved coverage from **0.140 → 0.333** — the
original reading would have looked like severe mode collapse.

**Memorisation was uninterpretable without a control.** `nn_p95 = 0.834` looks alarming until you
know how similar two *genuinely different* real cells are in phase-classifier feature space.
`real_to_real_baseline` adds leave-one-out nearest-neighbour similarity within the real training
set; the reported number is now `memorization_excess_p95 = generated − real-vs-real`.

For `p3_per_phase`: generated `nn_p95` 0.834 vs real-vs-real 0.904 → **excess −0.071**. Generated
images sit *further* from the training set than real crops sit from each other. No memorisation.
(This LoRA trains on every real crop, so there is no held-out real set; real-vs-real is the only
available control.)

### Results box

```
p3_per_phase, 100 samples/phase:
  phase_consistency 0.7525   (cross-checks exactly against the diagnostic's 75.2%)
  kid_classifier    0.20168
  kid_vqgan         0.13754
  coverage          0.3325     density 0.531
  memorization      nn_p95 0.8335, real baseline 0.9043, excess -0.0708
  vqgan_recon_ratio 1.2238     (generated are 22% harder to reconstruct than real)

All 9 runs re-scored at 100 samples/phase:

| experiment | phase_cons | kid_clf | kid_vq | coverage | density | mem_excess | avg_loss |
|---|---|---|---|---|---|---|---|
| p3_per_phase | **0.752** | **0.202** | 0.138 | 0.333 | **0.531** | −0.071 | 0.083 |
| p2_min_snr | 0.725 | 0.236 | 0.170 | 0.305 | 0.504 | −0.094 | 0.075 |
| p2_baseline | 0.703 | 0.250 | 0.166 | 0.275 | 0.473 | −0.095 | 0.115 |
| aug2x_r256 | 0.703 | 0.250 | 0.166 | 0.275 | 0.473 | −0.095 | 0.115 |
| p2_ip_noise | 0.672 | 0.276 | 0.194 | 0.287 | 0.465 | −0.106 | 0.123 |
| p2_noise_min_snr | 0.670 | 0.207 | **0.073** | **0.347** | 0.361 | −0.091 | 0.076 |
| p2_full_tricks | 0.647 | 0.235 | 0.108 | 0.287 | 0.411 | −0.086 | 0.084 |
| p2_noise_ip | 0.613 | 0.272 | 0.148 | 0.273 | 0.399 | −0.099 | 0.125 |
| p2_noise_offset | 0.608 | 0.238 | 0.086 | 0.287 | 0.346 | −0.093 | 0.116 |
```

**Conclusions changed — substantially.**

**`noise_offset` is harmful, with perfect separation.** Every run without it (0.672, 0.703, 0.725)
beats every run with it (0.608, 0.613, 0.647, 0.670). Seven runs, no overlap. The old
`final_loss` ranking placed `p2_noise_offset` 3rd. `min_snr_gamma` helps consistently (+0.022
without noise_offset, +0.062 with); `ip_noise_gamma` is neutral to slightly negative.

**The Phase 1 carry-forward base was the wrong choice.** `p2_full_tricks` ranks 6th of 7 p2 runs.
It was picked as the diagnostic subject before this metric existed, and `p3_per_phase` inherited
all three tricks from it — so p3 reached 0.752 *carrying* the one flag that demonstrably hurts.
Removing `noise_offset` is the highest-expected-value change into Phase 3.

**Reproducibility, end to end.** `aug2x_r256` and `p2_baseline` are byte-identical configs trained
in separate runs and scored identically to 5 decimal places on every metric — config → weights →
samples → metrics is deterministic.

**Statistical caveat.** 400 samples per run gives SE ≈ 0.023 on `phase_consistency`, so individual
pairwise gaps are 1–2 SE. The 7-run separation on `noise_offset` is far stronger evidence than any
single comparison. All single-seed.

**The metrics genuinely disagree.** `p2_noise_min_snr` has the best `kid_vqgan` (0.073) and best
coverage (0.347) but mediocre `phase_consistency` (0.670). No single number orders these runs,
which is why the Phase 3 objective needs explicit weights rather than one metric.

Coverage 0.33 with density 0.53 is the standing concern: the generator reaches about a third of
the real distribution's neighbourhoods. That is the diversity signal Phase 3 should optimise
against, and it is invisible to `phase_consistency` and only weakly visible to KID.

---

## Phase 3 — Optuna study

**Goal:** systematic HPO under the corrected metric. **Cost:** ~40h — the bulk of the budget.

**Optuna, not scikit-optimize.** With start/stop operation this is not a preference: Optuna's
SQLite storage makes the study resumable across interruptions, which skopt has no clean answer
for. It also handles the conditional/categorical structure of this space (a trick being
present/absent is not a continuous dimension) and skopt is effectively unmaintained against
current numpy/sklearn.

### Search space

| parameter | range |
|---|---|
| `learning_rate` | log-uniform 1e-6 … 1e-4 |
| `text_encoder_lr` | as a ratio of `learning_rate`, 0.5 … 10 |
| `network_dim` | categorical {16, 32, 64, 128, 256} |
| `network_alpha` | as an alpha/dim ratio, {0.25, 0.5, 1.0} |
| `noise_offset` | conditional: absent, or 0.02 … 0.15 |
| `min_snr_gamma` | conditional: absent, or 1 … 20 |
| `lr_scheduler` | categorical {cosine, constant_with_warmup} |
| `dataset_version` | categorical {per_phase, per_phase_aug2x} |

**Include the low ranks.** Rank 256 / alpha 128 is very high capacity for ~300 images per phase
and is the most likely source of the memorization risk in 2e. The current "best" config may be
memorizing.

### Tasks

- [x] `scripts/optimize_lora.py` — Optuna study, SQLite storage at
      `experiments/lora/_studies/lora_phase/study.db`, resumable via `load_if_exists=True`.
      Each trial writes a full experiment dir (`trial_XXX/config.yaml` + artifacts), so trials are
      ordinary experiments: inspectable, re-runnable, comparable with the hand-built ones.
      `optuna>=4.0.0` added to the `lora` dependency group.
- [x] Seeded via `enqueue_trial` (simpler and more robust than `add_trial`) with three points
      from the Phase 1/2 findings: p3_per_phase's exact config; the same minus `noise_offset`;
      and the same minus both `noise_offset` and `ip_noise_gamma`.
- [x] Composite objective (maximise), weights scaled to observed metric spreads:
      `phase_consistency − kid_classifier + 0.5·coverage − 5·max(0, memorization_excess)`.
      Validated against real numbers: `p3_per_phase` scores 0.717, while a hypothetical
      memoriser with *better* raw metrics (pc 0.90, kid 0.05) scores 0.50 — the guard bites.
      A trial with missing metrics returns −inf so a broken run is never mistaken for a good one.
      All secondary metrics are stored as trial `user_attrs` for post-hoc Pareto analysis.

**Pruning was dropped.** Kohya emits exactly one image per prompt per sample event, which is far
too few for a usable mid-training `phase_consistency` or KID. Getting enough would mean either
duplicating each prompt ~25× in `sample_prompts.txt` (adding ~12 min/run) or staged
train/save/resume — both cost more complexity than the ~1.5× throughput they buy. Instead
`max_train_steps` ∈ {800, 1200, 1600, 2000} is a searched dimension, letting Optuna find
cheap-and-good regions on its own.

Expected: **~30 trials in ~40h** (≈60 min train at the average step count + ≈18 min eval).

### 2026-08-17 — the metric stack had no intra-set diversity measure

Found by eye, not by metric. `lora_nofill_v1/trial_016` posted the best `kid_classifier` (0.607)
and the highest `coverage` (0.550) ever measured, with `phase_consistency` 0.980 — above the
judge's own 0.91 accuracy on *real* crops. Visual inspection of its 100 samples per phase found
**~4 recurring templates with variations**.

**Why every existing guard missed it.** All four objective terms compare generations to something
*else* — `phase_consistency` to the label, `kid_classifier` and `coverage` to real crops,
`memorization` to training crops. None compares generations **to each other**. A generator
emitting a few templates passes all of them: if the templates land in dense regions of the real
distribution, coverage counts many real neighbourhoods as reached and KID only checks aggregate
statistics. (One corroborating signal was already visible: trial_016's memorisation excess is
−0.010, the tightest of any run, against −0.03…−0.10 elsewhere.)

**Two failed attempts before one worked**, both worth keeping:

1. **`mean_pairwise_similarity` does not discriminate.** Across trial_016 (templated), `p5_noaug`
   (healthy) and trial_006 (poor) it returned **0.912 / 0.910 / 0.918**. Templating is *local
   clustering*, so the statistic must be local — averaging over all pairs is dominated by
   cross-template pairs and dilutes the near-twins away. This is the same argument already in
   `memorization_score`'s docstring ("a mean would dilute them away"), contradicted in practice.
2. **VQGAN feature space is useless for diversity.** Chosen first on the reasoning that the judge
   is invariant to the staining/illumination variation the eye notices. The measurement refuted
   it: GAP-pooled VQGAN features have an **effective rank of ~1.2** — a single dimension — and
   gave `nn_ratio` 0.998 / 0.997 / 0.992. The judge's 1408-d features carry the structure
   (effective rank ~10.6 on real crops).

**What works — two statistics in judge space, each a ratio to real crops:**

| | trial_016 | p5_noaug | trial_006 | reads as |
|---|---|---|---|---|
| `nn_self_similarity_ratio` | **1.053** | 0.935 | 0.785 | **>1 = TEMPLATING** (tighter clusters than real) |
| `effective_rank_ratio` | 1.090 | 0.825 | **0.267** | **<<1 = MODE COLLAPSE** (few modes) |

The signatures are **independent**, and the two runs fail differently: trial_016 has tight
clusters but a healthy number of them (many templates), while trial_006 genuinely collapsed to
few modes (effective rank 2.94 vs real 10.56). trial_016 is the only run whose generations sit
closer to each other than real cells do.

Registered in `evaluate_lora.py` as `self_similarity`, **reported but deliberately NOT in
`optimize_lora.score()`** — adding a term mid-study would make trials before and after
incomparable, the exact mistake that voided the first study. Both run on cached samples, so they
apply retroactively to every finished trial with no retraining.

Also fixed a footgun found while building it: `evaluate_lora.py` **overwrote** `metrics.json` with
only the metrics it computed, so `--metrics self_similarity` would have destroyed the rest. New
`--merge` flag, off by default so stale values on a superseded scale cannot silently survive a fix.

**TODO when the study lands:** apply both metrics retroactively across all 30 trials plus the
hand-built runs, re-rank with templating and collapse visible, and check whether trial_016's
1.364 composite survives.

### 2026-08-16 — REVISION: oversmoothing is intrinsic, not an augmentation artifact

The 4th arm (`p5_d4x3_nojitter`, D4 orientation with the colour jitter removed) completes the
ablation and **corrects the attribution made from the first three arms**:

| arm | jitter | resampling | composite | kid_cls | coverage | recon_ratio |
|---|---|---|---|---|---|---|
| **A `p5_noaug`** | no | no | **1.123** | **0.739** | **0.472** | 0.448 |
| D `p5_d4x3_nojitter` | no | no | 0.960 | 1.132 | 0.407 | 0.455 |
| C `p3_per_phase_nofill` | yes | yes | 0.954 | 1.273 | 0.340 | 0.444 |
| B3 `p5_d4x3` | yes | no | 0.935 | 1.243 | 0.372 | **0.653** |

**Earlier claim, now withdrawn:** that B3's `recon_ratio` recovery proved resampling caused the
oversmoothing. Arm D has no resampling *and no jitter* and reads 0.455, not 0.653. The effects are
roughly additive and opposite — **jitter raises `recon_ratio` ~+0.2, resampling lowers it ~−0.2**
— and in arm C they cancel. B3's 0.653 came mostly from the *jitter* adding photometric variation.

So with neither treatment the generator sits at ~0.45: **oversmoothing is intrinsic to the
generator, not caused by augmentation.** The jitter was masking it while simultaneously wrecking
the distributional metrics. This matters for Phase 4 — synthetic crops lack the grain, dust and
focus noise of real microscopy, and that needs a different fix entirely.

**Offline augmentation of any kind loses.** Even perfectly lossless D4 orientation copies (arm D)
lose to no copies at all. Likely mechanism: total exposure is fixed at ~30 000 images seen, so each
unique real crop is seen **25 times in A but only 6.3 times in D** — the copies consume training
budget to teach an orientation invariance SD1.5 already has.

### 2026-08-16 — the study's dataset was stale, and offline augmentation was hurting

Two findings, both from single runs but on metrics whose run-to-run noise is measured
(same config twice: `phase_consistency` 6.3%, `kid_classifier` 1.8%, `coverage` 1.5%).

**1. The whole 21-trial study trained on the wrong dataset.** `optimize_lora.py:150` hardcoded
`dataset_version: "per_phase"` — the version carrying the dark-red rotation wedges — so every
trial predated the fix, which is worth +0.085 `phase_consistency`, larger than almost every gap
the study was ranking. Now a `--dataset-version` flag, recorded in the study's `user_attrs`, with
a mismatch on resume being a hard error.

**The study's headline finding inverts on clean data.** `p4_r16_nofill` re-runs trial_020's exact
hyperparameters on `per_phase_nofill`:

| | rank 256 (C) | rank 16 (trial_020's params) |
|---|---|---|
| phase_consistency | **0.838** | 0.760 |
| kid_classifier | **1.273** | 1.995 |
| coverage | **0.340** | 0.302 |
| composite | **0.954** | 0.747 |

Telophase carries the loss: 0.79 → 0.44, with 41% of it predicted as prophase (was 15%). The
hypothesis was right — fitting the spurious canvas cost capacity, so rank 16 won only because it
was too small to learn the artifact. Remove the contamination and capacity earns its keep.
**The next study must re-explore rank from scratch, not seed low.**

**2. Offline augmentation was making things worse.** Three arms at identical hyperparameters,
differing only in training data:

| arm | augmentation | images | composite | pc | kid_cls | coverage | recon_ratio |
|---|---|---|---|---|---|---|---|
| C `p3_per_phase_nofill` | resample + jitter | 2388 | 0.954 | 0.838 | 1.273 | 0.340 | 0.444 |
| **A `p5_noaug`** | **none** | 1194 | **1.123** | 0.823 | **0.739** | **0.472** | 0.448 |
| B3 `p5_d4x3` | D4 (lossless) + jitter | 4776 | 0.935 | 0.787 | 1.243 | 0.372 | **0.653** |

**A wins, and the margins dwarf the noise.** `kid_classifier` 1.273 → 0.739 is a 42% improvement
against 1.8% run-to-run variation; `coverage` 0.340 → 0.472 is +39% against 1.5%. The
`phase_consistency` cost (0.838 → 0.823, 1.8%) sits inside the 6.3% noise band. Against the
reference points — real-vs-real floor 0.275, wrong-phase 6.21 — A closes the distributional gap
from 16.8% to **7.8%**.

**The two augmentation defects are separable, and the jitter is the expensive one.** B3 removes
the resampling and keeps the jitter: `vqgan_recon_ratio` recovers 0.444 → **0.653** (the blur
hypothesis confirmed — `rotate_without_fill`'s two BILINEAR passes plus 1.22x upscale really were
the oversmoothing), but `kid_classifier` barely moves (1.273 → 1.243) and coverage stays low.
A removes both and takes the distributional metrics. So the brightness/contrast jitter of 0.7–1.3
is what pushes generated colour statistics off the real distribution — **confirming the handoff's
unresolved item 3** ("oversaturated chromosomes... suspect `augment_mild`'s jitter").

**Consequence: the next study should train on `per_phase_noaug`.** Simpler *and* better — 1194
unique crops, no synthetic copies, per-phase folders, dedup, repeats balanced to ~3000/phase.

**Open, one run to settle:** A still has the best coverage despite the fewest images, so
orientation diversity did not help — but B3's only orientation arm carried the jitter that
independently suppresses coverage. A fourth arm (D4 x3, jitter removed) would disentangle them
and could combine A's clean statistics with genuine orientation diversity. Coverage is the
weakest term in the objective, so 1.5h against a 30h study is a reasonable price.

### 2026-08-10 — the KID estimator was broken, and it corrupted the study

After 21 trials, `--report` ranked **trial_003 first with a score of 9.04** despite a
`phase_consistency` of 0.412 (near chance) and the worst coverage in the study (0.180). Its
`kid_classifier` was **−8.54**, driven entirely by one phase: anaphase at **−35.0** while the
other three sat at 0.195 / 0.270 / 0.403.

**Root cause.** One degenerate generated anaphase crop had a classifier feature norm of **2566**
against a median of 13.7 — a 188× outlier. The canonical KID kernel `(x·y/d + 1)^3` assumes
Inception pool features, whose scale is well behaved; ours are not, and the cubic amplified that
sample to ~1e11 in the `k(y,y)` block. Because the unbiased estimator strips the `k(y,y)`
diagonal but has no diagonal to strip from `k(x,y)`, the outlier's contribution landed
asymmetrically — roughly `+340` into `k(y,y)` against `−376` from `−2·k(x,y)` — and MMD² went to
−35. A metric intended to be bounded below by ~0 became an unbounded reward.

**Fix.** L2-normalise features before the kernel, and use `(x·y + 1)^3`. Every dot product is
then in [−1, 1] and the kernel in [0, 8], so no single sample can dominate. Verified on known
answers: real-vs-real same phase 0.275, real-vs-real different phase 6.21, and trial_003's
anaphase moved from −35.0 to **2.16** — correctly *worse* than p3_per_phase's 1.62. Values are on
a new scale and are not comparable with anything computed before this date.

**Second-order breakage: the objective weights.** The fix moved KID's scale from ~0.2 to
~1.6–6.2, which left `W_KID = 1.0` dominating `phase_consistency` about 20:1 and ranking a
memoriser *above* an honest run. `W_KID` is now 0.1, re-derived from the new spread. This was
caught by a unit test, not by inspection.

**Recovery.** The study was stopped — TPE had been fitting a corrupted objective since roughly
trial 008. No training was lost: `--reseed-from` replays finished trials into a fresh study,
rescoring each from its cached samples via `params_from_config`, recovering ~28 GPU-hours.

**Tests added**, each asserting a property the failure violated:
`tests/test_lora_metrics.py` (12 tests — KID near zero on identical distributions, non-negative
on disjoint ones, stable under a 200× norm outlier; coverage invariant to reference-set size;
memorisation baseline excludes self-matches) and `tests/test_optimize_lora.py` (5 tests — params
round-trip losslessly through `config.yaml`, every sampled param is declared in `DISTRIBUTIONS`
and in range, disabled tricks are omitted rather than written as 0.0, and a memoriser cannot
outrank an honest run). Also repaired `test_registry_loss_reads_tb_events`, which had been
failing independently since `experiments/lora/sd15_rank16` was deleted.

**Lesson for the remaining phases.** Both defects produced plausible-looking numbers and neither
raised an exception. Any metric entering an optimisation objective needs a known-answer test
*before* it drives GPU spend, and any change to a metric's scale invalidates the objective
weights derived from the old one.

### Results box — 21 trials, corrected metrics (2026-08-11)

Full table and analysis: **`experiments/lora/_studies/LEADERBOARD.md`**. Reseeded study:
`lora_phase_v2`. Weights re-derived from measured spreads (pc 0.635 / kid 2.343 / cov 0.375):
`W_KID = 0.135`, `W_COVERAGE = 0.847`, giving the intended 2:1:1 contribution.

```
Best: trial_020  score 1.053  pc 0.850  kid 1.193  coverage 0.430  (network_dim 16!)
  2nd: trial_013 score 1.003  pc 0.828  kid 1.024  coverage 0.370  (network_dim 64)
  3rd: trial_000 score 0.911  pc 0.800  kid 1.232  coverage 0.328  (rank 256 = p3's config)
Memorisation: 0 of 30 experiments show any copying (excess p95 in [-0.448, -0.026]).
```

**Rank 16 beats rank 256**, confirming the over-parameterisation hypothesis — and meaning every
p2/p3 conclusion was drawn inside a bad region of the search space.

**`noise_offset` splits.** On `phase_consistency` the Phase 2 finding holds (dropping it gains
~4 points: 0.800 → 0.840/0.845 across the matched seeded trials). On the composite score it
reverses, because `noise_offset` markedly improves KID (1.23 vs 1.70). The best trial uses a
*small* value (0.024), less than half p2's 0.05. It trades phase accuracy for distributional
realism; 0.05 was simply too strong.

### The reproducibility problem — supersedes earlier variance estimates

`trial_000` is byte-for-byte `p3_per_phase`'s config, on the same dataset and latent caches
(built 2026-08-09 16:31–16:34, never rebuilt). Results diverge:

| | mean pc | anaphase | telophase | kid |
|---|---|---|---|---|
| p3_per_phase | 0.752 | **0.52** | 0.67 | 1.255 |
| trial_000 | 0.800 | **0.72** | 0.65 | 1.232 |

LoRA tensors genuinely differ (max abs diff 1.77e-2 over 151M params) — fp16/cuDNN
nondeterminism over 2000 steps, not a config or data difference. Yet `aug2x_r256` and
`p2_baseline`, also identical configs, are **bit-identical** (0.000e+00). Training is
deterministic only sometimes, depending on GPU conditions and kernel autotuning.

**This invalidates the earlier "SE ≈ 0.023" estimate**, which counted only sampling noise from
400 generated images. True run-to-run variance is **~5 points on mean `phase_consistency` and
~20 points on anaphase**. Any gap below ~0.05 score is noise, the study is single-seed and has
been optimising signal + noise, and TPE preferentially selects configurations that got lucky —
so the top of the leaderboard is biased upward.

It also means **the anaphase weakness was likely overstated**: 0.52 in p3, 0.72 in its own
duplicate, and 0.82–0.84 in several trials. The anaphase-specific fallback (rare token,
dedicated LoRA) should not be started until a multi-seed measurement confirms the weakness is
real.

**Recommendation before any further search:** re-run the top 3 configs at 2–3 seeds (~5h) to see
how much of the ranking survives. That determines whether the remaining ~19 trials are worth
running at all.

---

## Phase 4 — Downstream validation

**Goal:** answer the only question that justifies the project. **Cost:** ~4h.

Top 3 configs from Phase 3 only.

- [x] **`scripts/validate_synthetic_downstream.py` built and smoke-tested** (2026-08-10). Generates
      synthetic crops via the shared `ensure_phase_samples`, assembles train = real + ratio×synthetic
      with validation/test left purely real (symlinks, so a ratio-2.0 run costs no disk), trains,
      and reports per-phase F1 as mean ± std across seeds. Both guards are enforced in code: the
      script **refuses `--arch efficientnet_b2`** outright, and the test split is real crops only.
      `wilson_interval()` is available for per-phase recall CIs at these small n.
- [ ] Run for the top 3 Phase 3 configs at ratios 0 / 0.25 / 0.5 / 1.0 / 2.0, 3 seeds.

**Smoke test (p3_per_phase, 1 seed — not yet meaningful, plumbing only):**

```
ratio 0.00   macro_f1 0.7690   anap 0.750  meta 0.833  prop 0.765  telo 0.727
ratio 0.25   macro_f1 0.7896   anap 0.737  meta 0.844  prop 0.800  telo 0.778
```

**Decision needed on the evaluator architecture.** resnet50 baselines at 0.769 macro-F1 where the
efficientnet_b2 judge reaches 0.91 — it is simply a weaker learner on this task. That cuts both
ways: more headroom for synthetic data to show a gain, but a gain on a weak evaluator transfers
less convincingly. Options: (a) keep resnet50 and report the gap honestly, (b) tune resnet50 so
its baseline is competitive before measuring, (c) use efficientnet_b1 as a middle ground — still
architecturally distinct from the b2 judge, closer in capacity. Non-circularity is
non-negotiable; which non-judge architecture is a judgement call.

### Methodological guard — do not skip this

If the same classifier is both the Phase 2 judge and the Phase 4 evaluator, the result is
circular: generations are optimized to please classifier C, and then C is reported to have
improved. Mitigations, both required:

1. **Different architecture and seed** for the Phase 4 evaluator than the Phase 0/2 judge
   (e.g. judge = `efficientnet_b2`, evaluator = `resnet50`). Do not share weights.
2. **Evaluate only on the real held-out test set.** This is the main protection and holds
   regardless of the judge.

### Results box

```
                     macro-F1   prophase  metaphase  anaphase  telophase
real only (Phase 0)  ___        ___       ___        ___       ___
+ synth 25%          ___        ___       ___        ___       ___
+ synth 50%          ___        ___       ___        ___       ___
+ synth 100%         ___        ___       ___        ___       ___
+ synth 200%         ___        ___       ___        ___       ___
```

---

## Phase 5 — Agent as experiment manager (optional)

Only after the loop runs unattended.

**Not as the optimizer.** TPE will out-search an LLM on this space and will not burn tokens doing
it. The agent's value is in the parts Optuna structurally cannot do:

- **Triage** — diagnose OOM / NaN loss / crashed trials, repair the config, resume the study.
- **Qualitative review** — inspect sample grids for what KID cannot see ("every metaphase is the
  same cell", "backgrounds have gone purple").
- **Search-space surgery** — after ~20 trials, read Optuna param importances and propose
  narrowing or shifting bounds. This is the genuine judgment call: TPE cannot change its own
  search space.
- **Write-up** — turn the completed study into an ablation document in the style of
  `experiments/lora/TRAINING_TRICKS.md`.

---

## Phase 6 — Autoencoder fidelity (deferred)

Only worth doing once controllability is solved and the Phase 4 curve has flattened.

- [ ] **Cheap diagnostic first:** encode→decode the 1441 real crops through the *stock* SD1.5
      `AutoencoderKL` and inspect reconstructions. If thin chromosome structures survive the f=8
      bottleneck intact, the VAE is not a bottleneck and this phase can be dropped entirely.
- [ ] If reconstruction is poor: **decoder-only fine-tune of SD1.5's `AutoencoderKL`** on cell
      crops. This leaves the latent space untouched, so the UNet and every existing LoRA remain
      valid. The existing `VQModel` training setup is a reference for how to run this on this
      domain; the weights themselves are not reusable here.
- [ ] Wire `--vae` through `train_lora.py` for the `sd15` family (currently SD3-only) and
      **invalidate the cached latents** — `cache_latents_to_disk: true` leaves stale `.npz` files
      beside the images that would be silently reused with the old VAE.

---

## Sequencing summary

| phase | output | GPU cost |
|---|---|---|
| 0 | phase classifier + conditioning diagnostic | ~2h |
| 1 | per-phase dataset layout, conditioning fixed | ~2h |
| 2 | metric module; p2 runs re-scored | ~2h |
| 3 | Optuna study, 40–60 trials | ~40h |
| 4 | downstream F1 curve on real test set | ~4h |
| 5 | agent wrapper | — |
| 6 | AutoencoderKL decoder fine-tune (deferred) | TBD |

The ordering principle: **the metric comes before the optimizer, and conditioning comes before
the metric.** Optimizing a distributional metric over a model that cannot distinguish the four
phases would produce beautiful, useless data.

---

## 2026-08-18 — study complete, and the objective is measurably wrong

`lora_nofill_v1` finished: **30 COMPLETE, 2 FAIL** (trial_003 crashed 2026-08-16; trial_021 lost
its evaluation to the power-off, its `.safetensors` survives and was never re-scored).

Backfilled the two diversity statistics across all 30 trials + 5 hand-built runs from cached
samples. **The composite score is +0.898 correlated with `nn_self_similarity_ratio`** — 8 of the
top 10 trials template, 0 of the bottom 10 do. `phase_consistency` is the driver (+0.919).

This is the predictable failure of optimising a classifier-based proxy: "is this obviously
phase X" is maximised by emitting a few textbook prototypes, which is exactly the opposite of
what a data-augmentation generator is for. The user saw it by eye in trial_016 before we could
measure it.

`coverage` was supposed to be the guard and is not: it correlates **+0.858** with templating.
Coverage asks whether each real sample has a generated neighbour; prototypes parked in dense
regions satisfy that. **Mode collapse and prototype collapse are different failures** and the
metric stack only guarded the first.

Templating is an over-fitting signature: corr with `learning_rate` +0.569, `alpha_ratio` +0.393,
`max_train_steps` +0.388, `network_dim` −0.096.

**Deliberate decision: do NOT add a diversity term and re-run the study yet.** That would burn
~40h optimising against a proxy whose downstream value is still unmeasured, and would break
comparability with these 30 trials. Phase 4 has never run and is the only thing that can say
whether templating actually costs classifier accuracy. Run Phase 4 first, then decide.

**Phase 4 arms** (real held-out test crops only, evaluator arch != efficientnet_b2):
1. real only (control)
2. real + classical augmentation (mandatory — the baseline synthetic data must beat)
3. real + `trial_020` synthetic (max composite, heavy templating, nn 1.037)
4. real + `trial_012` synthetic (best clean run, nn 0.947, effR 0.855)

Arms 3 vs 4 are the experiment that matters: if the templated generator still wins downstream,
templating is tolerable and the current objective survives. If arm 4 wins, or if arm 2 beats
both, the objective needs the diversity term before any further HPO.

---

## 2026-08-18 — PHASE 4 RESULT: synthetic data helps, but on a weaker model than the deliverable

Ran `validate_synthetic_downstream.py` (resnet50 evaluator, 3 seeds, 500 samples/phase, no
judge-filter) on trial_020 (templated) and trial_012 (clean). Real held-out test crops only.

| ratio | trial_020 | trial_012 |
|---|---|---|
| 0.0 (real + classical aug) | 0.7410 | 0.7410 |
| 0.25 | 0.8044 | 0.7963 |
| 0.5 | 0.8184 | **0.8395** |
| 1.0 | 0.7916 | 0.7915 |
| 2.0 | 0.8256 | **0.8441** |

**1. Synthetic data beats classical augmentation, clearly.** Every non-zero ratio beats the
control; best is +0.103 macro-F1. The control already includes hflip/vflip/color_jitter, so this
is a gain *over* classical augmentation, not over nothing. Anaphase — the standing weakness —
goes 0.664 -> 0.850, the largest single gain.

**2. Templating did NOT measurably hurt.** trial_012 leads by +0.008 averaged over non-zero
ratios and +0.019 at the best ratio, against a seed std of ~0.02. Directionally it favours
diversity at every ratio where the gap exceeds noise (0.5 and 2.0), but this is a 1-sigma effect
on 118 test crops. **The objective does not need an emergency rewrite**; the diversity term
remains worth adding, on this evidence as a tie-break rather than a correction.

**3. THE LIMITATION THAT MATTERS.** The evaluator had to differ from the judge to avoid
circularity, so it is resnet50 — which reaches only **0.741** on real data. The deliverable
phase classifier, efficientnet_b2, already reaches **0.906 macro-F1 on real data with no
synthetic at all**. The entire demonstrated gain (0.741 -> 0.844) sits *below* what the
deliverable already achieves unaided.

So Phase 4 shows synthetic data can lift a weak classifier. It does **not** show it lifts a
strong one, and that is the question the project actually needs answered. The gain may be
resnet50 catching up to what better architecture/config already provides. (Note the two numbers
also differ in training config, not only architecture, so they are not a clean A/B.)

**Required follow-up before acting on any of this: replicate on efficientnet_b1** — not the
judge, so not circular, but close enough in capacity to give a real-data baseline near 0.9. If
synthetic still helps there, the result transfers to the deliverable. If it vanishes, the gain
was headroom, not information.

**4. Ratio 2.0 is confounded.** Only 500 samples/phase were generated, so prophase capped at
1.41x while anaphase got the full 2.0x — ratio 2.0 partially rebalances toward minority classes
rather than scaling uniformly. Some of the anaphase gain is rebalancing, not synthetic quality.
Regenerate with `--samples 750` to test 2.0 cleanly.

**5. The dip at ratio 1.0** (0.7916 / 0.7915) is not a bug — per-seed values differ across the
two arms and the means coincide by chance. It is within noise, but it means the ratio curve is
not monotone and 0.5 / 2.0 should not be read as "more is better".

---

## 2026-08-18 — FIVE-CLASS RESULT (the deliverable task): synthetic data helps, significantly

New script `scripts/validate_synthetic_5class.py`. Task: prophase/metaphase/anaphase/telophase +
**interphase**, efficientnet_b2, trial_020 synthetic added to train's mitotic classes only.

| ratio | macro-F1 | acc | anaphase | interphase | metaphase | prophase | telophase |
|---|---|---|---|---|---|---|---|
| 0 (real only) | 0.7301 ±0.036 | 0.880 | 0.753 | 0.933 | 0.682 | **0.460** | 0.822 |
| **0.5** | **0.7969** ±0.038 | 0.930 | 0.796 | 0.964 | 0.740 | **0.593** | 0.891 |
| max (500/phase) | 0.7761 ±0.014 | 0.912 | 0.791 | 0.954 | 0.765 | 0.505 | 0.866 |

**Paired by seed** (same seed = same init and data order, so the arms pair):

* ratio 0.5: diffs +0.0767 / +0.0545 / +0.0692 -> **mean +0.0668, sd 0.0113, t=10.3 (df=2)**.
  All three seeds positive and tightly clustered. The unpaired std (±0.036) badly understates
  the evidence because it is dominated by a shared seed effect that cancels in the pairing.
* ratio max: +0.0869 / +0.0498 / +0.0014 -> mean +0.0460, sd 0.0429, t=1.86. Positive but noisy;
  **0.5 beats max**, matching the resnet50 4-class finding that more synthetic is not better.

**Prophase is the new headline weakness, not anaphase.** In the 4-class task anaphase was worst
(0.66) and prophase was fine (0.77). Adding interphase inverts this: prophase collapses to
**0.460** because early prophase genuinely resembles interphase -- condensing chromatin is a
continuum, not a boundary. Synthetic data recovers the most ground exactly there (+0.133).
Anaphase, the old problem, is now mid-table. Any further generator work should target prophase.

### Dataset construction (`datasets/crops/phase5_classifier/`, symlinks)

Interphase comes from COCO `attributes.division == 0` via `classifier_dataset.py` (40 710
crops). Two hazards handled:

1. **Cross-dataset split integrity.** Phase crops were split group-aware; interphase crops carry
   their own per-crop COCO splits. Interphase is re-assigned here: md5-deduplicated, grouped by
   source micrograph with the same `group_key`, and any group already claimed by the phase
   manifest **inherits that split**. 293 of 936 interphase groups (31%) overlap phase groups --
   which is also reassuring against a source-artifact confound, since the two classes are not
   drawn from disjoint image populations. Zero md5 and zero group overlap across splits, asserted.
2. **Prior consistency.** Interphase is subsampled to **8 per mitotic crop in every split**
   (train 7762, val 972, test 979). Capping train alone would train under one class prior and
   test under another, so the arms would differ by a calibration artifact. Cost: absolute F1 is
   optimistic vs the true 34:1 problem; the arm comparison is unaffected since every arm sees the
   identical interphase subsample.

### Caveats

* **efficientnet_b2 is the judge architecture**, used deliberately here because it is the
  deliverable. trial_020's generations were selected on a `phase_consistency` score b2 produced,
  so b2 plausibly benefits more than a neutral architecture would. The resnet50 4-class numbers
  remain the unbiased reference; both point the same way. The interphase class is uncontaminated
  either way -- the 4-class judge never saw it.
* Only trial_020 was tested. trial_012 (the clean, non-templating run) was not run in 5-class.
* Interphase labels inherit COCO annotation quality: an unannotated mitotic cell is silently an
  interphase label.
