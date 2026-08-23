# LoRA experiment leaderboard — rewritten 2026-08-16

> **The `lora_phase` / `lora_phase_v2` studies are VOID. Do not rank from them.**
> All 21 trials trained on `per_phase`, the dataset carrying the dark-red rotation wedges
> (`optimize_lora.py:150` hardcoded the version). The fix for that artifact is worth +0.085
> `phase_consistency` — larger than almost every gap those trials were ranking — and the study's
> headline finding **inverts** on clean data (see §2). The databases are kept for provenance only.

## 1. Current standings — `lora_nofill_v1` COMPLETE, 30/30 trials (2026-08-18)

`score = phase_consistency − 0.135·kid_classifier + 0.847·coverage − 5·max(0, memorization_excess)`

**Read §3b before using this ranking.** The composite is +0.898 correlated with templating, so
the top of this table is partly a ranking of which runs collapsed onto prototypes.

| run | score | phase_cons | kid_cls | coverage | nnSS | effR | flag |
|---|---|---|---|---|---|---|---|
| `trial_020` | **1.369** | 0.943 | **0.572** | **0.595** | 1.037 | 0.840 | templating |
| `trial_016` | 1.364 | **0.980** | 0.607 | 0.550 | 1.070 | 0.755 | templating |
| `trial_015` | 1.316 | 0.950 | 0.630 | 0.532 | 1.046 | 0.783 | templating |
| `trial_030` | 1.290 | 0.953 | 0.807 | 0.527 | 1.053 | 0.806 | templating |
| `trial_023` | 1.271 | 0.910 | 0.763 | 0.547 | 1.025 | 0.643 | templating |
| **`trial_012`** | 1.125 | 0.887 | 0.844 | 0.415 | **0.947** | **0.855** | **clean** |
| `p5_noaug` | 1.123 | 0.823 | 0.739 | 0.472 | 0.962 | 0.554 | collapse |
| `trial_013` | 1.040 | 0.838 | 0.913 | 0.385 | 0.929 | 0.716 | clean |

`trial_020` and `trial_016` are separated by 0.005 — far inside the 6.3% `phase_consistency`
noise. They are tied, not ranked.

**Best clean run: `trial_012`** — rank 256, α-ratio 1.0, lr 4.73e-5, 2000 steps,
caption_dropout 0.12. Highest `effective_rank_ratio` of any trained run and no templating,
at the cost of 0.06 `phase_consistency` and 0.18 `coverage` vs `trial_020`.

**Optuna parameter importances** (composite, 30 trials): `learning_rate` 0.337,
`use_noise_offset` 0.214, `te_lr_ratio` 0.173, `max_train_steps` 0.102,
`caption_dropout_rate` 0.086, `alpha_ratio` 0.059, `network_dim` **0.017**. Rank barely
matters once the red-wedge artifact is gone — consistent with §2, where rank only mattered
because it gated artifact memorisation.

Per-phase `phase_consistency`:

| experiment | prophase | metaphase | anaphase | telophase |
|---|---|---|---|---|
| `p3_per_phase_nofill` | 0.94 | 0.97 | 0.65 | 0.79 |
| `p5_noaug` | 0.90 | 0.94 | 0.63 | 0.82 |
| `p5_d4x3` | 0.92 | 0.97 | 0.58 | 0.68 |
| `p4_r16_nofill` | 0.99 | 0.96 | 0.65 | **0.44** |

### Reading these numbers

**Run-to-run noise, measured** (same config, two separate runs — `p3_per_phase` vs `trial_000`):

| metric | noise |
|---|---|
| `phase_consistency` | **6.3%** |
| `kid_classifier` | 1.8% |
| `coverage` | 1.5% |

So `phase_consistency` differences under ~0.05 mean nothing on single runs, while `kid_classifier`
and `coverage` are stable enough that a 40% move is real. Training is deterministic *sometimes*
(`aug2x_r256` and `p2_baseline` are bit-identical) but not reliably — fp16/cuDNN nondeterminism
over 2000 steps.

**KID reference points**, needed because KID has no absolute meaning:

| | `kid_classifier` |
|---|---|
| real vs real, same phase (floor) | **0.275** |
| real vs real, different phase | **6.21** |
| `p5_noaug` | 0.739 → **7.8%** of the way from real to wrong-phase |
| `p3_per_phase_nofill` | 1.273 → 16.8% |

**`vqgan_recon_ratio` targets ~1.0 in BOTH directions** — above is artifacts, below is
oversmoothing. Everything here is below, i.e. generations are smoother than real microscopy,
which has grain, dust and focus noise the generator flattens away.

## 2. The rank-16 headline inverted

`p4_r16_nofill` re-runs `trial_020`'s exact hyperparameters — the old study's winner at score
1.053 — on the fixed dataset:

| | rank 256 | rank 16 |
|---|---|---|
| phase_consistency | **0.838** | 0.760 |
| kid_classifier | **1.273** | 1.995 |
| composite | **0.954** | 0.747 |

Telophase carries the loss: 0.79 → 0.44, with 41% of it predicted as prophase (was 15%).

**Why:** the red-wedge artifact was a spurious global canvas that had to be either learned or
ignored, and fitting it costs capacity. Rank 256 had enough to memorise it; rank 16 did not. So
low rank won only because it was *too small to learn the artifact*. Remove the contamination and
capacity earns its keep again.

**Consequence: the next study must re-explore rank from scratch, not seed toward low rank.**

## 3. Offline augmentation was hurting

Three arms, identical hyperparameters, differing only in training data:

| arm | augmentation | images | composite | kid_cls | coverage | recon_ratio |
|---|---|---|---|---|---|---|
| **A `p5_noaug`** | **none** | 1194 | **1.123** | **0.739** | **0.472** | 0.448 |
| D `p5_d4x3_nojitter` | D4 lossless, no jitter | 4776 | 0.960 | 1.132 | 0.407 | 0.455 |
| C `p3_per_phase_nofill` | resample + jitter | 2388 | 0.954 | 1.273 | 0.340 | 0.444 |
| B3 `p5_d4x3` | D4 lossless + jitter | 4776 | 0.935 | 1.243 | 0.372 | **0.653** |

**Offline augmentation of any kind loses.** Even perfectly lossless D4 orientation copies (arm D)
lose to no copies at all. Likely mechanism: total exposure is fixed at ~30 000 images seen, so
each unique real crop is seen **25 times in A but only 6.3 times in D** — the copies spend
training budget teaching an orientation invariance SD1.5 already has.

The **brightness/contrast jitter (0.7–1.3) is what pushes generated colour statistics off the real
distribution** (D beats B3 on kid 1.243 → 1.132 and coverage 0.372 → 0.407), confirming the
2026-08-15 handoff's unresolved item 3 about oversaturated chromosomes.

**WITHDRAWN — an earlier reading of the first three arms only.** B3's `recon_ratio` recovery
(0.444 → 0.653) was attributed to removing `rotate_without_fill`'s resampling. Arm D has no
resampling *and* no jitter and reads 0.455, not 0.653. The effects are additive and opposite —
jitter raises `recon_ratio` ~+0.2, resampling lowers it ~−0.2 — and they cancel in arm C. B3's
0.653 came mostly from the jitter. **Oversmoothing is intrinsic to the generator** (~0.45 with no
augmentation at all), not an augmentation artifact.

## 3b. Intra-set diversity — THE OBJECTIVE SELECTS FOR TEMPLATING (measured 2026-08-18)

Backfilled `nn_self_similarity_ratio` / `effective_rank_ratio` across all 30 completed trials
plus the five hand-built runs, from cached samples (no regeneration). Correlations over n=30:

| | vs composite |
|---|---|
| `nn_self_similarity_ratio` | **+0.898** |
| `effective_rank_ratio` | +0.555 |
| `nn_ratio` vs `phase_consistency` | **+0.919** |
| `nn_ratio` vs `coverage` | +0.858 |

**8 of the top 10 trials template; 0 of the bottom 10 do.** This is not a coincidence to be
explained away — `phase_consistency` asks a classifier "is this obviously phase X", and the
surest way to score is to emit a few textbook-perfect prototypes. Optuna found that. The user
independently spotted it by eye in `trial_016` (~3–4 recurring cells per 100 images) *before*
the statistic existed.

`coverage` did **not** catch it (+0.858, i.e. it moves *with* templating). Coverage asks whether
real samples have a generated neighbour; a few prototypes parked in high-density regions of the
real manifold satisfy that. It guards mode collapse, not prototype collapse — those are
different failures and we had a guard for only one.

What drives templating (corr with `nn_ratio`): `learning_rate` **+0.569**, `alpha_ratio` +0.393,
`max_train_steps` +0.388, `network_dim` −0.096. It is an over-fitting signature — high LR, high
effective LR via alpha, long training — not a capacity effect.

Two dead ends recorded so they are not retried: `mean_pairwise_similarity` does not discriminate
(0.912/0.910/0.918 — templating is local clustering, needs a nearest-neighbour statistic, not a
mean), and **VQGAN features are useless for diversity** (effective rank ~1.2, a single dimension).

Minor: ad-hoc values quoted on 2026-08-17 (`trial_016` 1.053, `p5_noaug` 0.935) differ slightly
from the registered metric (1.070, 0.962) — the metric subsamples 200 points per phase under a
fixed seed. Ordering and conclusions are unchanged; the registered values are canonical.

## 4. Open

- **Anaphase is the standing weakness** at 0.63–0.65 across every clean-data run, against a judge
  ceiling of 0.86 (the judge's own recall on real anaphase). It loses ~33% of samples to
  metaphase. The border artifact was *a* cause, not *the* cause; metaphase/anaphase is a genuine
  biological continuum.
- **DONE 2026-08-18 — and it did not survive cleanly.** trial_016 keeps its 1.364 but is
  confirmed templating (nn 1.070); so is the new leader trial_020. See §3b.
- **The objective needs a diversity term before any further HPO.** Adding one now would break
  comparability with these 30 trials, so it is deliberately deferred until Phase 4 says how much
  templating actually costs downstream. Do not run another study on the current objective.
- **Oversmoothing is intrinsic** (`recon_ratio` ~0.45 with no augmentation at all), not an
  augmentation artifact. Jitter raises it ~+0.2, resampling lowers it ~-0.2; they cancelled in
  arm C. Synthetic crops lack real microscopy's grain — a Phase 4 concern needing its own fix.
- **Phase 4 downstream validation** has not run. It is the only thing that answers whether any of
  this helps the actual classifier. A classical-augmentation arm is mandatory there.

## 5. Commands

```bash
# summarise a study (also prints its pinned dataset_version)
uv run python scripts/optimize_lora.py --study-name <name> --report

# resume after a shutdown; --n-trials is a TARGET TOTAL, not an increment
uv run python scripts/optimize_lora.py --study-name <name> --n-trials 30 --reset-stale

# re-score cached samples after any judge or metric change (no regeneration)
uv run python scripts/evaluate_lora.py --config experiments/lora/<name>/config.yaml --samples 100
```
