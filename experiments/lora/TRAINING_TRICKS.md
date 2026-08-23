# LoRA Training Tricks — Phase 2 Ablation

The p2 sweep keeps all hyperparameters fixed (rank 256, α 128, 2000 steps, AdamW8bit, cosine LR,
aug2x dataset) and ablates three optional noise-shaping flags to identify which ones help on
microscopy data.

Fixed base command (all experiments share this skeleton):

```
accelerate launch scripts/vendor/sd-scripts/train_network.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --network_module=networks.lora \
  --network_dim=256 \
  --network_alpha=128 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=15 \
  --max_train_steps=2000 \
  --learning_rate=1e-5 \
  --lr_scheduler=cosine \
  --optimizer_type=AdamW8bit \
  --mixed_precision=fp16 \
  --enable_bucket \
  --cache_latents \
  --cache_latents_to_disk \
  ...
```

The three tricks below are added on top of this base.

---

## 1. `noise_offset`

**Config key:** `training.noise_offset: 0.05`
**Flag added:** `--noise_offset=0.05`

### What problem it solves

SD 1.x trains on a noise schedule that never produces a fully-black or fully-white latent at
the noisiest timestep (T=1000). In practice this means the model is never trained to generate
images at the extremes of the luminance range, so it tends to produce washed-out results on
subjects that should have very dark backgrounds or very bright highlights.

Microscopy images are a classic case: dark background, bright cell structures. Without
noise_offset the LoRA can generate cells that look correct in shape but muddy in contrast.

### Mechanism

At each training step, a small constant `δ` (the offset) is added to the noise **ε** before
it is mixed with the clean latent:

```
ε_shifted = ε + δ · N(0, 1)
```

This shifts the effective noise distribution so the model occasionally sees near-black and
near-white conditions during training, forcing it to learn to reconstruct across the full
luminance range.

### Effect on the command

```diff
  accelerate launch scripts/vendor/sd-scripts/train_network.py \
    ... (base args) ...
+   --noise_offset=0.05
```

### When to use it

Recommended for any domain where the subject has high contrast or occupies only part of the
brightness range. A value of 0.05 is mild; 0.1 is stronger and may reduce saturation.

---

## 2. `min_snr_gamma`

**Config key:** `training.min_snr_gamma: 5.0`
**Flag added:** `--min_snr_gamma=5.0`

### What problem it solves

In the standard diffusion training objective each timestep contributes equally to the loss.
But high-noise timesteps (low signal-to-noise ratio, low SNR) produce very large loss values
even though they carry almost no structural information — the network is just learning to
denoise pure noise. This imbalance wastes capacity: the model spends too much time on
low-SNR timesteps and converges more slowly on the high-SNR timesteps that actually encode
details like cell boundaries and chromatin texture.

### Mechanism

Min-SNR-γ (Hang et al., 2023) reweights the per-timestep loss by capping the weight at a
maximum of γ relative to the natural SNR:

```
w(t) = min(SNR(t), γ) / SNR(t)
```

At high-noise timesteps, `SNR(t)` is small → the cap kicks in and the weight is reduced.
At low-noise timesteps (where SNR > γ), the weight stays at its natural value of 1.

With γ = 5 (the recommended default from the paper) the loss landscape becomes more
uniform across timesteps, the gradient signal is dominated by medium-to-low-noise steps,
and training tends to converge faster with better detail fidelity.

### Effect on the command

```diff
  accelerate launch scripts/vendor/sd-scripts/train_network.py \
    ... (base args) ...
+   --min_snr_gamma=5.0
```

### Effect on training loss

Because low-SNR loss values are down-weighted, the raw `avg_loss` number will appear lower
than the baseline even if quality is identical. Compare experiments by visual quality and
`final_loss` (the last step), not `avg_loss` across the run.

### When to use it

Almost always beneficial for fine-tuning on small datasets. The p2 sweep results show it
cuts avg_loss by ~35 % vs baseline (`p2_min_snr`: 0.0747 vs `p2_baseline`: 0.1148).

---

## 3. `ip_noise_gamma`

**Config key:** `training.ip_noise_gamma: 0.1`
**Flag added:** `--ip_noise_gamma=0.1`

### What problem it solves

When fine-tuning on a small dataset (ours has ~2800 images), the LoRA can memorise the
exact latent representation of each training image. This shows up as outputs that look like
degraded copies of training images rather than novel generations. ip_noise_gamma is a form
of latent-space data augmentation that breaks this exact memorisation.

### Mechanism

Before the diffusion noise is added, a small independent noise vector is mixed into the
**clean latent** z₀:

```
z₀_perturbed = z₀ + γ_ip · ε_independent
```

where `ε_independent` is drawn fresh for each sample. The network must now reconstruct the
original z₀ from a slightly corrupted starting point, which prevents it from learning
pixel-exact reconstructions and encourages more generalised feature representations.

This is distinct from noise_offset: noise_offset shifts the noise added during the forward
process; ip_noise_gamma perturbs the latent itself before that process starts.

### Effect on the command

```diff
  accelerate launch scripts/vendor/sd-scripts/train_network.py \
    ... (base args) ...
+   --ip_noise_gamma=0.1
```

### When to use it

Useful on small datasets or when you suspect memorisation. On its own (`p2_ip_noise`) it
increases avg_loss vs baseline (0.123 vs 0.115), which reflects the harder task — the model
is denoising a perturbed latent, not the clean one. This is not necessarily bad; it can
improve generation diversity at the cost of slightly higher training loss.

---

## Experiment Matrix

| Experiment       | `noise_offset` | `min_snr_gamma` | `ip_noise_gamma` | avg_loss | final_loss |
|:-----------------|:--------------:|:---------------:|:----------------:|:--------:|:----------:|
| p2_baseline      | —              | —               | —                | 0.1148   | 0.0447     |
| p2_noise_offset  | 0.05           | —               | —                | 0.1164   | 0.0462     |
| p2_min_snr       | —              | 5.0             | —                | 0.0747   | 0.0445     |
| p2_ip_noise      | —              | —               | 0.1              | 0.1230   | 0.0547     |
| p2_noise_min_snr | 0.05           | 5.0             | —                | 0.0761   | 0.0461     |
| p2_noise_ip      | 0.05           | —               | 0.1              | 0.1247   | 0.0554     |
| p2_full_tricks   | 0.05           | 5.0             | 0.1              | 0.0841   | 0.0553     |

### Reading the results

- **`min_snr_gamma` dominates**: it cuts avg_loss by ~35 % on its own and does not hurt final_loss.
- **`noise_offset` alone is neutral**: avg_loss is almost identical to baseline; it may help
  perceptually on high-contrast subjects without showing up in the loss.
- **`ip_noise_gamma` raises training loss** (harder task) while potentially improving generation
  diversity — assess it visually, not by loss alone.
- **Combining all three** (`p2_full_tricks`) underperforms `p2_min_snr` on both metrics,
  suggesting `ip_noise_gamma` adds noise that partially cancels the benefit of `min_snr_gamma`
  when used together.

### Recommendation

Start with `min_snr_gamma=5` alone. Add `noise_offset=0.05` if generated images look
low-contrast. Add `ip_noise_gamma=0.1` only if you observe memorisation in outputs.
