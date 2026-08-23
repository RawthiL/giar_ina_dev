"""
Distributional and memorisation metrics for generated mitotic-cell crops.

Design notes, from thoughts/shared/plans/2026-08-09-phase-conditional-lora-hpo.md:

* **KID, not FID.** KID is an unbiased MMD estimator and stays usable at the few-hundred-sample
  scale these experiments run at. FID's covariance term is rank-deficient below the feature
  dimension and is badly biased at small n.
* **Domain features, not Inception.** ImageNet features are near-blind to grayscale microscopy.
  The primary feature space is the trained phase classifier's backbone; the cell-domain VQGAN
  encoder provides a second, texture-oriented view.
* **Distributional metrics cannot see memorisation.** A LoRA that copies its training set scores
  excellent KID. `memorization_score` and `coverage_density` exist to catch that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Kernel Inception Distance (feature-space agnostic)
# ---------------------------------------------------------------------------


def _poly_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Polynomial kernel k(x,y) = (x·y + 1)^3 over **L2-normalised** features.

    Canonical KID uses (x·y/d + 1)^3 on raw Inception pool features, whose scale is well
    behaved. Ours are not: on 2026-08-10 a single degenerate generated crop had a feature norm
    of 2566 against a median of 13.7 (188x), the cubic turned that into ~1e11 in the k(y,y)
    block, and because the unbiased estimator removes the k(y,y) diagonal but has no diagonal
    to remove from k(x,y), the outlier landed asymmetrically and drove MMD^2 to **-35**. That
    trial then topped the Optuna study on a score of 9.04 with a phase_consistency of 0.41.

    Normalising first bounds every dot product to [-1, 1] and the kernel to [0, 8], so no
    single sample can dominate. Values are no longer comparable with the pre-fix run.
    """
    return (x @ y.T + 1.0) ** 3


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-8, None)


def kid(
    feat_real: np.ndarray,
    feat_fake: np.ndarray,
    n_subsets: int = 100,
    subset_size: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Kernel Inception Distance: mean and std of the unbiased MMD^2 over random subsets.

    Returns (mean, std). Lower is better; can go slightly negative, which is normal for an
    unbiased estimator when the two distributions are very close.
    """
    n = min(len(feat_real), len(feat_fake))
    if n < 4:
        return float("nan"), float("nan")
    subset_size = min(subset_size or n, n)

    feat_real = _l2_normalize(feat_real)
    feat_fake = _l2_normalize(feat_fake)

    rng = np.random.default_rng(seed)
    m = subset_size
    values: list[float] = []
    for _ in range(n_subsets):
        x = feat_real[rng.choice(len(feat_real), m, replace=False)]
        y = feat_fake[rng.choice(len(feat_fake), m, replace=False)]

        kxx, kyy, kxy = _poly_kernel(x, x), _poly_kernel(y, y), _poly_kernel(x, y)
        # Unbiased: drop the diagonal from the within-set terms.
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        values.append(kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean())

    return float(np.mean(values)), float(np.std(values))


# ---------------------------------------------------------------------------
# Fidelity / diversity (Naeem et al., "Reliable Fidelity and Diversity Metrics")
# ---------------------------------------------------------------------------


def coverage_density(
    feat_real: np.ndarray, feat_fake: np.ndarray, k: int = 5, seed: int = 0
) -> tuple[float, float]:
    """
    Returns (coverage, density).

    coverage — fraction of real samples with at least one generated sample inside their
               k-NN radius. Falls when the generator collapses onto part of the distribution.
    density  — how densely generated samples pack those neighbourhoods; >1 means they
               concentrate more tightly than the real data does.

    **Sample sizes are equalised by subsampling the larger set.** Coverage is biased downward
    when n_fake < n_real — with 100 generated against 442 real, a perfect generator still
    cannot light up every real neighbourhood — so leaving them unequal would make the metric
    depend on how many real crops a phase happens to have, and thus incomparable across phases.
    """
    if len(feat_real) <= k or len(feat_fake) == 0:
        return float("nan"), float("nan")

    n = min(len(feat_real), len(feat_fake))
    if n <= k:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if len(feat_real) > n:
        feat_real = feat_real[rng.choice(len(feat_real), n, replace=False)]
    if len(feat_fake) > n:
        feat_fake = feat_fake[rng.choice(len(feat_fake), n, replace=False)]

    real = torch.from_numpy(feat_real).float()
    fake = torch.from_numpy(feat_fake).float()

    # Radius of each real point's k-th nearest *real* neighbour (excluding itself).
    d_rr = torch.cdist(real, real)
    radii = d_rr.kthvalue(k + 1, dim=1).values  # +1 skips the zero self-distance

    d_rf = torch.cdist(real, fake)  # (n_real, n_fake)
    inside = d_rf <= radii.unsqueeze(1)

    coverage = inside.any(dim=1).float().mean().item()
    density = inside.sum().item() / (k * len(fake))
    return float(coverage), float(density)


# ---------------------------------------------------------------------------
# Memorisation
# ---------------------------------------------------------------------------


def memorization_score(
    feat_train: np.ndarray, feat_fake: np.ndarray, percentile: float = 95.0
) -> dict[str, float]:
    """
    Cosine similarity from each generated sample to its nearest training image.

    Reported at a high percentile rather than the mean: a handful of near-copies is the
    failure mode that matters, and a mean would dilute them away.
    """
    if len(feat_train) == 0 or len(feat_fake) == 0:
        return {"nn_p95": float("nan"), "nn_max": float("nan"), "nn_mean": float("nan")}

    train = torch.from_numpy(feat_train).float()
    fake = torch.from_numpy(feat_fake).float()
    train = train / train.norm(dim=1, keepdim=True).clamp_min(1e-8)
    fake = fake / fake.norm(dim=1, keepdim=True).clamp_min(1e-8)

    nearest = (fake @ train.T).max(dim=1).values.numpy()
    return {
        "nn_p95": float(np.percentile(nearest, percentile)),
        "nn_max": float(nearest.max()),
        "nn_mean": float(nearest.mean()),
    }


def nn_self_similarity(features: np.ndarray, n_sample: int = 200, seed: int = 0) -> float:
    """
    Mean leave-one-out *nearest-neighbour* similarity within one set — its templating score.

    Prefer this to `mean_pairwise_similarity`, which was tried first on 2026-08-16 and failed:
    across trial_016 (visibly 3-4 templates per phase), p5_noaug and a poor run, it returned
    0.912 / 0.910 / 0.918 — no discrimination whatsoever. Two reasons, both instructive:

    1. **Templating is local clustering, so the statistic must be local.** With 4 templates x 25
       images each, every image has ~24 near-twins, but averaging over all pairs is dominated by
       the cross-template pairs and dilutes them away. This is exactly the argument already made
       in `memorization_score`'s docstring, which reports a high percentile rather than a mean.
    2. **VQGAN cosine similarities are saturated** — real crops sit at 0.949-0.994 and everything
       is near-parallel, leaving a mean almost no dynamic range.

    A near-twin still stands out under a max even in a saturated space, which a mean cannot.
    """
    if len(features) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(features) > n_sample:
        features = features[rng.choice(len(features), n_sample, replace=False)]

    x = torch.from_numpy(features).float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = x @ x.T
    sim.fill_diagonal_(-1.0)  # leave-one-out
    return float(sim.max(dim=1).values.mean().item())


def effective_rank(features: np.ndarray, n_sample: int = 200, seed: int = 0) -> float:
    """
    How many dimensions the set actually spans, via the participation ratio of its covariance
    spectrum: `(sum lambda)^2 / sum(lambda^2)`.

    Complements `nn_self_similarity`: that one asks "does each sample have a twin", this asks
    "how many independent modes does the whole set occupy". A few templates give a low effective
    rank however the individual pairs happen to fall.
    """
    if len(features) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(features) > n_sample:
        features = features[rng.choice(len(features), n_sample, replace=False)]

    centred = features - features.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.svd(centred, compute_uv=False) ** 2
    total = eigenvalues.sum()
    if total <= 0:
        return float("nan")
    return float(total**2 / (eigenvalues**2).sum())


def mean_pairwise_similarity(features: np.ndarray, n_sample: int = 200, seed: int = 0) -> float:
    """
    Mean cosine similarity between distinct members of one set — how self-similar it is.

    Every other metric in this stack compares generations to something *else* (real crops for
    KID and coverage, training crops for memorisation). None of them measures how different the
    generations are from each other, and on 2026-08-16 that gap showed: trial_016 scored the best
    kid_classifier (0.607) and the highest coverage (0.550) ever measured, while visual inspection
    of its 100 samples per phase found only 3-4 recurring templates with variations.

    A generator emitting a few templates can pass the existing guards: if the templates land in
    dense regions of the real distribution, coverage counts many real neighbourhoods as covered
    and KID only compares aggregate statistics.
    """
    if len(features) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(features) > n_sample:
        features = features[rng.choice(len(features), n_sample, replace=False)]

    x = torch.from_numpy(features).float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = x @ x.T
    n = len(x)
    # Exclude the diagonal: every vector is trivially similar to itself.
    off_diagonal_sum = sim.sum().item() - sim.diagonal().sum().item()
    return float(off_diagonal_sum / (n * (n - 1)))


def real_to_real_baseline(
    feat_train: np.ndarray, n_sample: int = 500, percentile: float = 95.0, seed: int = 0
) -> dict[str, float]:
    """
    Leave-one-out nearest-neighbour similarity *within* the real training set.

    Without this, `memorization_score` is uninterpretable: an nn_p95 of 0.83 means nothing until
    you know how similar two genuinely different real cells are. Distinct real crops of the same
    phase are already highly similar in a phase-classifier feature space, so the question is
    never "is nn_p95 high" but "is it higher than real-vs-real".

    This LoRA trains on every real crop, so there is no held-out real set to compare against —
    real-vs-real is the only available control.
    """
    if len(feat_train) < 2:
        return {"nn_p95": float("nan"), "nn_max": float("nan"), "nn_mean": float("nan")}

    rng = np.random.default_rng(seed)
    if len(feat_train) > n_sample:
        feat_train = feat_train[rng.choice(len(feat_train), n_sample, replace=False)]

    x = torch.from_numpy(feat_train).float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = x @ x.T
    sim.fill_diagonal_(-1.0)  # leave-one-out: never match an image with itself
    nearest = sim.max(dim=1).values.numpy()
    return {
        "nn_p95": float(np.percentile(nearest, percentile)),
        "nn_max": float(nearest.max()),
        "nn_mean": float(nearest.mean()),
    }


# ---------------------------------------------------------------------------
# VQGAN encoder as a second feature space
# ---------------------------------------------------------------------------


class VQGANFeatures:
    """
    Cell-domain VQGAN encoder used as a texture-oriented feature extractor and artifact detector.

    This is a `VQModel` (discrete codebook, f=4) fine-tuned on cell images. It is *not*
    compatible with the SD1.5 UNet and is never loaded into the generation pipeline — see the
    plan's "The VQGAN weights: verdict". Here it only ever runs as an encoder/decoder.

    Features are tapped at the **512-channel activation feeding `encoder.conv_out`**, global-
    average-pooled to 512 dims. Two tempting alternatives are both wrong: `conv_out`'s own
    output is only 4 channels, and anything post-quantisation has been snapped to one of 16384
    codes in 4-D, destroying nearly all the information.

    **Input range is [0,1], not the SD-standard [-1,1]** (fixed 2026-08-15). This checkpoint was
    trained on [0,1] and the mismatch was silently costing ~200x on the reconstruction metric:
    measured over 48 test crops at 256px, mean recon MSE is 0.0022 fed [0,1] against 0.423 fed
    [-1,1]. Because the error applied to real and generated crops alike, `vqgan_recon_ratio` and
    the `kid_vqgan` *rankings* were still usable, which is why this hid for so long — but the
    absolute `vqgan_recon_mse` was meaningless as an artifact detector, since it was dominated by
    the normalisation offset rather than by artifacts.

    Two consequences for stored numbers:
      * `vqgan_recon_mse*` written before this date are on the old scale and are not comparable
        to new ones (the `..._real` baseline of 0.349327 is the giveaway).
      * `kid_vqgan` is computed in this feature space, so its values shift too. It is a reported
        secondary metric only — `optimize_lora.py`'s `W_KID` weights `kid_classifier`, which is
        the phase-judge space and untouched here, so the Optuna objective does not move.
    """

    def __init__(self, weights_dir: Path, device: str | None = None, resolution: int = 512):
        from diffusers import VQModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        self.model = VQModel.from_pretrained(str(weights_dir)).to(self.device).eval()

        self._captured: torch.Tensor | None = None
        self.model.encoder.conv_out.register_forward_hook(self._hook)

    def _hook(self, _module, inputs, _output) -> None:
        self._captured = inputs[0]

    def _batch(self, paths: list[Path]) -> torch.Tensor:
        arrays = []
        for p in paths:
            img = (
                Image.open(p)
                .convert("RGB")
                .resize((self.resolution, self.resolution), Image.BICUBIC)
            )
            arrays.append(np.asarray(img, dtype=np.float32) / 255.0)  # this checkpoint wants [0,1]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
        return tensor.to(self.device)

    @torch.no_grad()
    def features(self, paths: list[Path], batch_size: int = 8) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            self.model.encode(self._batch(paths[i : i + batch_size]))
            assert self._captured is not None, "forward hook did not fire"
            out.append(self._captured.mean(dim=(2, 3)).float().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 512))

    @torch.no_grad()
    def reconstruction_error(self, paths: list[Path], batch_size: int = 8) -> dict[str, float]:
        """
        Per-image MSE of encode->decode through the cell-domain autoencoder.

        High error means the image contains structure a cell-trained autoencoder never learned
        to represent — i.e. generation artifacts rather than cell morphology.
        """
        errors: list[float] = []
        for i in range(0, len(paths), batch_size):
            batch = self._batch(paths[i : i + batch_size])
            recon = self.model.decode(self.model.encode(batch).latents).sample
            errors.extend(((recon - batch) ** 2).mean(dim=(1, 2, 3)).float().cpu().numpy().tolist())
        if not errors:
            return {"recon_mse_mean": float("nan"), "recon_mse_p95": float("nan")}
        return {
            "recon_mse_mean": float(np.mean(errors)),
            "recon_mse_p95": float(np.percentile(errors, 95)),
        }
