from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from allium_cepa_classifier.config.vae_config import VAEExperimentConfig
from allium_cepa_classifier.training.vae_model import VAE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) float tensor → (H, W) numpy array in [0, 1]."""
    return tensor.squeeze(0).cpu().numpy()


def _build_test_loader(test_dir: Path, image_size: tuple[int, int]) -> DataLoader:
    h, w = image_size
    tfm = transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(str(test_dir), transform=tfm)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=4), ds.classes, ds


# ---------------------------------------------------------------------------
# Plot 1: training curves
# ---------------------------------------------------------------------------


def plot_training_curves(history: dict, out: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, title in zip(
        axes,
        ["recon", "kl", "loss"],
        ["Reconstruction Loss", "KL Loss", "Total Loss"],
        strict=True,
    ):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: reconstructions
# ---------------------------------------------------------------------------


def plot_reconstructions(
    model: VAE,
    val_loader: DataLoader,
    device: torch.device,
    out: Path,
    n: int = 4,
) -> None:
    model.eval()
    imgs: list[torch.Tensor] = []
    with torch.no_grad():
        for x in val_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            imgs.append(x[:n])
            break
    batch = imgs[0][:n].to(device)

    with torch.no_grad():
        _, _, recon = model(batch)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    for i in range(n):
        axes[i, 0].imshow(_to_numpy(batch[i].cpu()), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(_to_numpy(recon[i].cpu()), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: random samples from prior
# ---------------------------------------------------------------------------


def plot_random_samples(
    model: VAE,
    device: torch.device,
    seed: int,
    out: Path,
    n: int = 100,
) -> None:
    model.eval()
    torch.manual_seed(seed)
    grid_side = int(n**0.5)

    with torch.no_grad():
        prior_mean = model.prior_mean.to(device)
        prior_std = (0.5 * model.prior_log_var.to(device)).exp()
        eps = torch.randn(n, prior_mean.shape[0], device=device)
        z = prior_mean + prior_std * eps
        imgs = model.decode(z).cpu()

    fig, axes = plt.subplots(grid_side, grid_side, figsize=(grid_side * 1.5, grid_side * 1.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(_to_numpy(imgs[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.suptitle("Random samples from prior", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: t-SNE of test set latents
# ---------------------------------------------------------------------------


def plot_tsne_test_latents(
    model: VAE,
    test_dir: Path,
    device: torch.device,
    seed: int,
    image_size: tuple[int, int],
    out: Path,
) -> None:
    model.eval()
    loader, class_names, _ = _build_test_loader(test_dir, image_size)

    z_means: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            zm, _, _ = model(x)
            z_means.append(zm.cpu().numpy())
            labels.extend(y.numpy().tolist())

    z_all = np.concatenate(z_means, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed)
    emb = tsne.fit_transform(z_all)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for idx, (cls, col) in enumerate(zip(class_names, colors, strict=False)):
        mask = np.array(labels) == idx
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[col], label=cls, s=10, alpha=0.7)
    ax.legend(markerscale=3)
    ax.set_title("t-SNE of test-set latents (z_mean)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: latent walk between phase centroids
# ---------------------------------------------------------------------------


def plot_latent_walk(
    model: VAE,
    test_dir: Path,
    device: torch.device,
    image_size: tuple[int, int],
    out: Path,
    steps: int = 10,
) -> None:
    model.eval()
    loader, class_names, ds = _build_test_loader(test_dir, image_size)
    class_names = ["prophase", "metaphase", "anaphase", "telophase"]  # Map the biological order

    # collect z_mean per phase
    per_class: dict[int, list[np.ndarray]] = {i: [] for i in range(len(class_names))}
    with torch.no_grad():
        for x, y in loader:
            zm, _, _ = model(x.to(device))
            for z_i, lbl in zip(zm.cpu().numpy(), y.numpy(), strict=True):
                per_class[int(lbl)].append(z_i)

    centroids = [np.mean(per_class[i], axis=0) for i in range(len(class_names))]

    # interpolate between consecutive centroids
    decoded: list[np.ndarray] = []
    n_segs = len(centroids) - 1
    for seg in range(n_segs):
        a, b = centroids[seg], centroids[seg + 1]
        for t in np.linspace(0, 1, steps, endpoint=False):
            z = torch.tensor(a + t * (b - a), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                img = model.decode(z)[0].cpu()
            decoded.append(_to_numpy(img))

    fig, axes = plt.subplots(n_segs, steps, figsize=(steps * 1.5, n_segs * 1.5))
    if n_segs == 1:
        axes = axes[np.newaxis, :]
    for row in range(n_segs):
        for col in range(steps):
            ax = axes[row, col]
            ax.imshow(decoded[row * steps + col], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_title(class_names[row], fontsize=8, loc="left")
        axes[row, steps - 1].set_title(class_names[row + 1], fontsize=8)

    plt.suptitle("Latent walk (centroid interpolation)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point called by run_training
# ---------------------------------------------------------------------------


def run_evaluation(
    model: VAE,
    history: dict,
    cfg: VAEExperimentConfig,
    run_dir: Path,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_training_curves(history, plots_dir / "training_curves.png")

    plot_reconstructions(model, val_loader, device, plots_dir / "reconstructions.png")

    plot_random_samples(model, device, cfg.data.seed, plots_dir / "random_samples.png")

    test_dir = cfg.data.vae_crops_dir / "test"
    if test_dir.exists():
        plot_tsne_test_latents(
            model,
            test_dir,
            device,
            cfg.data.seed,
            cfg.model.image_size,
            plots_dir / "tsne_test_latents.png",
        )
        plot_latent_walk(
            model,
            test_dir,
            device,
            cfg.model.image_size,
            plots_dir / "latent_walk.png",
        )
    else:
        import logging

        logging.getLogger(__name__).warning(
            f"Test dir {test_dir} not found — skipping t-SNE and latent walk plots."
        )
