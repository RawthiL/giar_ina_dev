from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from allium_cepa_classifier.config.sd_vae_config import SDVAEExperimentConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).cpu().numpy()


def get_validation_samples(cfg: SDVAEExperimentConfig, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    """Return curated validation samples: one per phase, or explicit paths if configured."""
    from PIL import Image as PILImage

    transform = transforms.Compose([
        transforms.Resize((cfg.model.resolution, cfg.model.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    base_dir = cfg.data.vae_crops_dir
    paths: list[Path] = []
    names: list[str] = []

    if cfg.validation.images:
        # Resolve explicit paths relative to vae_crops_dir
        for p in cfg.validation.images:
            path = base_dir / p if not p.is_absolute() else p
            paths.append(path.resolve())
            # phase name from parent folder
            names.append(path.parent.name)
    else:
        # Auto-pick one per phase from val/tagged
        val_tagged = base_dir / "val" / "tagged"
        if val_tagged.exists():
            subdirs = sorted([d for d in val_tagged.iterdir() if d.is_dir()])
            for d in subdirs:
                # pick first image
                imgs = sorted([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}])
                if imgs:
                    paths.append(imgs[0])
                    names.append(d.name)
        # fallback to generic val if no tagged
        if not paths:
            val_dir = base_dir / "val"
            # pick first 4 images
            imgs = sorted([f for f in val_dir.rglob("*") if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:4]
            for p in imgs:
                paths.append(p)
                names.append(p.parent.name)

    if not paths:
        raise ValueError("No validation images found for SD VAE")

    tensors = []
    for p in paths:
        with PILImage.open(p).convert("RGB") as img:
            tensors.append(transform(img))

    batch = torch.stack(tensors).to(device)
    return batch, names


def _build_test_loader(test_dir: Path, resolution: int) -> tuple[DataLoader, list[str]]:
    tfm = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.ImageFolder(str(test_dir), transform=tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    return loader, ds.classes


def plot_training_curves(history: dict, out: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, key, title in zip(
        axes,
        ["recon", "kl", "loss"],
        ["Reconstruction Loss", "KL Loss", "Total Loss"],
    ):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_reconstructions(model, cfg: SDVAEExperimentConfig, device: torch.device, out: Path) -> None:
    model.eval()
    batch, names = get_validation_samples(cfg, device)
    
    with torch.no_grad():
        posterior = model.encode(batch)
        z = posterior.latent_dist.sample()
        recon = model.decode(z).sample
    
    # Denormalize for display
    def denorm(t):
        return t * 0.5 + 0.5
    
    n = batch.size(0)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for i in range(n):
        orig = denorm(batch[i].cpu())
        recon_i = denorm(recon[i].cpu())
        axes[i, 0].imshow(orig.permute(1, 2, 0).clamp(0, 1))
        axes[i, 0].set_title(f"Original – {names[i]}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(recon_i.permute(1, 2, 0).clamp(0, 1))
        axes[i, 1].set_title(f"Reconstructed – {names[i]}")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_random_samples(model, device: torch.device, seed: int, out: Path, n: int = 64) -> None:
    model.eval()
    torch.manual_seed(seed)
    grid_side = int(n ** 0.5)
    
    with torch.no_grad():
        # Sample from standard normal (SD VAE latent is standard normal)
        # SD VAE latent shape: (batch, channels, h, w) -> channels=4, h=w=resolution//8
        # For simplicity, sample from N(0,1) in the right shape
        # Get latent shape from a dummy forward
        dummy = torch.zeros(1, 3, model.config.sample_size, model.config.sample_size).to(device)
        posterior = model.encode(dummy)
        latent_shape = posterior.latent_dist.mean.shape
        
        eps = torch.randn(latent_shape[0] * n, *latent_shape[1:], device=device)
        z = eps
        imgs = model.decode(z).sample.cpu()
    
    def denorm(t):
        return t * 0.5 + 0.5
    
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(grid_side * 1.5, grid_side * 1.5))
    for i, ax in enumerate(axes.flat):
        img = denorm(imgs[i])
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        ax.axis("off")
    
    plt.suptitle("Random samples from prior", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_tsne_test_latents(model, test_dir: Path, device: torch.device, seed: int, resolution: int, out: Path) -> None:
    model.eval()
    loader, class_names = _build_test_loader(test_dir, resolution)
    
    z_means = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            posterior = model.encode(x)
            mean = posterior.latent_dist.mean
            # Flatten latent
            mean_flat = mean.view(mean.size(0), -1).cpu().numpy()
            z_means.append(mean_flat)
            labels.extend(y.numpy().tolist())
    
    z_all = np.concatenate(z_means, axis=0)
    # Subsample for t-SNE speed
    if z_all.shape[0] > 1000:
        idx = np.random.RandomState(seed).choice(z_all.shape[0], 1000, replace=False)
        z_all = z_all[idx]
        labels = np.array(labels)[idx].tolist()
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed)
    emb = tsne.fit_transform(z_all)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for idx, (cls, col) in enumerate(zip(class_names, colors)):
        mask = np.array(labels) == idx
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[col], label=cls, s=10, alpha=0.7)
    ax.legend(markerscale=3)
    ax.set_title("t-SNE of test-set latents (z_mean)")
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_latent_walk(model, test_dir: Path, device: torch.device, resolution: int, out: Path, steps: int = 10) -> None:
    model.eval()
    loader, class_names = _build_test_loader(test_dir, resolution)
    # Use biological order if available
    class_names = ["prophase", "metaphase", "anaphase", "telophase"]
    
    per_class = {i: [] for i in range(len(class_names))}
    with torch.no_grad():
        for x, y in loader:
            posterior = model.encode(x.to(device))
            mean = posterior.latent_dist.mean
            mean_flat = mean.view(mean.size(0), -1).cpu().numpy()
            for z_i, lbl in zip(mean_flat, y.numpy()):
                per_class[int(lbl)].append(z_i)
    
    centroids = [np.mean(per_class[i], axis=0) for i in range(len(class_names)) if len(per_class[i]) > 0]
    
    decoded = []
    n_segs = len(centroids) - 1
    if n_segs > 0:
        for seg in range(n_segs):
            a, b = centroids[seg], centroids[seg + 1]
            for t in np.linspace(0, 1, steps, endpoint=False):
                z_mean = a + t * (b - a)
                # Reshape to latent shape
                # Need to know latent shape - get from model
                dummy = torch.zeros(1, 3, resolution, resolution).to(device)
                posterior = model.encode(dummy)
                latent_shape = posterior.latent_dist.mean.shape
                z_tensor = torch.tensor(z_mean, dtype=torch.float32).view(1, *latent_shape[1:]).to(device)
                with torch.no_grad():
                    img = model.decode(z_tensor)[0].cpu()
                decoded.append(img)
    
    if decoded:
        fig, axes = plt.subplots(n_segs, steps, figsize=(steps * 1.5, n_segs * 1.5))
        if n_segs == 1:
            axes = axes[np.newaxis, :]
        for row in range(n_segs):
            for col in range(steps):
                ax = axes[row, col]
                img = decoded[row * steps + col]
                img_denorm = (img * 0.5 + 0.5).clamp(0, 1)
                ax.imshow(img_denorm.permute(1, 2, 0))
                ax.axis("off")
                if col == 0:
                    ax.set_title(class_names[row], fontsize=8, loc="left")
            axes[row, steps - 1].set_title(class_names[row + 1], fontsize=8)
        
        plt.suptitle("Latent walk (centroid interpolation)", fontsize=12)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)


def run_evaluation(model, history: dict, cfg: SDVAEExperimentConfig, run_dir: Path, val_loader: DataLoader, device: torch.device) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_training_curves(history, plots_dir / "training_curves.png")
    plot_reconstructions(model, cfg, device, plots_dir / "reconstructions.png")
    plot_random_samples(model, device, cfg.data.seed, plots_dir / "random_samples.png")
    
    test_dir = cfg.data.vae_crops_dir / "test"
    if test_dir.exists():
        plot_tsne_test_latents(
            model,
            test_dir,
            device,
            cfg.data.seed,
            cfg.model.resolution,
            plots_dir / "tsne_test_latents.png",
        )
        plot_latent_walk(
            model,
            test_dir,
            device,
            cfg.model.resolution,
            plots_dir / "latent_walk.png",
        )
    else:
        import logging
        logging.getLogger(__name__).warning(f"Test dir {test_dir} not found — skipping t-SNE and latent walk plots.")
