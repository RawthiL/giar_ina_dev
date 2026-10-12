"""Loading helpers shared by the VAE / SD-VAE trainers and evaluators.

They make the trainers agnostic to the crop-dataset layout:

* ``datasets/crops/vae``     tagged/{4 phases}   untagged/*.png          (flat)
* ``datasets/crops/vae_256`` tagged/{4 phases + interphase}   untagged/{mitotic,mixed}/*.png

``phases`` restricts an ImageFolder to the listed class folders, so the extra ``interphase``
folder is ignored unless it is asked for explicitly. ``untagged_kinds`` selects which
``untagged/<kind>`` subfolders are read; loose images directly under ``untagged/`` (the
legacy layout) are always read.
"""

from __future__ import annotations

from pathlib import Path

from torchvision import datasets

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
DEFAULT_PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
DEFAULT_UNTAGGED_KINDS = ["mitotic", "mixed"]


class PhaseImageFolder(datasets.ImageFolder):
    """ImageFolder that only sees the class folders listed in ``phases`` (None = all)."""

    def __init__(self, root, phases: list[str] | None = None, **kwargs):
        self._phases = None if phases is None else list(phases)
        super().__init__(str(root), **kwargs)

    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        if self._phases is not None:
            wanted = set(self._phases)
            classes = [c for c in classes if c in wanted]
            if not classes:
                raise FileNotFoundError(
                    f"None of the requested phases {self._phases} exist in {directory}"
                )
        return classes, {c: i for i, c in enumerate(classes)}


def collect_untagged_paths(root: Path, kinds: list[str] | None = None) -> list[Path]:
    """Images under ``root``: loose files (legacy layout) + files in ``root/<kind>/``."""
    kinds = DEFAULT_UNTAGGED_KINDS if kinds is None else kinds
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    for kind in kinds:
        sub = root / kind
        if sub.is_dir():
            paths.extend(p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return sorted(paths)


def phase_dirs(root: Path, phases: list[str] | None = None) -> list[Path]:
    """Class folders under ``root``, restricted to ``phases`` when given."""
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    return dirs if phases is None else [d for d in dirs if d.name in set(phases)]
