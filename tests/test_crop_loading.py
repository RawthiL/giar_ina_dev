"""Crop-dataset layout tests: SD-VAE readers are vae_256-only; the legacy 200px VAE
trainer opts in to the old flat ``untagged/*.png`` layout via ``include_loose``."""

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from allium_cepa_classifier.training import sd_vae_trainer, vae_trainer
from allium_cepa_classifier.training.crop_loading import (
    PhaseImageFolder,
    collect_untagged_paths,
    phase_dirs,
)

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]


def _img(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _build(root, *, with_interphase: bool, nested_untagged: bool):
    n = 0
    for phase in PHASES + (["interphase"] if with_interphase else []):
        for i in range(2):
            n += 1
            _img(root / "tagged" / phase / f"{phase}_{i}.png", n)
    base = root / "untagged"
    if nested_untagged:
        _img(base / "mitotic" / "m0.png", 90)
        _img(base / "mitotic" / "m1.png", 91)
        (base / "mixed").mkdir(parents=True, exist_ok=True)  # empty on purpose
    else:
        _img(base / "u0.png", 92)
        _img(base / "u1.png", 93)
    return root


@pytest.fixture
def legacy_split(tmp_path):
    return _build(tmp_path / "legacy", with_interphase=False, nested_untagged=False)


@pytest.fixture
def new_split(tmp_path):
    return _build(tmp_path / "new", with_interphase=True, nested_untagged=True)


def test_phase_folder_ignores_interphase_by_default_phases(new_split):
    ds = PhaseImageFolder(new_split / "tagged", PHASES)
    assert sorted(ds.classes) == sorted(PHASES) and len(ds) == 8


def test_phase_folder_all_classes_when_none(new_split):
    ds = PhaseImageFolder(new_split / "tagged", None)
    assert "interphase" in ds.classes and len(ds) == 10


def test_phase_folder_can_opt_in_to_interphase(new_split):
    ds = PhaseImageFolder(new_split / "tagged", [*PHASES, "interphase"])
    assert len(ds.classes) == 5


def test_phase_folder_unknown_phases_raises(new_split):
    with pytest.raises(FileNotFoundError):
        PhaseImageFolder(new_split / "tagged", ["nope"])


def test_collect_untagged_default_excludes_legacy_loose(legacy_split, new_split):
    # default (vae_256 consumers): nested kinds only, loose legacy files ignored
    assert collect_untagged_paths(legacy_split / "untagged") == []
    assert len(collect_untagged_paths(new_split / "untagged")) == 2
    assert collect_untagged_paths(new_split / "untagged", ["mixed"]) == []


def test_collect_untagged_include_loose_reads_legacy(legacy_split):
    assert len(collect_untagged_paths(legacy_split / "untagged", include_loose=True)) == 2


def test_phase_dirs_filters(new_split):
    names = [d.name for d in phase_dirs(new_split / "tagged", PHASES)]
    assert "interphase" not in names and len(names) == 4


def test_sd_vae_trainer_loads_vae_256_layout(new_split):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = sd_vae_trainer._load_split(
        new_split, ["tagged", "untagged"], tfm, PHASES, ["mitotic", "mixed"]
    )
    # 8 phase crops + 2 untagged; interphase dropped by the phases filter.
    assert len(ds) == 10


def test_sd_vae_trainer_ignores_legacy_loose_untagged(legacy_split):
    tfm = transforms.Compose([transforms.ToTensor()])
    # tagged-only load works; the legacy flat untagged/ is not read by the SD path.
    ds = sd_vae_trainer._load_split(legacy_split, ["tagged"], tfm, PHASES, None)
    assert len(ds) == 8


def test_legacy_vae_trainer_reads_loose_untagged(legacy_split):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = vae_trainer._load_split(legacy_split, ["tagged", "untagged"], tfm, PHASES, None)
    assert len(ds) == 10
