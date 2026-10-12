"""The VAE trainers must read both the legacy and the vae_256 crop layouts."""

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


def test_collect_untagged_both_layouts(legacy_split, new_split):
    assert len(collect_untagged_paths(legacy_split / "untagged")) == 2
    assert len(collect_untagged_paths(new_split / "untagged")) == 2
    assert collect_untagged_paths(new_split / "untagged", ["mixed"]) == []


def test_phase_dirs_filters(new_split):
    names = [d.name for d in phase_dirs(new_split / "tagged", PHASES)]
    assert "interphase" not in names and len(names) == 4


@pytest.mark.parametrize("trainer", [vae_trainer, sd_vae_trainer])
def test_trainers_load_old_and_new_layout_with_same_phase_content(trainer, legacy_split, new_split):
    tfm = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    old = trainer._load_split(legacy_split, ["tagged", "untagged"], tfm, PHASES, None)
    new = trainer._load_split(new_split, ["tagged", "untagged"], tfm, PHASES, ["mitotic", "mixed"])
    # Same 8 phase crops + 2 untagged, even though the new layout also holds interphase.
    assert len(old) == len(new) == 10
