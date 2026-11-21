"""Tests for the data-budget helpers in training/mitosis_phase_trainer.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets

from allium_cepa_classifier.training.mitosis_phase_trainer import (
    _class_counts,
    _compute_class_weights,
    _train_targets,
    subsample_by_class,
)

ALL5 = ["prophase", "metaphase", "anaphase", "telophase", "interphase"]


def _make_folder(tmp_path, counts: dict[str, int]) -> datasets.ImageFolder:
    root = tmp_path / "ds"
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"img_{i:03d}.png").write_bytes(b"")
    return datasets.ImageFolder(root)


def _selected_paths(sub) -> list[str]:
    return sorted(str(sub.dataset.samples[i][0]) for i in sub.indices)


def test_subsample_noop_returns_same_dataset(tmp_path):
    ds = _make_folder(tmp_path, dict.fromkeys(ALL5, 5))
    assert subsample_by_class(ds, None, 0) is ds
    assert subsample_by_class(ds, -1, 0) is ds
    assert subsample_by_class(ds, 5, 0) is ds  # every pool == cap
    assert subsample_by_class(ds, 100, 0) is ds  # every pool < cap


def test_subsample_caps_every_class_including_interphase(tmp_path):
    counts = {"prophase": 9, "metaphase": 4, "anaphase": 8, "telophase": 2, "interphase": 30}
    ds = _make_folder(tmp_path, counts)
    sub = subsample_by_class(ds, 5, seed=1234)
    assert isinstance(sub, Subset)
    targets = _train_targets(sub)
    per_class = {ds.classes[t]: int((targets == t).sum()) for t in range(len(ds.classes))}
    for cls, n in counts.items():
        assert per_class[cls] == min(n, 5)


def test_subsample_deterministic_and_seed_dependent(tmp_path):
    ds = _make_folder(tmp_path, dict.fromkeys(ALL5, 10))
    a = subsample_by_class(ds, 4, seed=7)
    b = subsample_by_class(ds, 4, seed=7)
    c = subsample_by_class(ds, 4, seed=8)
    assert _selected_paths(a) == _selected_paths(b)
    assert _selected_paths(a) != _selected_paths(c)


def test_subsample_independent_of_sample_order(tmp_path):
    ds_a = _make_folder(tmp_path / "a", dict.fromkeys(ALL5, 10))
    ds_b = _make_folder(tmp_path / "b", dict.fromkeys(ALL5, 10))
    ds_b.samples = list(reversed(ds_b.samples))
    sub_a = subsample_by_class(ds_a, 3, seed=99)
    sub_b = subsample_by_class(ds_b, 3, seed=99)
    # Same basenames selected regardless of the input order.
    assert [p.rsplit("/", 2)[-2:] for p in _selected_paths(sub_a)] == [
        p.rsplit("/", 2)[-2:] for p in _selected_paths(sub_b)
    ]


def test_subsample_indices_sorted_and_unique(tmp_path):
    ds = _make_folder(tmp_path, dict.fromkeys(ALL5, 8))
    sub = subsample_by_class(ds, 3, seed=5)
    assert sub.indices == sorted(set(sub.indices))


@pytest.fixture
def two_folders(tmp_path):
    root_r, root_s = tmp_path / "real", tmp_path / "synth"
    for root, counts in ((root_r, {"a": 9, "b": 6}), (root_s, {"a": 4, "b": 3})):
        for cls, n in counts.items():
            d = root / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"img_{i:03d}.png").write_bytes(b"")
    return datasets.ImageFolder(root_r), datasets.ImageFolder(root_s)


def test_train_targets_plain_imagefolder(two_folders):
    real, _ = two_folders
    targets = _train_targets(real)
    assert np.array_equal(targets, np.asarray(real.targets))


def test_train_targets_subset(two_folders):
    real, _ = two_folders
    sub = subsample_by_class(real, 4, seed=1)
    targets = _train_targets(sub)
    expected = np.asarray([real.targets[i] for i in sub.indices])
    assert np.array_equal(targets, expected)
    assert (targets == 0).sum() == 4  # class "a" capped
    assert (targets == 1).sum() == 4  # class "b" capped


def test_train_targets_concat_of_subsets(two_folders):
    real, synth = two_folders
    real_sub = subsample_by_class(real, 3, seed=1)
    synth_sub = subsample_by_class(synth, 2, seed=1)
    targets = _train_targets(ConcatDataset([real_sub, synth_sub]))
    assert len(targets) == len(real_sub) + len(synth_sub)
    expected = np.concatenate(
        [
            np.asarray([real.targets[i] for i in real_sub.indices]),
            np.asarray([synth.targets[i] for i in synth_sub.indices]),
        ]
    )
    assert np.array_equal(targets, expected)


def test_class_weights_no_multipliers_matches_sklearn_balanced():
    labels = np.array([0] * 9 + [1] * 3)
    got = _compute_class_weights(2, labels, torch.device("cpu"), ["a", "b"])
    assert torch.allclose(got, torch.tensor([12 / (2 * 9), 12 / (2 * 3)]))


def test_class_weights_multipliers_scale_per_class():
    labels = np.array([0] * 9 + [1] * 3)
    got = _compute_class_weights(2, labels, torch.device("cpu"), ["a", "b"], {"b": 2.0, "z": 5.0})
    assert torch.allclose(got, torch.tensor([12 / (2 * 9), 2.0 * 12 / (2 * 3)]))


def test_class_counts_reports_every_class():
    labels = np.array([0, 0, 0, 2, 2, 4])
    counts = _class_counts(labels, ["a", "b", "c", "d", "e"])
    assert counts == {"a": 3, "b": 0, "c": 2, "d": 0, "e": 1}
    assert sum(counts.values()) == len(labels)
