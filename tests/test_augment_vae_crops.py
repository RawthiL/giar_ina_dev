"""Regression tests for the VAE-crop augmentation mirror-fill rotation."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def augment_vae_crops():
    return _load_script("augment_vae_crops", _ROOT / "scripts" / "utils" / "augment_vae_crops.py")


def test_rotate_reflect_grayscale_mirror_stays_solid(augment_vae_crops):
    import numpy as np
    from PIL import Image

    solid = Image.new("L", (64, 48), 200)
    out = augment_vae_crops._rotate_reflect(solid, 6.0)
    arr = np.asarray(out)
    assert out.size == solid.size
    assert (arr == 200).all(), "mirror fill should keep a solid grayscale image solid"


def test_rotate_reflect_small_angle_noop(augment_vae_crops):
    from PIL import Image

    img = Image.new("L", (32, 32), 90)
    assert augment_vae_crops._rotate_reflect(img, 0.4) is img
