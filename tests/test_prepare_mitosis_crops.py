"""Tests for scripts/utils/prepare_mitosis_crops.py: class selection, interphase cap, tokens."""

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
def pmc():
    return _load_script(
        "prepare_mitosis_crops", _ROOT / "scripts" / "utils" / "prepare_mitosis_crops.py"
    )


MIT = ["prophase", "metaphase", "anaphase", "telophase"]
ALL5 = [*MIT, "interphase"]


def _make_vae_tree(root: Path, *, n_mitotic: int, n_inter: int) -> Path:
    for split, sub in [("train", "tagged"), ("val", "tagged"), ("test", "")]:
        for cls in ALL5:
            d = root / split / sub / cls if sub else root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_inter if cls == "interphase" else n_mitotic):
                (d / f"ina_{cls}_{i}.png").write_bytes(b"x")
    return root


def test_subsample_pool_deterministic_and_sorted(pmc):
    pool = [Path(f"i_{i:03d}.png") for i in range(10)]
    got = pmc.subsample_pool(pool, 4, seed=7)
    assert len(got) == 4
    assert got == pmc.subsample_pool(pool, 4, seed=7)
    assert got == sorted(got, key=str)
    assert pmc.subsample_pool(pool, 4, seed=8) != got
    assert pmc.subsample_pool(pool, -1, seed=7) == pool
    assert pmc.subsample_pool(pool[:2], 4, seed=7) == pool[:2]


def test_build_real_caps_interphase_train_and_val_only(pmc, tmp_path):
    vae = _make_vae_tree(tmp_path / "vae_256", n_mitotic=3, n_inter=10)
    out = tmp_path / "mitosis_phase5"
    counts = pmc.build_real(vae, out, ALL5, interphase_cap=4, seed=0)

    for split in ("train", "validation"):
        assert len(list((out / split / "interphase").iterdir())) == 4
        for cls in MIT:
            assert len(list((out / split / cls).iterdir())) == 3
    # test split keeps every interphase crop
    assert len(list((out / "test" / "interphase").iterdir())) == 10
    # 4*3 mitotic + 4 interphase = 16 for train/validation; 12 + 10 = 22 for test
    assert counts == {"train": 16, "validation": 16, "test": 22}


def test_build_real_four_class_ignores_interphase(pmc, tmp_path):
    vae = _make_vae_tree(tmp_path / "vae_256", n_mitotic=2, n_inter=9)
    out = tmp_path / "mitosis_phase"
    counts = pmc.build_real(vae, out, MIT, interphase_cap=4, seed=0)
    assert sorted(p.name for p in (out / "train").iterdir()) == sorted(MIT)
    assert counts["train"] == 8


def test_phase_from_name_parsing(pmc):
    assert pmc._phase_from_name("00042_interphase_seed3791", ALL5) == "interphase"
    assert pmc._phase_from_name("00042_prophase_seed3791_aug01", ALL5) == "prophase"
    # with the 4-class vocabulary, interphase-named files match nothing
    assert pmc._phase_from_name("00042_interphase_seed3791", MIT) is None
    assert pmc._phase_from_name("00042_diakinesis_seed1", ALL5) is None


def test_build_synthetic_routes_and_skips(pmc, tmp_path):
    src = tmp_path / "images"
    src.mkdir()
    for name in [
        "0_prophase_seed1.png",
        "1_interphase_seed2.png",
        "2_interphase_seed3.png",
        "3_bogus_seed4.png",
    ]:
        (src / name).write_bytes(b"x")
    out = tmp_path / "synth"
    counts = pmc.build_synthetic(src, out, ALL5)
    assert counts == {"prophase": 1, "metaphase": 0, "anaphase": 0, "telophase": 0, "interphase": 2}
    assert len(list((out / "interphase").iterdir())) == 2
