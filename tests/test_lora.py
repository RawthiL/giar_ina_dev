"""Unit tests for LoRA utilities: tb_bridge._parse, evaluator registry, and configs."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, path: Path):
    """Import a scripts/ module by file path (they are not packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Phase 3: lora_tb_bridge._parse  filename → (step, idx)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tb_bridge():
    return _load_script("lora_tb_bridge", _ROOT / "scripts" / "utils" / "lora_tb_bridge.py")


def test_parse_epoch_based(tb_bridge):
    step, idx = tb_bridge._parse("sd15_rank16_e000001_00_20241201123456")
    assert step == 1
    assert idx == 0


def test_parse_epoch_larger(tb_bridge):
    step, idx = tb_bridge._parse("mymodel_e000010_03_20260101000000")
    assert step == 10
    assert idx == 3


def test_parse_step_based(tb_bridge):
    step, idx = tb_bridge._parse("sd15_rank16_000100_02_20241201123456_42")
    assert step == 100
    assert idx == 2


def test_parse_step_based_no_seed(tb_bridge):
    step, idx = tb_bridge._parse("sd15_rank16_000050_01_20260707120000")
    assert step == 50
    assert idx == 1


def test_parse_invalid_raises(tb_bridge):
    with pytest.raises(ValueError, match="Cannot parse"):
        tb_bridge._parse("invalid_filename")


# ---------------------------------------------------------------------------
# Phase 4: config — all lora configs load with unique experiment_name
# ---------------------------------------------------------------------------


def test_lora_configs_unique_experiment_names():
    from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

    config_paths = sorted((_ROOT / "experiments" / "lora").glob("*/config.yaml"))
    config_paths = [p for p in config_paths if "_sweeps" not in str(p)]
    assert config_paths, "No LoRA experiment configs found under experiments/lora/"

    names = [LoRAExperimentConfig.from_yaml(p).experiment_name for p in config_paths]
    duplicates = [n for n in names if names.count(n) > 1]
    assert not duplicates, f"Duplicate experiment_name values: {duplicates}"


# ---------------------------------------------------------------------------
# Phase 4: evaluator registry — dummy metric is registered and called
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluate_lora():
    return _load_script("evaluate_lora", _ROOT / "scripts" / "evaluate_lora.py")


def test_registry_contains_loss(evaluate_lora):
    assert "loss" in evaluate_lora.METRICS


def test_registry_dummy_metric(evaluate_lora, tmp_path):
    """A metric registered at runtime appears in METRICS without touching sweep_lora.py."""
    called = []

    @evaluate_lora.register("_test_dummy")
    def dummy(cfg, run_dir):
        called.append(run_dir)
        return {"dummy_value": 42.0}

    assert "_test_dummy" in evaluate_lora.METRICS
    result = evaluate_lora.METRICS["_test_dummy"](None, tmp_path)
    assert result == {"dummy_value": 42.0}
    assert called == [tmp_path]


def test_registry_loss_reads_tb_events(evaluate_lora, tmp_path):
    """loss_from_tb returns correct final/min/avg from a real TensorBoard event file."""
    from torch.utils.tensorboard import SummaryWriter

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    loss_values = [0.9, 0.7, 0.5, 0.4]
    writer = SummaryWriter(log_dir=str(log_dir))
    for step, val in enumerate(loss_values, start=1):
        writer.add_scalar("loss/current", val, global_step=step)
    writer.close()

    from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

    cfg = LoRAExperimentConfig.from_yaml(
        _ROOT / "experiments" / "lora" / "sd15_rank16" / "config.yaml"
    )
    result = evaluate_lora.loss_from_tb(cfg, tmp_path)

    assert result["final_loss"] == pytest.approx(loss_values[-1], abs=1e-5)
    assert result["min_loss"] == pytest.approx(min(loss_values), abs=1e-5)
    assert result["avg_loss"] == pytest.approx(sum(loss_values) / len(loss_values), abs=1e-5)
    assert result["loss_tag"] == "loss/current"


# ---------------------------------------------------------------------------
# Provenance source cue (name_types.source_from_name)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def name_types():
    return _load_script("name_types", _ROOT / "scripts" / "utils" / "name_types.py")


def test_source_from_name_reads_vae_256_prefix(name_types):
    cases = {
        "ina_004_00054_23354.png": "ina",
        "ina_004_00094_23752.png": "ina",
        "ina_legacy_005_00010_59.png": "ina",
        "onion_A_208_43.png": "onion",
        "roboflow_108_21797.png": "roboflow",
        "roboflow_legacy_a4b667fc_6.png": "roboflow",
    }
    for name, expected in cases.items():
        assert name_types.source_from_name(name) == expected, name


def test_source_from_name_only_returns_valid_cues_or_none(name_types):
    for name in [
        "zzz_1_2.png",
        "random.png",
        "IMG_x.rf.deadbeef_9.png",
        "004_00100_24.png",
        "A_208_43.png",
    ]:
        assert name_types.source_from_name(name) is None
    assert set(name_types.SOURCE_CUES) == {"ina", "onion", "roboflow"}


def test_caption_for_strings_pinned(name_types):
    assert name_types.caption_for("metaphase") == (
        "micrograph of allium cepa root tip mitotic cell in metaphase phase"
    )
    assert name_types.caption_for("metaphase", "ina") == (
        "micrograph of allium cepa root tip mitotic cell in metaphase phase and source ina"
    )
    # interphase is not a division state: its own wording, no "mitotic"/"phase"
    assert name_types.caption_for("interphase") == (
        "micrograph of allium cepa root tip cell in interphase"
    )
    assert name_types.caption_for("interphase", "onion") == (
        "micrograph of allium cepa root tip cell in interphase and source onion"
    )


def test_caption_for_phase_vocabulary(name_types):
    assert name_types.PHASES == [*name_types.MITOTIC_PHASES, "interphase"]
    with pytest.raises(ValueError):
        name_types.caption_for("diakinesis")


# ---------------------------------------------------------------------------
# LoRA dataset augmentation: mirror-padded rotation (no gray letterbox)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lora_dataset():
    return _load_script("lora_dataset", _ROOT / "scripts" / "utils" / "lora_dataset.py")


def test_rotate_reflect_size_preserved_and_no_constant_fill(lora_dataset):
    from PIL import Image

    # Reflect-padding a solid image must keep it solid everywhere: no artificial corner
    # block is introduced (the old fillcolor=128 actually produced red (128,0,0) corners
    # on RGB because Pillow applies a single int only to band 0).
    solid = (10, 20, 30)
    img = Image.new("RGB", (64, 48), solid)
    out = lora_dataset._rotate_reflect(img, 7.0)
    assert out.size == img.size
    import numpy as np

    arr = np.asarray(out.convert("RGB"))
    assert np.all(arr == np.array(solid, dtype=np.uint8)), "mirror fill leaked a new color"


def test_rotate_reflect_small_angle_is_noop(lora_dataset):
    from PIL import Image

    img = Image.new("RGB", (32, 32), (200, 10, 10))
    assert lora_dataset._rotate_reflect(img, 0.4) is img


def test_subsample_pool_cap_is_deterministic_and_sorted(lora_dataset):
    from pathlib import Path

    pool = [(Path(f"ina_0_{i:04d}.png"), "interphase", "train") for i in range(10)]
    got = lora_dataset.subsample_pool(pool, 4, seed=7)
    assert len(got) == 4
    assert got == lora_dataset.subsample_pool(pool, 4, seed=7), "same seed must reproduce"
    assert [str(e[0]) for e in got] == sorted(str(e[0]) for e in got), "result stays sorted"
    assert lora_dataset.subsample_pool(pool, 4, seed=8) != got, "seed must change the pick"
    # cap disabled / no-op paths return the pool untouched
    assert lora_dataset.subsample_pool(pool, -1, seed=7) == pool
    assert lora_dataset.subsample_pool(pool[:3], 4, seed=7) == pool[:3]


def test_lora_dataset_phase_pool_includes_interphase(lora_dataset):
    assert "interphase" in lora_dataset.PHASES
    assert lora_dataset.INTERPHASE == "interphase"
