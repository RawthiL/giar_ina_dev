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
# Image-type cue taxonomy (name_types.classify_name_type)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def name_types():
    return _load_script("name_types", _ROOT / "scripts" / "utils" / "name_types.py")


def test_classify_name_type_buckets(name_types):
    cases = {
        "A_208_43.png": "annotated",
        "a_39_9.png": "annotated",
        "E_3_4.png": "annotated",
        "Entrega1_00017_15.png": "annotated",
        "004_00100_24.png": "numbered",
        "002_00033_67.png": "numbered",
        "IMG_1569_JPG.rf.40de59a15831067135e747d6888468cc_0.png": "camera",
        "a4b667fc-MS-ALLROOT__63460_jpg.rf.d6df7817fd8bf3fe72da8d16dc84989f_6.png": "scraped",
        "---------Mitotic-cell-division-stages-of-Allium-cepa-L--014_jpg.rf."
        "0be4791dda547c9c1e436c883851cc96_1.png": "web",
        "istockphoto-933909424-1024x1024_jpg.rf.ed718999b95b4bfa082dc88596f71729_1.png": "web",
    }
    for name, expected in cases.items():
        assert name_types.classify_name_type(name) == expected, name


def test_classify_name_type_only_returns_valid_cues(name_types):
    for name in ["zzz_1_2.png", "random.png", "IMG_x.rf.deadbeef_9.png", "12_34_5.png"]:
        assert name_types.classify_name_type(name) in name_types.TYPE_CUES


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
