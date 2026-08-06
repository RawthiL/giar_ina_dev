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
