"""Unit tests for LoRA utilities: tb_bridge._parse, evaluator registry, and configs."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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

    # loss_from_tb only reads ctx.run_dir; a stub keeps the test independent of any experiment
    # directory existing on disk (it previously loaded experiments/lora/sd15_rank16, which was
    # deleted, so this test failed on collection rather than on its actual assertions).
    ctx = SimpleNamespace(run_dir=tmp_path)
    result = evaluate_lora.loss_from_tb(ctx)

    assert result["final_loss"] == pytest.approx(loss_values[-1], abs=1e-5)
    assert result["min_loss"] == pytest.approx(min(loss_values), abs=1e-5)
    assert result["avg_loss"] == pytest.approx(sum(loss_values) / len(loss_values), abs=1e-5)
    assert result["loss_tag"] == "loss/current"


# ---------------------------------------------------------------------------
# lora_dataset.rotate_without_fill — guards the rotation-fill regression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lora_dataset():
    return _load_script("lora_dataset", _ROOT / "scripts" / "utils" / "lora_dataset.py")


def _corner_blocks(image, size: int = 12):
    import numpy as np

    a = np.asarray(image)
    return [a[:size, :size], a[:size, -size:], a[-size:, :size], a[-size:, -size:]]


@pytest.mark.parametrize("angle", [-45.0, -15.0, -5.0, 0.0, 5.0, 15.0, 45.0])
def test_rotate_without_fill_leaves_no_constant_corner(lora_dataset, angle):
    """
    Every corner must retain real image texture.

    The bug this guards: `image.rotate(angle, fillcolor=128)` on an RGB image fills the corner
    wedges with (128, 0, 0) — PIL expands a scalar into the first channel only, so the intended
    neutral grey came out dark red. Half of every augmented LoRA set carried those wedges and
    p3_per_phase learned the tilted red-cornered frame as part of the concept.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    out = lora_dataset.rotate_without_fill(Image.fromarray(noise, "RGB"), angle)

    assert out.size == (200, 200)
    for block in _corner_blocks(out):
        assert block.std() > 5.0, f"constant fill wedge in a corner at angle={angle}"


def test_rotate_without_fill_regression_reference(lora_dataset):
    """The old call really does produce a constant dark-red wedge — the behaviour we removed."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    old = Image.fromarray(noise, "RGB").rotate(30.0, resample=Image.BILINEAR, fillcolor=128)

    assert tuple(np.asarray(old)[0, 0]) == (128, 0, 0)


# ---------------------------------------------------------------------------
# lora_dataset.augment_d4 — the lossless alternative to arbitrary-angle rotation
# ---------------------------------------------------------------------------


def test_d4_transposes_are_pixel_permutations(lora_dataset):
    """
    The whole point of the d4 augmenter: orientation diversity at zero resampling cost.

    `rotate_without_fill` costs two BILINEAR passes plus a 1.22x upscale at +-15deg, and
    `p3_per_phase_nofill` measured the damage — vqgan_recon_ratio 0.737 -> 0.444, i.e. markedly
    smoother than real crops. A transpose invents no pixel values, so the sorted pixel multiset
    must survive it exactly. If this ever fails, the augmenter has started interpolating.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(src, "RGB")

    for ops in lora_dataset.D4_TRANSPOSES:
        out = img
        for op in ops:
            out = out.transpose(op)
        arr = np.asarray(out)
        assert arr.shape == src.shape
        assert np.array_equal(np.sort(arr.ravel()), np.sort(src.ravel()))


def test_d4_variants_give_distinct_orientations(lora_dataset):
    """Distinct `variant` values must produce distinct geometry, or copies=3 yields duplicates."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8), "RGB")

    seen = set()
    for variant in range(len(lora_dataset.D4_TRANSPOSES)):
        out = img
        for op in lora_dataset.D4_TRANSPOSES[variant]:
            out = out.transpose(op)
        seen.add(np.asarray(out).tobytes())
    assert len(seen) == len(lora_dataset.D4_TRANSPOSES)


def test_new_dataset_versions_are_registered(lora_dataset):
    defaults = lora_dataset._VERSION_DEFAULTS
    assert defaults["per_phase_noaug"] == (0, "mild", "per-phase", True)
    assert defaults["per_phase_d4x3"] == (3, "d4", "per-phase", True)
