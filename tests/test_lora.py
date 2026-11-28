"""Unit tests for LoRA utilities: tb_bridge._parse, evaluator registry, and configs."""

import importlib.util
import json
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
        _ROOT / "experiments" / "lora" / "vae256_tagcue" / "config.yaml"
    )
    result = evaluate_lora.loss_from_tb(cfg, tmp_path)

    assert result["final_loss"] == pytest.approx(loss_values[-1], abs=1e-5)
    assert result["min_loss"] == pytest.approx(min(loss_values), abs=1e-5)
    assert result["avg_loss"] == pytest.approx(sum(loss_values) / len(loss_values), abs=1e-5)
    assert result["loss_tag"] == "loss/current"


# ---------------------------------------------------------------------------
# VLM-tag captions (folder phase wins; detailed_mitotic_stage is ignored)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def name_types():
    return _load_script("name_types", _ROOT / "scripts" / "utils" / "name_types.py")


@pytest.fixture(scope="module")
def vlm_tags():
    return _load_script("vlm_tags", _ROOT / "scripts" / "utils" / "vlm_tags.py")


def test_phase_caption_strings_pinned(name_types):
    assert name_types.phase_caption("metaphase") == (
        "micrograph of allium cepa root tip mitotic cell in metaphase phase"
    )
    assert name_types.phase_caption("interphase") == (
        "micrograph of allium cepa root tip cell in interphase"
    )


def test_phase_caption_phase_vocabulary(name_types):
    assert name_types.PHASES == [*name_types.MITOTIC_PHASES, "interphase"]
    with pytest.raises(ValueError):
        name_types.phase_caption("diakinesis")


def test_vlm_caption_uses_folder_phase_and_ignores_detailed_stage(vlm_tags):
    tags = {
        "cell_shape": "elongated_rectangular",
        "cell_wall_contrast": "high_contrast_bright_lines",
        "detailed_mitotic_stage": "aligned_metaphase",
    }
    caption = vlm_tags.caption_for("prophase", tags)
    assert caption.startswith("micrograph of allium cepa root tip mitotic cell in prophase phase")
    assert "elongated rectangular cell" in caption
    assert "high contrast bright cell wall lines" in caption
    assert "metaphase" not in caption


def test_vlm_caption_phase_specific_fields(vlm_tags):
    interphase_tags = {
        "cell_shape": "circular_rounded_protoplast",
        "interphase_chromatin_texture": "distinct_dark_nucleoli_visible",
        "axis_of_division": "horizontal",
    }
    caption = vlm_tags.caption_for("interphase", interphase_tags)
    assert "distinct dark nucleoli visible" in caption
    assert "horizontal division axis" not in caption
    anaphase_caption = vlm_tags.caption_for("anaphase", interphase_tags)
    assert "horizontal division axis" in anaphase_caption
    assert "distinct dark nucleoli visible" not in anaphase_caption


def test_vlm_caption_missing_and_rare_tags(vlm_tags):
    assert vlm_tags.caption_for("telophase", {}) == (
        "micrograph of allium cepa root tip mitotic cell in telophase phase"
    )
    assert vlm_tags.caption_for("telophase", {"cell_shape": "undefined_due_to_rupture"}) == (
        "micrograph of allium cepa root tip mitotic cell in telophase phase"
    )


def test_vlm_extract_only_selected_fields(vlm_tags):
    raw = {
        "geometry_and_fov": {
            "cell_shape": "polyhedral",
            "zoom_and_crop_level": "tight_crop_nucleus_only",
        },
        "chromosome_and_nuclear_dynamics": {
            "detailed_mitotic_stage": "late_anaphase",
            "axis_of_division": "horizontal",
        },
        "optics_color_and_artifacts": {"focus_quality": "sharp_high_definition"},
    }
    assert vlm_tags.extract_tags(raw) == {
        "cell_shape": "polyhedral",
        "axis_of_division": "horizontal",
        "focus_quality": "sharp_high_definition",
    }


def test_prompt_for_accepts_unseen_valid_combination(vlm_tags):
    caption = vlm_tags.prompt_for(
        "prophase",
        {
            "focus_quality": "sharp_high_definition",
            "neighboring_cell_intrusion": "faint_neighboring_walls",
        },
    )
    assert "sharp focus" in caption
    assert "faint neighboring cell walls" in caption


def test_prompt_for_rejects_unsupported_value(vlm_tags):
    with pytest.raises(vlm_tags.TagValidationError, match="Unsupported value"):
        vlm_tags.prompt_for("prophase", {"focus_quality": "extremely_sharp"})


def test_prompt_for_rejects_unknown_field(vlm_tags):
    with pytest.raises(vlm_tags.TagValidationError, match="Unknown tag field"):
        vlm_tags.prompt_for("prophase", {"not_a_tag": "whatever"})


def test_prompt_for_rejects_phase_specific_misuse(vlm_tags):
    with pytest.raises(vlm_tags.TagValidationError, match="axis_of_division"):
        vlm_tags.prompt_for("prophase", {"axis_of_division": "horizontal"})
    with pytest.raises(vlm_tags.TagValidationError, match="interphase_chromatin_texture"):
        vlm_tags.prompt_for("metaphase", {"interphase_chromatin_texture": "smooth_solid_color"})


def test_prompt_for_accepts_valid_phase_specific_tags(vlm_tags):
    anaphase = vlm_tags.prompt_for("anaphase", {"axis_of_division": "horizontal"})
    assert "horizontal division axis" in anaphase
    interphase = vlm_tags.prompt_for(
        "interphase", {"interphase_chromatin_texture": "smooth_solid_color"}
    )
    assert "smooth solid chromatin" in interphase


def test_prompt_for_canonicalizes_case_and_bool(vlm_tags):
    caption = vlm_tags.prompt_for(
        "prophase",
        {
            "cell_shape": "ELONGATED_RECTANGULAR",
            "central_vacuole_presence": False,
        },
    )
    assert "elongated rectangular cell" in caption
    assert "no central vacuole" in caption


def test_filter_supported_tags_drops_unknown_fields_and_values(vlm_tags):
    assert vlm_tags.filter_supported_tags(
        {
            "cell_shape": "polyhedral",
            "focus_quality": "severely_defocused",
            "not_a_tag": "whatever",
        }
    ) == {"cell_shape": "polyhedral"}


# ---------------------------------------------------------------------------
# Division-sequence state validation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def division_sequence():
    return _load_script(
        "generate_division_sequence", _ROOT / "scripts" / "generate_division_sequence.py"
    )


def _common_tags(**overrides):
    tags = {
        "cell_shape": "elongated_rectangular",
        "cell_wall_contrast": "faint_dark_borders",
        "neighboring_cell_intrusion": "faint_neighboring_walls",
        "focus_quality": "sharp_high_definition",
        "central_vacuole_presence": False,
    }
    tags.update(overrides)
    return tags


def test_division_states_sorted_and_axis_copied(division_sequence, tmp_path):
    path = tmp_path / "states.json"
    path.write_text(
        json.dumps(
            [
                {"phase": "telophase", "tags": _common_tags()},
                {"phase": "anaphase", "tags": _common_tags(axis_of_division="horizontal")},
                {"phase": "interphase", "tags": _common_tags()},
            ]
        )
    )
    states = division_sequence._load_states(path, 0.4, False)
    assert [state["phase"] for state in states] == ["interphase", "anaphase", "telophase"]
    assert states[1]["tags"]["axis_of_division"] == "horizontal"
    assert states[2]["tags"]["axis_of_division"] == "horizontal"


def test_division_axis_mismatch_raises(division_sequence, tmp_path):
    path = tmp_path / "states.json"
    path.write_text(
        json.dumps(
            [
                {"phase": "anaphase", "tags": _common_tags(axis_of_division="horizontal")},
                {"phase": "telophase", "tags": _common_tags(axis_of_division="vertical")},
            ]
        )
    )
    with pytest.raises(SystemExit, match="axis mismatch"):
        division_sequence._load_states(path, 0.4, False)


def test_division_common_tag_mismatch_raises(division_sequence, tmp_path):
    path = tmp_path / "states.json"
    path.write_text(
        json.dumps(
            [
                {"phase": "interphase", "tags": _common_tags()},
                {
                    "phase": "prophase",
                    "tags": _common_tags(cell_shape="circular_rounded_protoplast"),
                },
            ]
        )
    )
    with pytest.raises(SystemExit, match="differs across states"):
        division_sequence._load_states(path, 0.4, False)


def test_division_unknown_value_raises(division_sequence, tmp_path):
    path = tmp_path / "states.json"
    path.write_text(
        json.dumps(
            [
                {"phase": "interphase", "tags": _common_tags()},
                {"phase": "prophase", "tags": _common_tags(focus_quality="extremely_sharp")},
            ]
        )
    )
    with pytest.raises(SystemExit, match="Invalid state"):
        division_sequence._load_states(path, 0.4, False)


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
