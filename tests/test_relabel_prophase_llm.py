"""Tests for scripts/utils/relabel_prophase_llm.py (pure helpers + offline run pieces)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rl():
    path = _ROOT / "scripts" / "utils" / "relabel_prophase_llm.py"
    spec = importlib.util.spec_from_file_location("relabel_prophase_llm", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["relabel_prophase_llm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _choice(name, choice, probs, conf=0.9):
    return {
        "type": "choice",
        "name": name,
        "choice": choice,
        "confidence": conf,
        "probabilities": [{"value": k, "probability": v} for k, v in probs.items()],
    }


def test_confidence_label_thresholds(rl):
    assert rl.confidence_label(0.95) == "high"
    assert rl.confidence_label(0.8) == "high"
    assert rl.confidence_label(0.6) == "medium"
    assert rl.confidence_label(0.49) == "low"


def test_request_cost_uses_cache_tiers(rl):
    usage = {
        "input_tokens": 10_000,
        "input_tokens_details": {"cached_tokens": 8_000, "cache_write_tokens": 1_000},
        "output_tokens": 0,
    }
    expected = (1_000 * 0.10 + 8_000 * 0.01 + 1_000 * 0.125) / 1e6
    assert rl.request_cost(usage) == pytest.approx(expected)


def test_parse_answers_prophase_stage(rl):
    answers = [
        _choice("stage", "mid_prophase", {"mid_prophase": 0.7, "late_prophase": 0.3}),
        _choice("other_reason", "not_applicable", {"not_applicable": 1.0}),
    ]
    out = rl.parse_answers(answers)
    assert out["label"] == "mid_prophase"
    assert out["p_label"] == pytest.approx(0.7)
    assert out["conf_label"] == "medium"
    assert out["other_reason"] == ""
    assert out["refused"] is False


def test_parse_answers_other_keeps_reason(rl):
    answers = [
        _choice("stage", "other", {"other": 0.9, "late_prophase": 0.1}),
        _choice("other_reason", "blurry", {"blurry": 0.8, "unclear": 0.2}),
    ]
    out = rl.parse_answers(answers)
    assert out["label"] == "other"
    assert out["other_reason"] == "blurry"
    assert out["conf_label"] == "high"


def test_parse_answers_refusal_goes_to_other(rl):
    out = rl.parse_answers([{"type": "refusal", "name": "stage"}])
    assert out["label"] == "other"
    assert out["refused"] is True
    assert out["conf_label"] == "low"


def test_build_questions_choices(rl):
    cfg = yaml.safe_load(rl.DEFAULT_PROMPT.read_text())
    qs = rl.build_questions(cfg)
    assert [q["name"] for q in qs] == ["stage", "other_reason"]
    assert [c["value"] for c in qs[0]["choices"]] == rl.STAGES


def test_shared_input_has_one_breakpoint_on_last_reference(rl, tmp_path):
    cfg = yaml.safe_load(rl.DEFAULT_PROMPT.read_text())
    for ref in cfg["references"]:
        Image.new("RGB", (8, 8)).save(tmp_path / ref["file"])
    shared = rl.build_shared_input(cfg, tmp_path, "auto")
    assert shared[0]["role"] == "developer"
    assert len(shared) == 1 + len(cfg["references"])
    marked = [i for i, m in enumerate(shared[1:]) if "prompt_cache_breakpoint" in m["content"][1]]
    assert marked == [len(cfg["references"]) - 1]
    query = rl.build_query_message(cfg, tmp_path / cfg["references"][0]["file"], "auto")
    assert "prompt_cache_breakpoint" not in query["content"][1]


def test_prompt_references_use_valid_labels(rl):
    cfg = yaml.safe_load(rl.DEFAULT_PROMPT.read_text())
    for ref in cfg["references"]:
        assert ref["label"] in rl.STAGES
        if ref["label"] == "other":
            assert ref["reason"] in rl.OTHER_REASONS


def test_manifest_index_and_collect_excludes_refs(rl, tmp_path):
    src = tmp_path / "prophase"
    src.mkdir()
    for n in ("a.png", "b.png", "ref.png", "orphan.png"):
        Image.new("RGB", (4, 4)).save(src / n)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "output,split,legacy,dropped\n"
        "train/tagged/prophase/a.png,train,0,\n"
        "val/tagged/prophase/b.png,val,1,\n"
        "test/prophase/ref.png,test,1,\n"
        "train/tagged/prophase/gone.png,train,0,border cell\n"
    )
    index = rl.load_manifest_index(manifest)
    assert "gone.png" not in index
    files, unmatched = rl.collect_files(src, index, {"ref.png"})
    assert [f["file"] for f in files] == ["a.png", "b.png", "orphan.png"]
    assert unmatched == ["orphan.png"]
    assert {f["file"]: f["split"] for f in files}["b.png"] == "val"


def test_pilot_sample_is_stratified_and_deterministic(rl):
    files = [{"file": f"t{i}.png", "split": "train", "legacy": "0"} for i in range(80)]
    files += [{"file": f"v{i}.png", "split": "val", "legacy": "1"} for i in range(4)]
    a = rl.pilot_sample(files, 10, seed=1)
    b = rl.pilot_sample(files, 10, seed=1)
    assert a == b
    assert len(a) == 10
    assert any(x["split"] == "val" for x in a)


def test_finalize_links_and_summarizes(rl, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    Image.new("RGB", (4, 4)).save(src / "a.png")
    Image.new("RGB", (4, 4)).save(src / "b.png")
    usage = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 80}}
    recs = [
        {
            "file": "a.png",
            "split": "train",
            "legacy": "0",
            "label": "late_prophase",
            "other_reason": "",
            "p_label": 0.9,
            "conf_label": "high",
            "api_confidence": 0.9,
            "refused": False,
            "probs": {"late_prophase": 0.9},
            "usage": usage,
            "cost_usd": 0.001,
        },
        {
            "file": "b.png",
            "split": "val",
            "legacy": "1",
            "label": "other",
            "other_reason": "blurry",
            "p_label": 0.4,
            "conf_label": "low",
            "api_confidence": 0.4,
            "refused": False,
            "probs": {"other": 0.4},
            "usage": usage,
            "cost_usd": 0.001,
        },
    ]
    raw = run / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    summary = rl.finalize(run, src, raw)
    assert (run / "late_prophase" / "a.png").exists()
    assert (run / "other" / "b.png").exists()
    assert summary["n"] == 2
    assert summary["label_counts"] == {"late_prophase": 1, "other": 1}
    assert summary["cache_hit_rate"] == pytest.approx(0.8)
    review = (run / "review.csv").read_text().splitlines()
    assert len(review) == 2 and "b.png" in review[1]
