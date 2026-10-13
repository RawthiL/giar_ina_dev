"""Tests for scripts/utils/crop_dataset.py (256px cell-crop dataset builder)."""

import csv
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cd():
    path = _ROOT / "scripts" / "utils" / "crop_dataset.py"
    spec = importlib.util.spec_from_file_location("crop_dataset", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crop_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


STAGES = {1: "metaphase", 2: "anaphase", 3: "telophase"}


def test_classify_origin(cd):
    assert cd.classify_origin("001_00029") == "ina"
    assert cd.classify_origin("A_100") == "onion"
    assert cd.classify_origin("IMG_1569_JPG.rf.deadbeefcafe") == "roboflow"
    assert cd.classify_origin("---Mitotic-048_jpg.rf.d55d95ff") == "roboflow"


def test_legacy_names(cd):
    assert cd.legacy_output_name("001_00029", 7) == "ina_legacy_001_00029_7.png"
    name = cd.legacy_output_name("IMG_1_JPG.rf.54e9663752c17ced", 3)
    assert name == "roboflow_legacy_54e96637_3.png"


def test_assign_tag(cd):
    assert cd.assign_tag(True, 2, STAGES) == ("anaphase", True)
    # division=False with a stage is contradictory: the stage wins and flips division.
    assert cd.assign_tag(False, 3, STAGES) == ("telophase", True)
    assert cd.assign_tag(True, 0, STAGES) == ("mitotic", True)
    assert cd.assign_tag(False, 0, STAGES) == ("interphase", False)


def test_square_window_uses_longest_side(cd):
    left, top, right, bottom = cd.square_window([100, 100, 40, 80])
    assert right - left == bottom - top == 80
    assert (left + right) / 2 == 120 and (top + bottom) / 2 == 140


def test_crop_is_square_and_resized(cd):
    arr = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    out = cd.crop_square_resized(arr, [150, 100, 50, 120], (256, 256))
    assert out is not None and out.size == (256, 256)


def test_edge_crop_reflects_about_image_border(cd):
    arr = np.random.default_rng(0).integers(1, 255, (100, 100, 3), dtype=np.uint8)
    patch = cd.crop_window_reflect(arr, (-20, 40, 40, 100))
    assert patch.shape[:2] == (60, 60)
    assert (patch[:, 20:40] == arr[40:100, 0:20]).all()  # in-bounds part untouched
    assert (patch[:, 19] == arr[40:100, 1]).all()  # first padded column mirrors column 1


def test_edge_crop_has_no_constant_fill_wedge(cd):
    rng = np.random.default_rng(1)
    arr = rng.integers(1, 255, (80, 80, 3), dtype=np.uint8)
    patch = cd.crop_window_reflect(arr, (-30, -30, 30, 30))
    assert (patch == 0).sum() == 0, "padded area must come from the image, not a zero fill"


def test_window_outside_image_returns_none(cd):
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    assert cd.crop_window_reflect(arr, (100, 100, 120, 120)) is None


def _rand_img(path: Path, size=(220, 220), seed=0):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def _coco(split_dir: Path, images: list[tuple[str, list[dict]]]):
    imgs, anns = [], []
    ann_id = 1
    for i, (fname, boxes) in enumerate(images, start=1):
        _rand_img(split_dir / "images" / fname, seed=zlib.crc32(fname.encode()))
        imgs.append({"id": i, "file_name": fname, "width": 220, "height": 220})
        for b in boxes:
            anns.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": b["bbox"],
                    "attributes": {
                        "division": b.get("division", False),
                        "mitosis_stage": b.get("stage", 0),
                    },
                }
            )
            ann_id += 1
    (split_dir / "data").mkdir(parents=True, exist_ok=True)
    (split_dir / "data" / "annotations.json").write_text(
        json.dumps({"images": imgs, "annotations": anns, "categories": [{"id": 1}]})
    )


@pytest.fixture
def tiny_dataset(tmp_path, cd):
    raw = tmp_path / "raw"
    box = [60, 60, 50, 70]
    _coco(
        raw / "train",
        [
            ("001_00001.jpg", [{"bbox": box}, {"bbox": [10, 10, 40, 40], "division": True}]),
            ("A_1.png", [{"bbox": box, "division": True, "stage": 2}]),
        ],
    )
    _coco(raw / "validation", [("B_2.png", [{"bbox": box}, {"bbox": box, "division": True}])])
    _coco(
        raw / "test",
        [("rf_img_jpg.rf.abcdef0123456789.jpg", [{"bbox": box, "division": False, "stage": 3}])],
    )
    legacy = tmp_path / "legacy"
    # Legacy prophase crop whose source image exists (-> inherits train) ...
    _rand_img(legacy / "train/tagged/prophase/001_00001_5.png", (200, 200), seed=90)
    # ... and one with no source image at all, plus an _aug copy that must be ignored.
    _rand_img(legacy / "train/tagged/metaphase/ghost_jpg.rf.1122334455667788_2.png", (200, 200), 91)
    _rand_img(
        legacy / "train/tagged/metaphase/ghost_jpg.rf.1122334455667788_2_aug.png", (200, 200), 92
    )
    return cd.Settings(
        raw_dir=raw,
        legacy_dir=legacy,
        out_dir=tmp_path / "out",
        overrides=_ROOT / "scripts" / "utils" / "annotation_overrides.yaml",
        size=(64, 64),
    )


def test_end_to_end_layout_naming_and_no_leak(cd, tiny_dataset):
    report = cd.run(tiny_dataset)
    out = tiny_dataset.out_dir
    assert report["legacy"]["aug_skipped"] == 1 and report["legacy"]["kept"] == 2

    names = {p.relative_to(out).as_posix() for p in out.rglob("*.png")}
    assert "train/tagged/interphase/ina_001_00001_1.png" in names
    assert "train/untagged/mitotic/ina_001_00001_2.png" in names
    assert "train/tagged/anaphase/onion_A_1_3.png" in names  # stage 2 -> anaphase
    assert "val/tagged/interphase/onion_B_2_1.png" in names
    assert "val/untagged/mitotic/onion_B_2_2.png" in names
    # division=False + stage 3 is promoted to telophase, and test keeps plain class folders.
    assert any(n.startswith("test/telophase/roboflow_1_") for n in names)
    assert "train/tagged/prophase/ina_legacy_001_00001_5.png" in names
    assert any("roboflow_legacy_11223344_2" in n for n in names)
    assert (out / "train/untagged/mixed").is_dir()
    assert all(Image.open(out / n).size == (64, 64) for n in names)

    rows = list(csv.DictReader((out / "manifest.csv").open()))
    legacy_rows = [r for r in rows if r["legacy"] == "1"]
    assert len(legacy_rows) == 2
    assert all("upscaled" in r["note"] for r in legacy_rows)
    by_split: dict[str, set[str]] = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["content_hash"])
    splits = list(by_split.values())
    assert all(not (a & b) for i, a in enumerate(splits) for b in splits[i + 1 :])


def test_legacy_match_consumes_annotation(cd, tiny_dataset):
    """A legacy crop cut from an annotation's own box consumes it (no duplicate cell)."""
    raw = tiny_dataset.raw_dir
    src = Image.open(raw / "train/images/001_00001.jpg").convert("RGB")
    x, y, w, h = 60, 60, 50, 70
    squished = src.crop((x, y, x + w, y + h)).resize((200, 200))
    legacy_file = tiny_dataset.legacy_dir / "train/tagged/prophase/001_00001_0.png"
    squished.save(legacy_file)
    report = cd.run(tiny_dataset)
    assert report["annotations_consumed_by_legacy"] >= 1
    rows = list(csv.DictReader((tiny_dataset.out_dir / "manifest.csv").open()))
    # The interphase annotation (id 1) was consumed, so only the legacy prophase carries it.
    assert not any(
        r["split"] == "train" and r["annotation_id"] == "1" and r["legacy"] == "0" for r in rows
    )


def test_refuses_to_wipe_unrelated_directory(cd, tiny_dataset):
    tiny_dataset.out_dir.mkdir(parents=True)
    (tiny_dataset.out_dir / "precious.txt").write_text("keep me")
    with pytest.raises(SystemExit):
        cd.run(tiny_dataset)
    assert (tiny_dataset.out_dir / "precious.txt").exists()
