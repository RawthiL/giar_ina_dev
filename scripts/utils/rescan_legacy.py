"""
Rescan the legacy manual crops against the full FOV images.

Many legacy crops (``datasets/crops/vae``) show cells that are visible in the original images
but are missing from the COCO annotations. For every legacy crop whose source image is in the
merged dataset this script:

1. runs the YOLO detector on that image (low confidence) to find cells not yet annotated,
2. matches each legacy crop to the best candidate (existing annotation or new detection) with
   LPIPS (see ``legacy_match.py``),
3. appends the new detections to the merged dataset:
   - matched to a legacy crop -> annotation inheriting the legacy phase,
   - not matched (and not on the image border) -> annotation with empty ``attributes``
     to classify by hand; ``crop_dataset.py`` puts them in ``untagged/mixed`` until then,
4. writes ``legacy_matches.json`` (legacy crop -> annotation) that ``crop_dataset.py`` consumes.

Legacy-matched annotations carry ``attributes.source = "yolo_rescan"``; unmatched ones have none. They are added to the per-split
``{split}/data/annotations.json`` and mirrored into ``full_annotations.json`` for the images it
contains. Originals are backed up first. Default is a dry run; pass ``--apply`` to write.
Re-running with yolo_rescan annotations present requires ``--reset`` (which drops them first,
including cells still unclassified).

Usage:
    uv run python scripts/utils/rescan_legacy.py                 # dry run + report
    uv run python scripts/utils/rescan_legacy.py --apply
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import crop_dataset as cd  # noqa: E402
import legacy_match as lm  # noqa: E402

from allium_cepa_classifier.config import AlliumCepaConfig  # noqa: E402

SOURCE = "yolo_rescan"
REVIEW = "REVIEW"
PHASE_ATTRS = {  # phase -> (division, mitosis_stage); prophase cells are annotated division=False
    "prophase": (False, 0),
    "metaphase": (True, 1),
    "anaphase": (True, 2),
    "telophase": (True, 3),
}
MATCHES_FILE = Path(__file__).resolve().parent / "legacy_matches.json"


def load_raw(raw_dir: Path) -> dict[str, dict]:
    return {
        s: json.loads((raw_dir / s / "data" / "annotations.json").read_text()) for s in cd.SPLITS
    }


def is_rescan(ann: dict) -> bool:
    """Cells added by this script: tagged ``source``, or unclassified (no ``division`` key)."""
    attrs = ann.get("attributes") or {}
    return attrs.get("source") == SOURCE or "division" not in attrs


def strip_rescan(coco: dict) -> int:
    before = len(coco["annotations"])
    coco["annotations"] = [a for a in coco["annotations"] if not is_rescan(a)]
    return before - len(coco["annotations"])


def detect(model, calibrator, path: Path, conf: float) -> list[dict]:
    res = model.predict(source=str(path), conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    p_hat = np.clip(calibrator.predict(confs), 0, 1) if calibrator is not None else confs
    return [
        {
            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            "conf": float(c),
            "p_hat": float(p),
        }
        for (x0, y0, x1, y1), c, p in zip(xyxy, confs, p_hat, strict=True)
    ]


def new_annotation(box: dict, phase: str, legacy: str | None, lpips: float | None) -> dict:
    x, y, w, h = box["bbox"]
    if phase == REVIEW:
        attrs = {}  # detected only: no classification attributes, to be classified by hand
    else:
        div, stage = PHASE_ATTRS[phase]
        attrs = {
            "division": div,
            "mitosis_stage": stage,
            "occluded": False,
            "rotation": 0,
            "source": SOURCE,
            "phase": phase,
            "yolo_conf": round(box["conf"], 4),
            "yolo_p_hat": round(box["p_hat"], 4),
            "legacy_match": legacy,
            "lpips": round(lpips or 0.0, 4),
        }
    return {
        "category_id": 1,
        "segmentation": [],
        "area": w * h,
        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
        "iscrowd": 0,
        "attributes": attrs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write annotations + legacy_matches.json")
    ap.add_argument("--reset", action="store_true", help="drop existing yolo_rescan annotations")
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO confidence floor")
    ap.add_argument("--new-iou", type=float, default=0.3, help="max IoU with an annotation")
    ap.add_argument("--threshold", type=float, default=0.30, help="max LPIPS for a match")
    ap.add_argument("--max-ratio", type=float, default=0.75, help="best / runner-up LPIPS")
    ap.add_argument(
        "--review-conf", type=float, default=0.5, help="min YOLO conf for unclassified cells"
    )
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--report", default=None, help="write the JSON report here")
    args = ap.parse_args()

    t = cd.TrainingConfig()
    margin = t.crop_margin if args.margin is None else args.margin
    raw_dir = t.raw_dataset_dir
    cfg = cd.Settings(
        raw_dir=raw_dir,
        legacy_dir=t.vae_crops_dir,
        out_dir=Path("/nonexistent"),
        overrides=Path("/nonexistent"),
    )
    raw = load_raw(raw_dir)
    existing = {s: sum(is_rescan(a) for a in c["annotations"]) for s, c in raw.items()}
    if any(existing.values()) and not args.reset:
        raise SystemExit(
            f"yolo_rescan annotations already present {existing}; --reset drops them "
            "(including cells still unclassified)"
        )
    full_path = raw_dir / "full_annotations.json"
    full = json.loads(full_path.read_text())
    for coco in (*raw.values(), full):
        strip_rescan(coco)

    # image index (stem -> split, path, coco image record) from the cleaned split files
    images: dict[str, dict] = {}
    for split, coco in raw.items():
        for im in coco["images"]:
            stem = Path(im["file_name"]).stem
            images[stem] = {
                "split": split,
                "path": raw_dir / split / "images" / im["file_name"],
                "im": im,
            }
    anns_by_image: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for split, coco in raw.items():
        for a in coco["annotations"]:
            anns_by_image[(split, a["image_id"])].append(a)

    legacy, _ = cd.load_legacy(cfg)
    by_stem: dict[str, list[cd.Legacy]] = defaultdict(list)
    for lg in legacy:
        if lg.stem in images:
            by_stem[lg.stem].append(lg)
    print(f"{sum(map(len, by_stem.values()))} legacy crops across {len(by_stem)} source images")

    from ultralytics import YOLO

    acfg = AlliumCepaConfig()
    model = YOLO(str(acfg.detection_weights_path))
    calibrator = None
    if acfg.detection_calibrator_path.exists():
        with open(acfg.detection_calibrator_path, "rb") as f:
            calibrator = pickle.load(f)
    scorer = lm.LpipsScorer()

    stats: Counter = Counter()
    matches: dict[str, dict] = {}
    adds: list[dict] = []  # {split, image_id, file_name, ann}
    t0 = time.time()
    for n, (stem, lgs) in enumerate(sorted(by_stem.items()), 1):
        meta = images[stem]
        split, im = meta["split"], meta["im"]
        existing_anns = anns_by_image[(split, im["id"])]
        ex_boxes = np.array([a["bbox"] for a in existing_anns], dtype=float).reshape(-1, 4)
        dets = detect(model, calibrator, meta["path"], args.conf)
        stats["yolo_detections"] += len(dets)
        new = [d for d in dets if lm.iou_xywh(d["bbox"], ex_boxes).max(initial=0.0) < args.new_iou]
        stats["yolo_new_candidates"] += len(new)

        arr = np.asarray(Image.open(meta["path"]).convert("RGB"))
        cand_boxes = [a["bbox"] for a in existing_anns] + [d["bbox"] for d in new]
        views = [lm.candidate_views(arr, b, margin) for b in cand_boxes]
        dist = scorer.matrix([Image.open(lg.path).convert("RGB") for lg in lgs], views)
        pairs = lm.mutual_best(
            dist, args.threshold, args.max_ratio, np.array(cand_boxes, dtype=float).reshape(-1, 4)
        )

        used_new: dict[int, cd.Legacy] = {}
        for i, j, d, ratio in pairs:
            lg = lgs[i]
            rec = {"split": split, "image": stem, "lpips": round(d, 4), "ratio": round(ratio, 3)}
            if j < len(existing_anns):
                rec["ann_id"] = existing_anns[j]["id"]
                rec["new"] = False
                stats["legacy_matched_existing"] += 1
            else:
                used_new[j - len(existing_anns)] = lg
                rec["new"] = True
                rec["_new_idx"] = j - len(existing_anns)
                stats["legacy_matched_new_cell"] += 1
            matches[lg.out_name] = rec
        stats["legacy_unmatched"] += len(lgs) - len(pairs)

        for k, d in enumerate(new):
            lg = used_new.get(k)
            border = lm.is_border_cell(d["bbox"], im["width"], im["height"], margin)
            if lg is not None:
                ann = new_annotation(d, lg.phase, lg.out_name, matches[lg.out_name]["lpips"])
                matches[lg.out_name]["_ann"] = ann
            elif d["conf"] >= args.review_conf and not border:
                ann = new_annotation(d, REVIEW, None, None)
                stats["review_cells"] += 1
            else:
                stats["new_candidates_discarded"] += 1
                continue
            adds.append(
                {"split": split, "image_id": im["id"], "file_name": im["file_name"], "ann": ann}
            )
        if n % 25 == 0:
            print(f"  {n}/{len(by_stem)} images ({time.time() - t0:.0f}s)")

    # ids: next free per split / per full_annotations
    next_split = {
        s: max((a["id"] for a in c["annotations"]), default=0) + 1 for s, c in raw.items()
    }
    next_full = max((a["id"] for a in full["annotations"]), default=0) + 1
    full_images = {i["file_name"]: i["id"] for i in full["images"]}
    for add in adds:
        a = add["ann"]
        sid = next_split[add["split"]]
        next_split[add["split"]] += 1
        raw[add["split"]]["annotations"].append({"id": sid, "image_id": add["image_id"], **a})
        a["_id"] = sid
        if add["file_name"] in full_images:
            full["annotations"].append(
                {
                    "id": next_full,
                    "image_id": full_images[add["file_name"]],
                    **{k: v for k, v in a.items() if k != "_id"},
                }
            )
            next_full += 1
            stats["mirrored_in_full_annotations"] += 1
    for rec in matches.values():
        if rec.get("new"):
            rec["ann_id"] = rec["_ann"]["_id"]
        rec.pop("_ann", None)
        rec.pop("_new_idx", None)

    report = {
        "legacy_with_source": sum(map(len, by_stem.values())),
        "images_scanned": len(by_stem),
        "params": {k: v for k, v in vars(args).items() if k not in ("apply", "report")},
        "stats": dict(sorted(stats.items())),
        "added_per_split": dict(Counter(a["split"] for a in adds)),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    if not args.apply:
        print("dry run: nothing written (use --apply)")
        return

    backup = raw_dir / "_backups" / time.strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True)
    for s in cd.SPLITS:
        (backup / s).mkdir()
        shutil.copy2(raw_dir / s / "data" / "annotations.json", backup / s / "annotations.json")
    shutil.copy2(full_path, backup / "full_annotations.json")
    for s, coco in raw.items():
        (raw_dir / s / "data" / "annotations.json").write_text(json.dumps(coco))
    full_path.write_text(json.dumps(full))
    MATCHES_FILE.write_text(json.dumps(dict(sorted(matches.items())), indent=1))
    print(f"wrote annotations (backup: {backup}) and {MATCHES_FILE}")


if __name__ == "__main__":
    main()
