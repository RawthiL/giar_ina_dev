"""
Build the cell-crop dataset (``datasets/crops/vae_256``) from the merged COCO annotations
plus the manually tagged legacy crops in ``datasets/crops/vae``.

Cropping: a square window centred on each bbox with side ``max(w, h) * (1 + margin)``,
reflection-padded where it leaves the image, resized to ``--width x --height``. Aspect
ratio is preserved (the legacy crops were squished, this does not do that).

Tags (annotation attributes after ``annotation_overrides.yaml``):
  mitosis_stage 1/2/3     -> tagged/{metaphase,anaphase,telophase}   (division forced True)
  division=True, stage 0  -> untagged/mitotic   (mitotic, phase unknown)
  division=False          -> tagged/interphase
  tagged/prophase         -> legacy manual crops only (annotations cannot identify prophase)

Legacy crops are authoritative: their folder is the phase, they are resized to the target
size and renamed ``{origin}_legacy_{id}_{idx}``. A legacy crop that matches an annotation
(mutual-best normalised correlation >= ``--match-threshold``) consumes that annotation, so
the same cell is never emitted twice.

Layout (mirrors the legacy dataset, plus ``interphase`` and ``untagged/{mitotic,mixed}``):
  train/tagged/{phase}/  train/untagged/{mitotic,mixed}/
  val/tagged/{phase}/    val/untagged/{mitotic,mixed}/
  test/{phase}/
  manifest.csv

Splits follow the COCO image splits. Legacy crops inherit the split of their source image;
those without one are grouped by hash-stripped source name and assigned whole.

Usage:
    uv run python scripts/utils/crop_dataset.py
    uv run python scripts/utils/crop_dataset.py --width 256 --height 256 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from allium_cepa_classifier.config.training_config import TrainingConfig

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
INTERPHASE = "interphase"
CLASSES = [*PHASES, INTERPHASE]
SPLITS = ["train", "validation", "test"]
OUT_SPLIT = {"train": "train", "validation": "val", "test": "test"}
# Cross-split duplicates are removed from the lowest-priority split first.
DEDUP_KEEP_PRIORITY = {"test": 0, "validation": 1, "train": 2}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FEATURE_SIZE = 48
REPORT_THRESHOLDS = (0.65, 0.75, 0.85, 0.9)

_INA_RE = re.compile(r"^\d{3}_\d{5}$")
_ONION_RE = re.compile(r"^[A-F]_\d+$")
_LEGACY_NAME_RE = re.compile(r"^(.*)_(\d+)$")

MANIFEST_FIELDS = [
    "output",
    "split",
    "tag",
    "untagged_kind",
    "origin",
    "source_image",
    "annotation_id",
    "bbox",
    "division",
    "mitosis_stage",
    "legacy",
    "legacy_original",
    "match_score",
    "content_hash",
    "dropped",
    "note",
]


# --------------------------------------------------------------------------- naming / tags


def classify_origin(stem: str) -> str:
    """Dataset of origin from an image stem: ina (001_00029), onion (A_100), else roboflow."""
    if _INA_RE.match(stem):
        return "ina"
    if _ONION_RE.match(stem):
        return "onion"
    return "roboflow"


def roboflow_hash8(stem: str) -> str:
    """First 8 chars of the Roboflow export hash; md5 of the stem when there is none."""
    if ".rf." in stem:
        return stem.split(".rf.", 1)[1][:8]
    return hashlib.md5(stem.encode()).hexdigest()[:8]


def legacy_output_name(stem: str, idx: int) -> str:
    """``{origin}_legacy_{id}_{idx}.png``; roboflow ids are shortened to the 8-char hash."""
    origin = classify_origin(stem)
    ident = roboflow_hash8(stem) if origin == "roboflow" else stem
    return f"{origin}_legacy_{ident}_{idx}.png"


def source_group(stem: str) -> str:
    """Hash-stripped source name: re-exports of one photo share a group."""
    return stem.split(".rf.", 1)[0]


def assign_tag(division: bool, stage: int, stage_to_phase: dict[int, str]) -> tuple[str, bool]:
    """(tag, effective_division). Tag is a phase, 'interphase' or 'mitotic' (phase unknown)."""
    if stage in stage_to_phase:
        return stage_to_phase[stage], True
    if division:
        return "mitotic", True
    return INTERPHASE, False


# --------------------------------------------------------------------------- geometry


def square_window(bbox: list[float], margin: float = 0.0) -> tuple[int, int, int, int]:
    """Square (left, top, right, bottom) centred on the bbox, side max(w, h)*(1+margin)."""
    x, y, w, h = bbox
    side = max(2, int(round(max(w, h) * (1.0 + margin))))
    cx, cy = x + w / 2.0, y + h / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    return left, top, left + side, top + side


def crop_window_reflect(arr: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray | None:
    """Cut ``window`` from ``arr`` (H, W[, C]); reflect about the image border outside it.

    Returns None when the window does not overlap the image at all.
    """
    left, top, right, bottom = window
    height, width = arr.shape[:2]
    x0, x1 = max(left, 0), min(right, width)
    y0, y1 = max(top, 0), min(bottom, height)
    if x1 <= x0 or y1 <= y0:
        return None
    inner = arr[y0:y1, x0:x1]
    pad = [(y0 - top, bottom - y1), (x0 - left, right - x1)] + [(0, 0)] * (arr.ndim - 2)
    if any(p for pair in pad for p in pair):
        # np.pad(reflect) re-tiles for pads larger than the patch, so thin boxes are safe.
        # Reflect needs >1 sample on the padded axis; fall back to edge replicate otherwise.
        mode = "reflect" if min(inner.shape[:2]) > 1 else "edge"
        inner = np.pad(inner, pad, mode=mode)
    return inner


def crop_square_resized(
    arr: np.ndarray, bbox: list[float], size: tuple[int, int], margin: float = 0.0
) -> Image.Image | None:
    """Square max(w,h) crop with reflection padding, resized to ``size`` (W, H)."""
    patch = crop_window_reflect(arr, square_window(bbox, margin))
    if patch is None:
        return None
    return Image.fromarray(patch).resize(size, Image.LANCZOS)


def _gray_vec(img: Image.Image) -> np.ndarray:
    v = np.asarray(img.convert("L").resize((FEATURE_SIZE, FEATURE_SIZE)), dtype=np.float32).ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def annotation_features(arr: np.ndarray, bbox: list[float], margin: float) -> list[np.ndarray]:
    """Feature vectors for the two ways a legacy crop may have been cut (squish / square)."""
    x, y, w, h = bbox
    height, width = arr.shape[:2]
    feats = []
    sq = crop_window_reflect(arr, square_window(bbox, margin))
    if sq is not None:
        feats.append(_gray_vec(Image.fromarray(sq)))
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x + w), width), min(int(y + h), height)
    if x1 > x0 and y1 > y0:
        feats.append(_gray_vec(Image.fromarray(arr[y0:y1, x0:x1])))
    return feats


# --------------------------------------------------------------------------- data classes


@dataclass
class Ann:
    split: str
    ann_id: int
    image_key: str  # stem
    image_path: Path
    origin: str
    bbox: list[float]
    division: bool
    stage: int
    tag: str = ""
    consumed_by: str | None = None
    match_score: float = 0.0


@dataclass
class Legacy:
    path: Path
    phase: str
    stem: str
    idx: int
    out_name: str
    split: str | None = None
    ann: Ann | None = None
    score: float = 0.0
    note: str = ""


@dataclass
class Settings:
    raw_dir: Path
    legacy_dir: Path | None
    out_dir: Path
    overrides: Path
    size: tuple[int, int] = (256, 256)
    margin: float = 0.0
    match_threshold: float = 0.85
    min_box: float = 8.0
    seed: int = 42
    legacy_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    dry_run: bool = False
    limit_images: int | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- loading


def load_overrides(path: Path) -> tuple[dict[int, str], dict[tuple[str, int], dict]]:
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    stage_map = {int(k): v for k, v in (data.get("stage_to_phase") or {}).items()}
    per_ann = {
        (str(o["split"]), int(o["annotation_id"])): o for o in (data.get("annotations") or [])
    }
    return stage_map, per_ann


def load_annotations(cfg: Settings) -> tuple[list[Ann], dict[str, dict]]:
    """All annotations (with overrides + tags applied) and an image index keyed by stem."""
    stage_map, per_ann = load_overrides(cfg.overrides)
    anns: list[Ann] = []
    images: dict[str, dict] = {}
    for split in SPLITS:
        ann_path = cfg.raw_dir / split / "data" / "annotations.json"
        if not ann_path.exists():
            cfg.notes.append(f"missing annotations for split {split}: {ann_path}")
            continue
        coco = json.loads(ann_path.read_text())
        by_id = {im["id"]: im for im in coco["images"]}
        if cfg.limit_images:
            keep = set(sorted(by_id)[: cfg.limit_images])
            by_id = {k: v for k, v in by_id.items() if k in keep}
        for im in by_id.values():
            stem = Path(im["file_name"]).stem
            if stem in images:
                cfg.notes.append(f"duplicate image stem across splits: {stem}")
            images[stem] = {
                "split": split,
                "path": cfg.raw_dir / split / "images" / im["file_name"],
                "file_name": im["file_name"],
                "origin": classify_origin(stem),
            }
        for a in coco["annotations"]:
            im = by_id.get(a["image_id"])
            if im is None:
                continue
            attrs = a.get("attributes", {})
            division, stage = bool(attrs.get("division", False)), int(attrs.get("mitosis_stage", 0))
            ov = per_ann.get((split, int(a["id"])))
            if ov:
                division = bool(ov.get("division", division))
                stage = int(ov.get("mitosis_stage", stage))
            stem = Path(im["file_name"]).stem
            tag, division = assign_tag(division, stage, stage_map)
            anns.append(
                Ann(
                    split=split,
                    ann_id=int(a["id"]),
                    image_key=stem,
                    image_path=images[stem]["path"],
                    origin=images[stem]["origin"],
                    bbox=[float(v) for v in a["bbox"]],
                    division=division,
                    stage=stage,
                    tag=tag,
                )
            )

    # Sequential roboflow numbers by sorted original stem (stable across splits).
    rf = sorted(s for s, m in images.items() if m["origin"] == "roboflow")
    for n, stem in enumerate(rf, start=1):
        images[stem]["number"] = n
    return anns, images


def ann_output_name(a: Ann, images: dict[str, dict]) -> str:
    ident = str(images[a.image_key]["number"]) if a.origin == "roboflow" else a.image_key
    return f"{a.origin}_{ident}_{a.ann_id}.png"


def load_legacy(cfg: Settings) -> tuple[list[Legacy], Counter]:
    """Legacy manual crops (tagged + test), minus `_aug` copies and exact duplicates."""
    stats: Counter = Counter()
    if cfg.legacy_dir is None or not cfg.legacy_dir.exists():
        return [], stats
    sources = []
    for phase in PHASES:
        sources.append((cfg.legacy_dir / "train" / "tagged" / phase, phase))
        sources.append((cfg.legacy_dir / "val" / "tagged" / phase, phase))
        sources.append((cfg.legacy_dir / "test" / phase, phase))
    seen: dict[str, Legacy] = {}
    conflicted: set[str] = set()
    for folder, phase in sources:
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if "_aug" in p.stem:
                stats["aug_skipped"] += 1
                continue
            m = _LEGACY_NAME_RE.match(p.stem)
            if not m:
                stats["unparsable_name"] += 1
                continue
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in conflicted:
                stats["duplicate_conflict"] += 1
                continue
            if digest in seen:
                if seen[digest].phase != phase:
                    # Same pixels filed under two phases: untrustworthy, drop both.
                    conflicted.add(digest)
                    del seen[digest]
                    stats["duplicate_conflict"] += 2
                else:
                    stats["duplicate_dropped"] += 1
                continue
            stem, idx = m.group(1), int(m.group(2))
            seen[digest] = Legacy(p, phase, stem, idx, legacy_output_name(stem, idx))
    legacy = sorted(seen.values(), key=lambda lg: str(lg.path))
    stats["kept"] = len(legacy)
    # Unique output names (two different crops can collide on origin+hash8+idx).
    used: Counter = Counter()
    for lg in legacy:
        used[lg.out_name] += 1
        if used[lg.out_name] > 1:
            lg.out_name = lg.out_name.replace(".png", f"-{used[lg.out_name]}.png")
            stats["name_collision"] += 1
    return legacy, stats


# --------------------------------------------------------------------------- matching


def match_legacy(
    legacy: list[Legacy], anns: list[Ann], images: dict[str, dict], cfg: Settings
) -> dict[float, int]:
    """Mutual-best match legacy crops to annotations of their source image.

    Sets ``lg.split`` (from the source image), and ``lg.ann`` for accepted matches.
    Returns how many mutual matches each report threshold would accept.
    """
    by_image: dict[str, list[Ann]] = defaultdict(list)
    for a in anns:
        by_image[a.image_key].append(a)
    prefix_to_splits: dict[str, Counter] = defaultdict(Counter)
    for stem, meta in images.items():
        prefix_to_splits[source_group(stem)][meta["split"]] += 1

    legacy_by_image: dict[str, list[Legacy]] = defaultdict(list)
    for lg in legacy:
        if lg.stem in images:
            legacy_by_image[lg.stem].append(lg)
            lg.split = images[lg.stem]["split"]
        else:
            splits = prefix_to_splits.get(source_group(lg.stem))
            if splits:
                lg.split = splits.most_common(1)[0][0]
                lg.note = "split inherited from re-export of same source"

    counts = dict.fromkeys(REPORT_THRESHOLDS, 0)
    for stem, lgs in legacy_by_image.items():
        image_anns = [a for a in by_image.get(stem, []) if max(a.bbox[2], a.bbox[3]) >= cfg.min_box]
        if not image_anns:
            continue
        arr = np.asarray(Image.open(images[stem]["path"]).convert("RGB"))
        feats = [annotation_features(arr, a.bbox, cfg.margin) for a in image_anns]
        keep = [i for i, f in enumerate(feats) if f]
        if not keep:
            continue
        legacy_vecs = np.stack([_gray_vec(Image.open(lg.path)) for lg in lgs])
        sim = np.full((len(lgs), len(keep)), -1.0, dtype=np.float32)
        for col, i in enumerate(keep):
            sim[:, col] = np.max([legacy_vecs @ f for f in feats[i]], axis=0)
        best_ann = sim.argmax(axis=1)
        best_leg = sim.argmax(axis=0)
        for r, lg in enumerate(lgs):
            c = int(best_ann[r])
            score = float(sim[r, c])
            mutual = int(best_leg[c]) == r
            lg.score = score
            if not mutual:
                continue
            for t in REPORT_THRESHOLDS:
                if score >= t:
                    counts[t] += 1
            if score >= cfg.match_threshold:
                lg.ann = image_anns[keep[c]]
                lg.ann.consumed_by = lg.out_name
                lg.ann.match_score = score
    return counts


def assign_legacy_splits(legacy: list[Legacy], cfg: Settings) -> Counter:
    """Give every split-less legacy crop a split, whole source groups at a time."""
    rng = random.Random(cfg.seed)
    current: Counter = Counter(lg.split for lg in legacy if lg.split)
    pending: dict[str, list[Legacy]] = defaultdict(list)
    for lg in legacy:
        if lg.split is None:
            pending[source_group(lg.stem)].append(lg)
    groups = sorted(pending.items())
    rng.shuffle(groups)
    groups.sort(key=lambda kv: -len(kv[1]))  # big groups first, then fill the deficits
    total = len(legacy)
    targets = dict(zip(SPLITS, (r * total for r in cfg.legacy_ratios), strict=True))
    assigned: Counter = Counter()
    for _, members in groups:
        split = max(SPLITS, key=lambda s: targets[s] - current[s])
        for lg in members:
            lg.split = split
            lg.note = "no source image in merged set; split assigned by source group"
        current[split] += len(members)
        assigned[split] += len(members)
    return assigned


# --------------------------------------------------------------------------- writing


def dest_dir(out_dir: Path, split: str, tag: str, untagged_kind: str | None = None) -> Path | None:
    base = out_dir / OUT_SPLIT[split]
    if tag == "mitotic":
        if split == "test":
            return None  # test keeps plain class folders only
        return base / "untagged" / (untagged_kind or "mitotic")
    return base / tag if split == "test" else base / "tagged" / tag


def _save(img: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img.save(path)
    return hashlib.sha1(img.tobytes()).hexdigest()


def write_dataset(
    anns: list[Ann],
    legacy: list[Legacy],
    images: dict[str, dict],
    cfg: Settings,
) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    stats: Counter = Counter()

    def row(**kw) -> dict:
        base = dict.fromkeys(MANIFEST_FIELDS, "")
        base.update(kw)
        return base

    by_image: dict[str, list[Ann]] = defaultdict(list)
    for a in anns:
        if a.consumed_by is None:
            by_image[a.image_key].append(a)

    for stem, items in sorted(by_image.items()):
        arr = None
        for a in items:
            dest = dest_dir(cfg.out_dir, a.split, a.tag)
            if dest is None:
                stats[f"skipped_test_untagged_{a.tag}"] += 1
                continue
            if max(a.bbox[2], a.bbox[3]) < cfg.min_box:
                stats["skipped_tiny_box"] += 1
                continue
            if cfg.dry_run:
                stats[f"{a.split}/{a.tag}"] += 1
                continue
            if arr is None:
                arr = np.asarray(Image.open(a.image_path).convert("RGB"))
            crop = crop_square_resized(arr, a.bbox, cfg.size, cfg.margin)
            if crop is None:
                stats["skipped_outside_image"] += 1
                continue
            name = ann_output_name(a, images)
            digest = _save(crop, dest / name)
            stats[f"{a.split}/{a.tag}"] += 1
            rows.append(
                row(
                    output=str((dest / name).relative_to(cfg.out_dir)),
                    split=a.split,
                    tag=a.tag,
                    untagged_kind="mitotic" if a.tag == "mitotic" else "",
                    origin=a.origin,
                    source_image=images[stem]["file_name"],
                    annotation_id=a.ann_id,
                    bbox=json.dumps([round(v, 2) for v in a.bbox]),
                    division=int(a.division),
                    mitosis_stage=a.stage,
                    legacy=0,
                    content_hash=digest,
                )
            )

    for lg in legacy:
        assert lg.split is not None
        dest = dest_dir(cfg.out_dir, lg.split, lg.phase)
        assert dest is not None
        stats[f"{lg.split}/{lg.phase}(legacy)"] += 1
        if cfg.dry_run:
            continue
        img = Image.open(lg.path).convert("RGB").resize(cfg.size, Image.LANCZOS)
        digest = _save(img, dest / lg.out_name)
        note = [lg.note] if lg.note else []
        note.append("legacy: upscaled from 200x200 (squished at source, aspect not recoverable)")
        if lg.ann is not None:
            note.append(f"matched annotation {lg.ann.ann_id} (stage {lg.ann.stage}, {lg.ann.tag})")
            if lg.ann.tag in PHASES and lg.ann.tag != lg.phase:
                stats["stage_folder_conflicts"] += 1
                note.append(f"CONFLICT stage says {lg.ann.tag}; folder wins")
        rows.append(
            row(
                output=str((dest / lg.out_name).relative_to(cfg.out_dir)),
                split=lg.split,
                tag=lg.phase,
                origin=classify_origin(lg.stem),
                source_image=lg.stem,
                annotation_id=lg.ann.ann_id if lg.ann else "",
                division=1,
                mitosis_stage=lg.ann.stage if lg.ann else "",
                legacy=1,
                legacy_original=lg.path.name,
                match_score=round(lg.score, 3),
                content_hash=digest,
                note="; ".join(note),
            )
        )
    return rows, stats


def drop_cross_split_duplicates(rows: list[dict], cfg: Settings) -> int:
    """Identical pixels in two splits are leakage: delete the copy in the lowest-priority split."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_hash[r["content_hash"]].append(r)
    dropped = 0
    for group in by_hash.values():
        splits = {r["split"] for r in group}
        if len(splits) < 2:
            continue
        keep = min(splits, key=lambda s: DEDUP_KEEP_PRIORITY[s])
        for r in group:
            if r["split"] != keep:
                (cfg.out_dir / r["output"]).unlink(missing_ok=True)
                r["dropped"] = f"cross-split duplicate of {keep}"
                dropped += 1
    return dropped


def verify_no_leaks(rows: list[dict]) -> None:
    live = [r for r in rows if not r["dropped"]]
    hash_splits: dict[str, set[str]] = defaultdict(set)
    img_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in live:
        hash_splits[r["content_hash"]].add(r["split"])
        if not r["legacy"]:
            img_splits[(r["origin"], r["source_image"])].add(r["split"])
    bad_hash = [h for h, s in hash_splits.items() if len(s) > 1]
    bad_img = [k for k, s in img_splits.items() if len(s) > 1]
    if bad_hash or bad_img:
        raise SystemExit(
            f"LEAK: {len(bad_hash)} content hashes and {len(bad_img)} source images "
            "appear in more than one split"
        )


# --------------------------------------------------------------------------- driver


def run(cfg: Settings) -> dict:
    anns, images = load_annotations(cfg)
    legacy, legacy_stats = load_legacy(cfg)
    thresholds = match_legacy(legacy, anns, images, cfg)
    assigned = assign_legacy_splits(legacy, cfg)

    if not cfg.dry_run:
        if cfg.out_dir.exists():
            # Only ever delete a previous output of this script (or an empty dir).
            if any(cfg.out_dir.iterdir()) and not (cfg.out_dir / "manifest.csv").exists():
                raise SystemExit(
                    f"Refusing to overwrite {cfg.out_dir}: not empty and has no manifest.csv"
                )
            shutil.rmtree(cfg.out_dir)
        for split in SPLITS:
            for cls in CLASSES:
                d = dest_dir(cfg.out_dir, split, cls)
                if d:
                    d.mkdir(parents=True, exist_ok=True)
            if split != "test":
                for kind in ("mitotic", "mixed"):
                    dest_dir(cfg.out_dir, split, "mitotic", kind).mkdir(  # type: ignore[union-attr]
                        parents=True, exist_ok=True
                    )

    rows, stats = write_dataset(anns, legacy, images, cfg)
    dropped = 0
    if not cfg.dry_run:
        dropped = drop_cross_split_duplicates(rows, cfg)
        verify_no_leaks(rows)
        with (cfg.out_dir / "manifest.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)

    consumed = sum(1 for a in anns if a.consumed_by)
    report = {
        "annotations": len(anns),
        "annotations_consumed_by_legacy": consumed,
        "legacy": dict(legacy_stats),
        "legacy_with_source_image": sum(1 for lg in legacy if lg.stem in images),
        "legacy_split_assigned_by_group": dict(assigned),
        "mutual_matches_by_threshold": {str(t): n for t, n in thresholds.items()},
        "match_threshold_used": cfg.match_threshold,
        "cross_split_duplicates_dropped": dropped,
        "written": dict(sorted(stats.items())),
        "notes": cfg.notes,
    }
    return report


def build_settings(args: argparse.Namespace) -> Settings:
    t = TrainingConfig()
    root = t.crops_dir.parent.parent
    legacy = Path(args.legacy_dir) if args.legacy_dir else t.vae_crops_dir
    return Settings(
        raw_dir=Path(args.raw_dir) if args.raw_dir else t.raw_dataset_dir,
        legacy_dir=None if args.no_legacy else legacy,
        out_dir=Path(args.out) if args.out else t.vae_256_crops_dir,
        overrides=Path(args.overrides)
        if args.overrides
        else root / "scripts" / "utils" / "annotation_overrides.yaml",
        size=(args.width or t.crop_size[0], args.height or t.crop_size[1]),
        margin=t.crop_margin if args.margin is None else args.margin,
        match_threshold=args.match_threshold,
        min_box=args.min_box,
        seed=args.seed,
        dry_run=args.dry_run,
        limit_images=args.limit_images,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the parameterised cell-crop dataset.")
    p.add_argument("--width", type=int, default=None, help="default: TrainingConfig.crop_size[0]")
    p.add_argument("--height", type=int, default=None, help="default: TrainingConfig.crop_size[1]")
    p.add_argument("--margin", type=float, default=None, help="extra context, fraction of side")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--legacy-dir", default=None)
    p.add_argument("--no-legacy", action="store_true", help="annotations only")
    p.add_argument("--out", default=None)
    p.add_argument("--overrides", default=None)
    p.add_argument("--match-threshold", type=float, default=0.85)
    p.add_argument("--min-box", type=float, default=8.0, help="skip boxes with max(w,h) below this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-images", type=int, default=None, help="debug: first N images/split")
    p.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = p.parse_args()

    report = run(build_settings(args))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
