"""
Build the cell-crop dataset (``datasets/crops/vae_256``) from the merged COCO annotations
plus the manually tagged legacy crops in ``datasets/crops/vae``.

Cropping: a square window centred on each bbox with side ``max(w, h) * (1 + margin)``,
resized to ``--width x --height``. Aspect ratio is preserved (the legacy crops were squished,
this does not do that). No pixels are ever synthesised: a cell whose window is not fully
inside the image (a border cell) is dropped, never padded or mirrored.

Tags (annotation attributes after ``annotation_overrides.yaml``):
  mitosis_stage 1/2/3     -> tagged/{metaphase,anaphase,telophase}   (division forced True)
  division=True, stage 0  -> untagged/mitotic   (mitotic, phase unknown)
  division=False          -> tagged/interphase
  tagged/prophase         -> legacy manual crops only (annotations cannot identify prophase)

Legacy crops are authoritative for the phase (their folder). ``rescan_legacy.py`` matches them
to annotations with LPIPS (``legacy_matches.json``). A matched legacy crop is replaced by a fresh
crop of that annotation from the full image (phase from the legacy folder), unless the cell is a
border cell, in which case it is dropped. Unmatched legacy crops are kept, resized to the target
size and renamed ``{origin}_legacy_{id}_{idx}``. Annotations with ``attributes.phase == "REVIEW"``
(new YOLO cells awaiting a manual phase) are skipped; a real class name there sets the tag.

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
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from allium_cepa_classifier.config.training_config import TrainingConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

from legacy_match import square_window, window_inside  # noqa: E402

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
INTERPHASE = "interphase"
CLASSES = [*PHASES, INTERPHASE]
SPLITS = ["train", "validation", "test"]
OUT_SPLIT = {"train": "train", "validation": "val", "test": "test"}
# Cross-split duplicates are removed from the lowest-priority split first.
DEDUP_KEEP_PRIORITY = {"test": 0, "validation": 1, "train": 2}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
REVIEW = "REVIEW"  # attributes.phase of rescanned cells awaiting a manual phase

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


def crop_square_resized(
    arr: np.ndarray, bbox: list[float], size: tuple[int, int], margin: float = 0.0
) -> Image.Image | None:
    """Square max(w,h) crop resized to ``size`` (W, H); None for border cells (no padding)."""
    win = square_window(bbox, margin)
    if not window_inside(win, arr.shape[1], arr.shape[0]):
        return None
    left, top, right, bottom = win
    return Image.fromarray(arr[top:bottom, left:right]).resize(size, Image.LANCZOS)


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
    pending: bool = False  # rescanned cell awaiting a manual phase (attributes.phase == REVIEW)
    border: bool = False  # square crop window leaves the image


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
    duplicate_of: str | None = None  # another legacy crop of the same cell that was kept


@dataclass
class Settings:
    raw_dir: Path
    legacy_dir: Path | None
    out_dir: Path
    overrides: Path
    matches_path: Path | None = None  # legacy_matches.json from rescan_legacy.py
    size: tuple[int, int] = (256, 256)
    margin: float = 0.0
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
                "width": int(im["width"]),
                "height": int(im["height"]),
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
            pending = False
            phase_attr = str(attrs.get("phase", ""))
            if phase_attr.upper() == REVIEW:
                pending = True
            elif phase_attr in CLASSES:  # rescanned cell: explicit (legacy / hand-set) phase
                tag, division = phase_attr, phase_attr in PHASES
            elif phase_attr:
                cfg.notes.append(f"unknown attributes.phase {phase_attr!r} on {split}/{a['id']}")
            bbox = [float(v) for v in a["bbox"]]
            anns.append(
                Ann(
                    split=split,
                    ann_id=int(a["id"]),
                    image_key=stem,
                    image_path=images[stem]["path"],
                    origin=images[stem]["origin"],
                    bbox=bbox,
                    division=division,
                    stage=stage,
                    tag=tag,
                    pending=pending,
                    border=not window_inside(
                        square_window(bbox, cfg.margin),
                        images[stem]["width"],
                        images[stem]["height"],
                    ),
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
) -> int:
    """Set ``lg.split`` from the source image and ``lg.ann`` from ``legacy_matches.json``.

    The matches are produced by ``rescan_legacy.py`` (LPIPS). Returns the matches applied.
    """
    prefix_to_splits: dict[str, Counter] = defaultdict(Counter)
    for stem, meta in images.items():
        prefix_to_splits[source_group(stem)][meta["split"]] += 1
    for lg in legacy:
        if lg.stem in images:
            lg.split = images[lg.stem]["split"]
        else:
            splits = prefix_to_splits.get(source_group(lg.stem))
            if splits:
                lg.split = splits.most_common(1)[0][0]
                lg.note = "split inherited from re-export of same source"

    if cfg.matches_path is None or not cfg.matches_path.exists():
        cfg.notes.append(f"no legacy matches file ({cfg.matches_path}): legacy crops all kept")
        return 0
    matches = json.loads(cfg.matches_path.read_text())
    by_id = {(a.split, a.ann_id): a for a in anns}
    applied = 0
    for lg in legacy:
        m = matches.get(lg.out_name)
        if m is None:
            continue
        ann = by_id.get((m["split"], int(m["ann_id"])))
        if ann is None or ann.image_key != lg.stem:
            cfg.notes.append(f"stale legacy match for {lg.out_name}: annotation not found")
            continue
        if ann.consumed_by is not None:
            # Two legacy crops of one cell: keep the closer one, the other is a duplicate.
            other = ann.consumed_by
            if m["lpips"] >= ann.match_score:
                lg.duplicate_of = other
                continue
            for o in legacy:
                if o.out_name == other:
                    o.ann, o.duplicate_of = None, lg.out_name
        lg.ann, lg.score = ann, float(m["lpips"])
        ann.consumed_by, ann.match_score = lg.out_name, float(m["lpips"])
        applied += 1
    return applied


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

    def ann_row(a: Ann, **kw) -> dict:
        return row(
            split=a.split,
            origin=a.origin,
            source_image=images[a.image_key]["file_name"],
            annotation_id=a.ann_id,
            bbox=json.dumps([round(v, 2) for v in a.bbox]),
            **kw,
        )

    def emit(
        a: Ann, tag: str, arr: np.ndarray, untagged_kind: str | None = None, **kw
    ) -> dict | None:
        """Crop ``a`` into ``tag``'s folder; None (and nothing written) for border cells."""
        dest = dest_dir(cfg.out_dir, a.split, tag, untagged_kind)
        assert dest is not None
        crop = crop_square_resized(arr, a.bbox, cfg.size, cfg.margin)
        if crop is None:
            return None
        name = ann_output_name(a, images)
        digest = _save(crop, dest / name)
        return ann_row(
            a,
            output=str((dest / name).relative_to(cfg.out_dir)),
            tag=tag,
            untagged_kind=(untagged_kind or "mitotic") if tag == "mitotic" else "",
            division="" if a.pending else int(a.division),
            mitosis_stage="" if a.pending else a.stage,
            content_hash=digest,
            **kw,
        )

    by_image: dict[str, list[Ann]] = defaultdict(list)
    for a in anns:
        if a.consumed_by is None:
            by_image[a.image_key].append(a)

    for _stem, items in sorted(by_image.items()):
        arr = None
        for a in items:
            if max(a.bbox[2], a.bbox[3]) < cfg.min_box:
                stats["skipped_tiny_box"] += 1
                continue
            if a.pending:
                a.tag = "mitotic"  # phase unknown: unverified cells go to untagged/mixed
            dest = dest_dir(cfg.out_dir, a.split, a.tag, "mixed" if a.pending else None)
            if dest is None:
                stats[f"skipped_test_untagged_{a.tag}"] += 1
                continue
            if a.border:
                stats["dropped_border"] += 1
                rows.append(ann_row(a, tag=a.tag, dropped="border cell (crop window leaves image)"))
                continue
            if cfg.dry_run:
                stats[f"{a.split}/{'mixed' if a.pending else a.tag}"] += 1
                continue
            if arr is None:
                arr = np.asarray(Image.open(a.image_path).convert("RGB"))
            r = emit(
                a,
                a.tag,
                arr,
                legacy=0,
                untagged_kind="mixed" if a.pending else None,
                note="REVIEW: unverified YOLO detection, phase unknown" if a.pending else "",
            )
            if r is None:  # image size disagrees with the COCO record
                stats["skipped_outside_image"] += 1
                continue
            rows.append(r)
            stats[f"{a.split}/{'mixed' if a.pending else a.tag}"] += 1

    for lg in legacy:
        assert lg.split is not None
        dest = dest_dir(cfg.out_dir, lg.split, lg.phase)
        assert dest is not None
        base = {
            "tag": lg.phase,
            "split": lg.split,
            "origin": classify_origin(lg.stem),
            "source_image": lg.stem,
            "legacy_original": lg.path.name,
        }
        if lg.duplicate_of:
            stats["legacy_dropped_duplicate_cell"] += 1
            rows.append(
                row(
                    **base,
                    legacy=1,
                    dropped=f"duplicate legacy crop of cell kept as {lg.duplicate_of}",
                )
            )
            continue
        if lg.ann is not None and lg.ann.border:
            stats["legacy_dropped_border"] += 1
            rows.append(
                row(
                    **base,
                    legacy=1,
                    annotation_id=lg.ann.ann_id,
                    match_score=round(lg.score, 4),
                    dropped="legacy crop matches a border cell",
                )
            )
            continue
        if lg.ann is not None:
            # Replace the upscaled 200x200 legacy crop by a native crop of the matched cell.
            stats[f"{lg.split}/{lg.phase}(legacy_recropped)"] += 1
            if cfg.dry_run:
                continue
            arr = np.asarray(Image.open(lg.ann.image_path).convert("RGB"))
            note = [lg.note] if lg.note else []
            note.append(
                f"recropped from full image, replaces legacy crop {lg.path.name} "
                f"(annotation stage {lg.ann.stage}, {lg.ann.tag}; phase from legacy folder)"
            )
            if lg.ann.tag in PHASES and lg.ann.tag != lg.phase:
                stats["stage_folder_conflicts"] += 1
                note.append(f"CONFLICT stage says {lg.ann.tag}; folder wins")
            r = emit(
                lg.ann,
                lg.phase,
                arr,
                legacy=0,
                legacy_original=lg.path.name,
                match_score=round(lg.score, 4),
                note="; ".join(note),
            )
            if r is None:
                stats["skipped_outside_image"] += 1
                continue
            r["division"] = 1
            rows.append(r)
            continue
        stats[f"{lg.split}/{lg.phase}(legacy)"] += 1
        if cfg.dry_run:
            continue
        img = Image.open(lg.path).convert("RGB").resize(cfg.size, Image.LANCZOS)
        digest = _save(img, dest / lg.out_name)
        note = [lg.note] if lg.note else []
        note.append("legacy: upscaled from 200x200 (squished at source, aspect not recoverable)")
        rows.append(
            row(
                **base,
                output=str((dest / lg.out_name).relative_to(cfg.out_dir)),
                division=1,
                legacy=1,
                content_hash=digest,
                note="; ".join(note),
            )
        )
    return rows, stats


def drop_cross_split_duplicates(rows: list[dict], cfg: Settings) -> int:
    """Identical pixels in two splits are leakage: delete the copy in the lowest-priority split."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["content_hash"] and not r["dropped"]:
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
    matched = match_legacy(legacy, anns, images, cfg)
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
        "legacy_matches_applied": matched,
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
        matches_path=Path(args.matches)
        if args.matches
        else root / "scripts" / "utils" / "legacy_matches.json",
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
    p.add_argument("--matches", default=None, help="legacy_matches.json (from rescan_legacy.py)")
    p.add_argument("--min-box", type=float, default=8.0, help="skip boxes with max(w,h) below this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-images", type=int, default=None, help="debug: first N images/split")
    p.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = p.parse_args()

    report = run(build_settings(args))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
