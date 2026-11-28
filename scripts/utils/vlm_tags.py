"""
Build captions and generation prompts from the VLM tag JSON files.

The tag tree mirrors the 256px crop tree::

    datasets/crops/vae_256/train/tagged/prophase/foo.png
    datasets/crops/vae_256_vlm-tags/train/tagged/prophase/foo.json

Only the selected appearance fields are used.  ``detailed_mitotic_stage`` is
never read: the class folder is the source of truth for the cell phase.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_types import INTERPHASE, PHASES, phase_caption  # noqa: E402

CORE_TAG_FIELDS: list[str] = [
    "cell_shape",
    "cell_wall_contrast",
    "neighboring_cell_intrusion",
    "focus_quality",
    "central_vacuole_presence",
]

PHASE_TAG_FIELDS: dict[str, list[str]] = {
    INTERPHASE: ["interphase_chromatin_texture"],
    "anaphase": ["axis_of_division"],
    "telophase": ["axis_of_division"],
}

TAG_CUE_FIELDS: list[str] = [
    *CORE_TAG_FIELDS,
    *sorted({field for fields in PHASE_TAG_FIELDS.values() for field in fields}),
]

FIELD_PHASES: dict[str, set[str]] = {}
for _phase, _fields in PHASE_TAG_FIELDS.items():
    for _field in _fields:
        FIELD_PHASES.setdefault(_field, set()).add(_phase)

VALUE_PHRASES: dict[str, dict[str, str]] = {
    "cell_shape": {
        "elongated_rectangular": "elongated rectangular cell",
        "circular_rounded_protoplast": "circular rounded protoplast",
        "polyhedral": "polyhedral cell",
    },
    "cell_wall_contrast": {
        "faint_dark_borders": "faint dark cell borders",
        "high_contrast_bright_lines": "high contrast bright cell wall lines",
        "not_visible": "cell wall not visible",
    },
    "neighboring_cell_intrusion": {
        "faint_neighboring_walls": "faint neighboring cell walls",
        "clean_empty_borders": "clean empty borders",
        "dense_dark_neighboring_nuclei_at_edges": "dense dark neighboring nuclei at edges",
    },
    "focus_quality": {
        "sharp_high_definition": "sharp focus",
        "slightly_defocused": "slightly defocused",
    },
    "central_vacuole_presence": {
        "true": "central vacuole present",
        "false": "no central vacuole",
    },
    "interphase_chromatin_texture": {
        "smooth_solid_color": "smooth solid chromatin",
        "distinct_dark_nucleoli_visible": "distinct dark nucleoli visible",
        "mottled_granular_sponge": "mottled granular chromatin",
    },
    "axis_of_division": {
        "horizontal": "horizontal division axis",
        "vertical": "vertical division axis",
        "diagonal_top_left_to_bottom_right": "diagonal division axis",
        "diagonal_top_right_to_bottom_left": "diagonal division axis",
    },
}


SUPPORTED_VALUES: dict[str, frozenset[str]] = {
    field: frozenset(values) for field, values in VALUE_PHRASES.items()
}


class TagValidationError(ValueError):
    """Raised when manual inference tags are outside the fixed cue vocabulary."""


def _canonical_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _iter_sections(raw: dict):
    for value in raw.values():
        if isinstance(value, dict):
            yield value


def extract_tags(raw: dict) -> dict[str, str]:
    """Keep only the selected tag fields from a VLM JSON document."""
    selected: dict[str, str] = {}
    for section in _iter_sections(raw):
        for field in TAG_CUE_FIELDS:
            if field in section:
                selected[field] = _canonical_value(section[field])
    return selected


def tag_path_for(image_path: Path, crops_dir: Path, tags_dir: Path) -> Path:
    """Return the mirrored VLM tag path for a crop path."""
    try:
        rel = image_path.resolve().relative_to(crops_dir.resolve())
    except ValueError:
        rel = image_path.relative_to(crops_dir)
    return (tags_dir / rel).with_suffix(".json")


def load_vlm_tags(
    image_path: Path,
    crops_dir: Path = Path("datasets/crops/vae_256"),
    tags_dir: Path = Path("datasets/crops/vae_256_vlm-tags"),
) -> dict[str, str]:
    """Load and filter the VLM tags for one crop; missing tags return an empty dict."""
    json_path = tag_path_for(Path(image_path), Path(crops_dir), Path(tags_dir))
    if not json_path.is_file():
        return {}
    try:
        raw = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return extract_tags(raw) if isinstance(raw, dict) else {}


def validate_tags(phase: str, tags: dict[str, str] | None) -> None:
    """Fail loudly when manual inference tags are outside the fixed vocabulary.

    Training uses the permissive ``caption_for`` path, because the VLM can emit
    excluded values (e.g. ``severely_defocused``).  Manual inference uses
    ``prompt_for``, which calls this validator and rejects unknown fields,
    unknown values, and phase-specific fields used with the wrong phase.
    """
    if phase not in PHASES:
        raise TagValidationError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    if tags is None:
        return
    if not isinstance(tags, dict):
        raise TagValidationError(f"tags must be a dict, got {type(tags).__name__}")

    for field, value in tags.items():
        if field not in TAG_CUE_FIELDS:
            raise TagValidationError(
                f"Unknown tag field {field!r}; allowed fields are {', '.join(TAG_CUE_FIELDS)}"
            )
        canonical = _canonical_value(value)
        allowed_values = SUPPORTED_VALUES.get(field, frozenset())
        if canonical not in allowed_values:
            raise TagValidationError(
                f"Unsupported value {canonical!r} for {field!r}; "
                f"allowed values are {', '.join(sorted(allowed_values))}"
            )
        allowed_phases = FIELD_PHASES.get(field)
        if allowed_phases is not None and phase not in allowed_phases:
            raise TagValidationError(
                f"{field} is only valid for phases {', '.join(sorted(allowed_phases))}; "
                f"got {phase!r}"
            )


def filter_supported_tags(tags: dict[str, str] | None) -> dict[str, str]:
    """Keep only known fields with dictionary-supported values (training path)."""
    if not tags:
        return {}
    filtered: dict[str, str] = {}
    for field, value in tags.items():
        if field not in TAG_CUE_FIELDS:
            continue
        canonical = _canonical_value(value)
        if canonical in SUPPORTED_VALUES.get(field, frozenset()):
            filtered[field] = canonical
    return filtered


def tag_phrases(phase: str, tags: dict[str, str] | None, *, strict: bool = False) -> list[str]:
    """Return the ordered cue phrases applicable to a phase.

    ``strict=False`` is the training/synthetic-pool path: unsupported values and
    phase-inapplicable fields are omitted.  ``strict=True`` is the manual
    inference path: invalid tags raise ``TagValidationError``.
    """
    if strict:
        validate_tags(phase, tags)
    if not tags:
        return []
    fields = [*CORE_TAG_FIELDS, *PHASE_TAG_FIELDS.get(phase, [])]
    phrases: list[str] = []
    for field in fields:
        if field not in tags:
            continue
        canonical = _canonical_value(tags[field])
        phrase = VALUE_PHRASES.get(field, {}).get(canonical)
        if phrase:
            phrases.append(phrase)
        elif strict:
            raise TagValidationError(
                f"Unsupported value {canonical!r} for {field!r}; "
                f"allowed values are {', '.join(sorted(SUPPORTED_VALUES[field]))}"
            )
    return phrases


def caption_for(phase: str, tags: dict[str, str] | None = None) -> str:
    """Permissive training caption builder: drop unsupported/phase-inapplicable tags."""
    base = phase_caption(phase)
    phrases = tag_phrases(phase, tags, strict=False)
    return ", ".join([base, *phrases]) if phrases else base


def prompt_for(phase: str, tags: dict[str, str] | None = None) -> str:
    """Strict manual-inference prompt builder: validate tags and render all cues."""
    base = phase_caption(phase)
    phrases = tag_phrases(phase, tags, strict=True)
    return ", ".join([base, *phrases]) if phrases else base


def build_train_tag_index(
    crops_dir: Path = Path("datasets/crops/vae_256"),
    tags_dir: Path = Path("datasets/crops/vae_256_vlm-tags"),
    phases: list[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Collect the tag tuples observed in the official train split, per phase."""
    phases = list(PHASES) if phases is None else list(phases)
    index: dict[str, list[dict[str, str]]] = {}
    for phase in phases:
        phase_dir = Path(crops_dir) / "train" / "tagged" / phase
        entries: list[dict[str, str]] = []
        if phase_dir.is_dir():
            for image_path in sorted(phase_dir.glob("*.png")):
                tags = filter_supported_tags(load_vlm_tags(image_path, crops_dir, tags_dir))
                tags = {
                    field: value
                    for field, value in tags.items()
                    if field in CORE_TAG_FIELDS or phase in FIELD_PHASES.get(field, set())
                }
                if tags:
                    entries.append(tags)
        index[phase] = entries
    return index


def sample_tag_tuple(
    phase: str,
    tag_index: dict[str, list[dict[str, str]]],
    rng: random.Random,
) -> dict[str, str]:
    """Sample an observed train tag tuple for a phase (empty dict if none exist)."""
    entries = tag_index.get(phase, [])
    return rng.choice(entries) if entries else {}
