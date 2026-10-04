"""
Shared "image type" cue taxonomy for the LoRA diffusion pipeline.

Mitotic crops in ``datasets/crops/vae`` come from several sources, and each source has
a recognizable *filename structure*. We derive a small set of structural "types" (not
hard matches on specific names, only name shape) and expose them as a text cue so the
diffusion model can condition on the provenance/style of a crop:

    annotated  -> single-letter/word lab prefix + numbers, e.g. "A_208_43", "Entrega1_00017_15"
    numbered   -> pure numeric ids, e.g. "004_00100_24", "002_00033_67"
    camera     -> camera-roll Roboflow export, e.g. "IMG_1569_JPG.rf.<hash>_0"
    scraped    -> uuid-prefixed Roboflow export, e.g. "a4b667fc-...-jpg.rf.<hash>_6"
    web        -> everything else Roboflow-exported (.rf.<hash>): descriptive/stock/figure slugs

``classify_name_type`` covers 100% of the current corpus. Both the training caption
builder (scripts/utils/lora_dataset.py) and the sample generator
(scripts/generate_lora_samples.py) import from here so the cue token used at training
time is byte-identical to the one used at generation time.
"""

from __future__ import annotations

import re
from pathlib import Path

# Canonical, stable ordering of the cue tokens. Also the cycle order used when
# generating one prompt per sample with --add_type_cue.
TYPE_CUES: list[str] = ["annotated", "numbered", "web", "scraped", "camera"]

CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"
CAPTION_TEMPLATE_TYPE = CAPTION_TEMPLATE + " and type {type}"


def classify_name_type(name: str) -> str:
    """Map a crop filename to one of TYPE_CUES based purely on its name structure.

    Accepts a bare filename or a path; extension and the trailing ``_<crop index>`` are
    ignored. Unknown shapes fall back to ``"web"`` (the broadest bucket).
    """
    stem = Path(name).stem
    # Every crop name ends with a trailing "_<digits>" instance index; drop it.
    m = re.match(r"^(.*)_(\d+)$", stem)
    base = m.group(1) if m else stem

    # Roboflow web-derived exports always carry a ".rf.<hex-hash>" marker.
    has_rf = ".rf." in base

    # Lab/annotated: a short letter/word token then numbers, e.g. A_208, Entrega1_00017.
    if not has_rf and re.fullmatch(r"[A-Za-z]+\d*_\d+", base):
        return "annotated"
    # Numbered: pure numeric id groups, e.g. 004_00100.
    if not has_rf and re.fullmatch(r"\d+_\d+", base):
        return "numbered"

    if has_rf:
        if base.startswith("IMG_"):
            return "camera"
        if re.match(r"[0-9a-f]{8}-", base):
            return "scraped"
        return "web"

    # No rf marker and not an annotated/numbered shape: be conservative.
    return "web"
