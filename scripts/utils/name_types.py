"""
Shared phase/caption vocabulary for the diffusion pipelines.

The 256px cell-crop dataset (``datasets/crops/vae_256``) encodes each crop's provenance
directly in its filename as a leading ``<source>_`` token:

    ina      -> UTN/INA lab scans, e.g. "ina_004_00054_23354.png", "ina_legacy_005_..._59.png"
    onion    -> onion root-tip series, e.g. "onion_A_208_43.png"
    roboflow -> Roboflow-exported cells, e.g. "roboflow_108_21797.png"

The source is the first underscore-delimited token, so the cue is trivially readable from
the name and equally easy to strip (drop the prefix) when provenance must be hidden. This
replaces the old structural taxonomy (annotated/numbered/web/scraped/camera) that had to be
*inferred* from name shape on the legacy 200px corpus.

``caption_for`` is the single caption builder shared by the LoRA caption builder
(scripts/utils/lora_dataset.py), the LoRA sample generator (scripts/generate_lora_samples.py)
and the diffuser latent generator (scripts/generate_diffuser_latent_dataset.py), so the exact
token string used at training time is byte-identical to the one used at generation time.
Interphase is a phase label but NOT a mitosis state, so it gets its own template
("root tip cell in interphase") instead of "... mitotic cell in interphase phase".
"""

from __future__ import annotations

from pathlib import Path

# Canonical, stable ordering of the source cue tokens. Also the cycle order used when
# generating one prompt per sample with --add_source_cue.
SOURCE_CUES: list[str] = ["ina", "onion", "roboflow"]

# Every generatable cell class in the LoRA vocabulary.
MITOTIC_PHASES: list[str] = ["prophase", "metaphase", "anaphase", "telophase"]
INTERPHASE = "interphase"
PHASES: list[str] = [*MITOTIC_PHASES, INTERPHASE]

CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"
CAPTION_TEMPLATE_INTERPHASE = "micrograph of allium cepa root tip cell in interphase"


def source_from_name(name: str) -> str | None:
    """Return the provenance source encoded in a crop filename's leading token.

    Accepts a bare filename or a path. Returns one of SOURCE_CUES, or None when the name
    carries no known source prefix (e.g. the legacy 200px corpus).
    """
    stem = Path(name).stem
    prefix = stem.split("_", 1)[0]
    return prefix if prefix in SOURCE_CUES else None


def caption_for(phase: str, source: str | None = None) -> str:
    """Build the canonical caption/prompt for a cell class, optionally with a source cue.

    Mitotic phases use the "mitotic cell in <phase> phase" template; interphase (not a
    division state) uses "root tip cell in interphase". When ``source`` is given it is
    appended as " and source <cue>".
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    base = (
        CAPTION_TEMPLATE_INTERPHASE
        if phase == INTERPHASE
        else CAPTION_TEMPLATE.format(phase=phase)
    )
    return f"{base} and source {source}" if source else base
