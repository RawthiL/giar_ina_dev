"""
Shared phase vocabulary for the diffusion pipelines.

The cell class of a crop is always taken from its folder name (``prophase``,
``metaphase``, ``anaphase``, ``telophase``, ``interphase``).  The VLM tag
``detailed_mitotic_stage`` is deliberately ignored: the class folder is the
source of truth.  Appearance conditioning comes from ``vlm_tags`` instead of
the old provenance source cue.
"""

from __future__ import annotations

# Every generatable cell class in the LoRA vocabulary.
MITOTIC_PHASES: list[str] = ["prophase", "metaphase", "anaphase", "telophase"]
INTERPHASE = "interphase"
PHASES: list[str] = [*MITOTIC_PHASES, INTERPHASE]

CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"
CAPTION_TEMPLATE_INTERPHASE = "micrograph of allium cepa root tip cell in interphase"


def phase_caption(phase: str) -> str:
    """Return the phase-only caption for a cell class."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    if phase == INTERPHASE:
        return CAPTION_TEMPLATE_INTERPHASE
    return CAPTION_TEMPLATE.format(phase=phase)
