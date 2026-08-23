#!/bin/bash
# 1) Train + evaluate dataset arm D (per_phase_d4x3_nojitter), ~1.5h.
# 2) Pick the winning dataset by composite score between arm A (p5_noaug) and arm D.
# 3) Launch the Optuna study on the winner, target 30 trials, ~30h.
#
# The winner criterion is the study's own objective, so this is not a judgement call:
#   score = phase_consistency - 0.135*kid_classifier + 0.847*coverage - 5*max(0, mem_excess)
#
# RESUMABLE. Completed trials are committed to study.db as they finish. After a shutdown:
#   uv run python scripts/optimize_lora.py --study-name lora_nofill_v1 --n-trials 30 --reset-stale
# (--n-trials is a TARGET TOTAL, not an increment. --reset-stale clears trials left in RUNNING
#  by a hard kill. The dataset_version is pinned in the study and mismatches are a hard error.)

set -u
cd /home/nicolas/Documentos/UTN/INA/giar_ina_dev

STUDY=lora_nofill_v1

if pgrep -f "train_network.py" > /dev/null; then
  echo "REFUSING: a kohya training run is already on the GPU."
  exit 1
fi

echo "MARKER_ARMD_TRAIN_START $(date +%F_%H:%M)"
uv run python scripts/train_lora.py --config experiments/lora/p5_d4x3_nojitter/config.yaml \
    > /tmp/p5_d4x3_nojitter_train.log 2>&1
echo "MARKER_ARMD_TRAIN_DONE rc=$? $(date +%F_%H:%M)"

echo "MARKER_ARMD_EVAL_START $(date +%F_%H:%M)"
uv run python scripts/evaluate_lora.py --config experiments/lora/p5_d4x3_nojitter/config.yaml \
    --samples 100 > /tmp/p5_d4x3_nojitter_eval.log 2>&1
echo "MARKER_ARMD_EVAL_DONE rc=$? $(date +%F_%H:%M)"

echo
echo "=== dataset ablation, all four arms (identical hyperparameters) ==="
WINNER=$(python3 - <<'PY'
import json
from pathlib import Path

W_K, W_C, W_M = 0.135, 0.847, 5.0
arms = [
    ("C  p3_per_phase_nofill", "p3_per_phase_nofill", "per_phase_nofill",         "resample+jitter"),
    ("A  p5_noaug",            "p5_noaug",            "per_phase_noaug",          "none"),
    ("B3 p5_d4x3",             "p5_d4x3",             "per_phase_d4x3",           "D4+jitter"),
    ("D  p5_d4x3_nojitter",    "p5_d4x3_nojitter",    "per_phase_d4x3_nojitter",  "D4, no jitter"),
]

rows = []
for label, exp, version, desc in arms:
    p = Path(f"experiments/lora/{exp}/metrics.json")
    if not p.exists():
        continue
    m = json.load(open(p))
    s = (m["phase_consistency"] - W_K * m["kid_classifier"] + W_C * m["coverage"]
         - W_M * max(0.0, m["memorization_excess_p95"]))
    rows.append((s, label, version, desc, m))

hdr = f"{'arm':<24}{'score':>8}{'pc':>7}{'kid':>8}{'cov':>7}{'recon':>8}  augmentation"
print(hdr); print("-" * len(hdr))
for s, label, _, desc, m in sorted(rows, reverse=True):
    print(f"{label:<24}{s:>8.3f}{m['phase_consistency']:>7.3f}{m['kid_classifier']:>8.3f}"
          f"{m['coverage']:>7.3f}{m['vqgan_recon_ratio']:>8.3f}  {desc}")

# Only A and D are candidates: C and B3 are already beaten and carry the jitter.
candidates = [r for r in rows if r[2] in ("per_phase_noaug", "per_phase_d4x3_nojitter")]
best = max(candidates)
print(f"\nWINNER: {best[1]}  (dataset_version={best[2]}, score {best[0]:.3f})")
print("Note: phase_consistency noise is ~6.3% run-to-run; kid ~1.8%, coverage ~1.5%.")
Path("/tmp/winner_dataset").write_text(best[2])
PY
)
echo "$WINNER"
DATASET=$(cat /tmp/winner_dataset)
echo "MARKER_WINNER ${DATASET} $(date +%F_%H:%M)"

echo
echo "MARKER_STUDY_START ${STUDY} on ${DATASET} $(date +%F_%H:%M)"
uv run python scripts/optimize_lora.py \
    --study-name "${STUDY}" \
    --dataset-version "${DATASET}" \
    --n-trials 30 \
    --samples 100
echo "MARKER_STUDY_DONE rc=$? $(date +%F_%H:%M)"
echo "MARKER_ALL_COMPLETE"
