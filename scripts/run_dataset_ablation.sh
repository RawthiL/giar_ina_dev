#!/bin/bash
# Dataset ablation: does offline augmentation help, and does lossless orientation beat blurry
# rotation? Three arms at IDENTICAL hyperparameters, so only the training data differs.
#
#   A  p5_noaug   per_phase_noaug   1194 images, no augmented copies
#   B3 p5_d4x3    per_phase_d4x3    4776 images, 3 distinct D4 transposes each, zero resampling
#   C  (already measured) p3_per_phase_nofill  2388 images, +-15deg rotation with resampling
#        phase_consistency 0.8375 | kid_classifier 1.273 | coverage 0.340 | recon_ratio 0.444
#
# ~3h total on one 24 GB card (2 x [~1h15m train + ~15m eval]). Nothing else may use the GPU.
#
# Read the result as:
#   A ~= C            -> augmented copies add nothing; simplify to the unique set
#   B3 recon_ratio -> ~1.0, coverage held  -> lossless orientation wins; adopt it and drop
#                                             rotate_without_fill from the augmenters
#   B3 coverage falls -> 8 discrete orientations were not enough; the continuum mattered
#
# Differences under ~0.05 phase_consistency are inside run-to-run noise (same config measured
# 0.752 and 0.800 on two runs). vqgan_recon_ratio is the metric with a large mechanistic effect
# here, so it is the one a single run per arm can actually settle.
#
# NOTE: if p4_r16_nofill beats p3_per_phase_nofill's 0.8375, rank 16 is the better base and both
# configs should switch to network_dim 16 / network_alpha 16 before running, so the comparison
# baseline becomes p4_r16_nofill rather than p3_per_phase_nofill.

set -u
cd /home/nicolas/Documentos/UTN/INA/giar_ina_dev

if pgrep -f "train_network.py" > /dev/null; then
  echo "REFUSING: a kohya training run is already on the GPU. Wait for it to finish."
  exit 1
fi

for exp in p5_noaug p5_d4x3; do
  echo "MARKER_${exp}_TRAIN_START $(date +%F_%H:%M)"
  uv run python scripts/train_lora.py --config "experiments/lora/${exp}/config.yaml" \
      > "/tmp/${exp}_train.log" 2>&1
  echo "MARKER_${exp}_TRAIN_DONE rc=$? $(date +%F_%H:%M)"

  echo "MARKER_${exp}_EVAL_START $(date +%F_%H:%M)"
  uv run python scripts/evaluate_lora.py --config "experiments/lora/${exp}/config.yaml" \
      --samples 100 > "/tmp/${exp}_eval.log" 2>&1
  echo "MARKER_${exp}_EVAL_DONE rc=$? $(date +%F_%H:%M)"
done

echo
echo "=== dataset ablation summary (all at identical hyperparameters) ==="
python3 - <<'PY'
import json
from pathlib import Path

arms = [
    ("C  p3_per_phase_nofill", "experiments/lora/p3_per_phase_nofill/metrics.json"),
    ("A  p5_noaug",            "experiments/lora/p5_noaug/metrics.json"),
    ("B3 p5_d4x3",             "experiments/lora/p5_d4x3/metrics.json"),
]
hdr = f"{'arm':<24}{'phase_c':>9}{'kid_cls':>9}{'kid_vq':>9}{'cover':>8}{'dens':>8}{'recon_r':>9}{'mem_ex':>9}"
print(hdr); print("-" * len(hdr))
for label, path in arms:
    if not Path(path).exists():
        print(f"{label:<24}  (no metrics.json)"); continue
    m = json.load(open(path))
    print(f"{label:<24}{m['phase_consistency']:>9.3f}{m['kid_classifier']:>9.3f}"
          f"{m['kid_vqgan']:>9.3f}{m['coverage']:>8.3f}{m['density']:>8.3f}"
          f"{m['vqgan_recon_ratio']:>9.3f}{m['memorization_excess_p95']:>9.3f}")
print("\nper-phase phase_consistency:")
for label, path in arms:
    if Path(path).exists():
        print(f"  {label:<24}{json.load(open(path))['phase_consistency_per_phase']}")
print("\nrecon_ratio target is ~1.0 in BOTH directions (>1 artifacts, <1 oversmoothing).")
PY

echo "MARKER_ABLATION_COMPLETE"
