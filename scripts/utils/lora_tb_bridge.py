"""
Bridge kohya-generated sample PNGs into the experiment's TensorBoard IMAGES tab.

kohya writes samples to <run_dir>/weights/sample/ but only logs them to wandb.
This script reads those PNGs and writes them to the TensorBoard event file under
<run_dir>/logs/ so they appear in the IMAGES tab alongside scalar loss curves.

Filename format (sampling.py:488):
  {output_name}_{num_suffix}_{i:02d}_{ts_str}[_{seed}].png
  num_suffix = e{epoch:06d}  (epoch-based)  or  {steps:06d}  (step-based)
  ts_str     = 14 decimal digits (%Y%m%d%H%M%S)

Usage (standalone):
    uv run python scripts/utils/lora_tb_bridge.py <run_dir>
"""

import argparse
import re
import sys
from pathlib import Path

# Matches the fixed-length suffix: _{num_suffix}_{i:02d}_{14-digit ts}[_{seed}]
_FILENAME_RE = re.compile(r"_((?:e\d{6}|\d{6}))_(\d{2})_\d{14}(?:_\d+)?$")


def _parse(stem: str) -> tuple[int, int]:
    """Return (global_step, prompt_index) parsed from a kohya sample filename stem."""
    m = _FILENAME_RE.search(stem)
    if m is None:
        raise ValueError(f"Cannot parse kohya sample filename: {stem!r}")
    num_suffix, idx_str = m.group(1), m.group(2)
    # epoch-based: e000001 → step = 1; step-based: 000100 → step = 100
    step = int(num_suffix[1:]) if num_suffix.startswith("e") else int(num_suffix)
    return step, int(idx_str)


def bridge_samples(run_dir: Path) -> int:
    """Write kohya sample PNGs from <run_dir>/weights/sample/ into TensorBoard.

    Returns the number of images written.
    """
    from PIL import Image
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.transforms.functional import to_tensor

    sample_dir = run_dir / "weights" / "sample"
    if not sample_dir.exists():
        return 0

    pngs = sorted(sample_dir.glob("*.png"))
    if not pngs:
        return 0

    writer = SummaryWriter(log_dir=str(run_dir / "logs"))
    count = 0
    for png in pngs:
        try:
            step, idx = _parse(png.stem)
        except ValueError as e:
            print(f"  [tb-bridge] skip {png.name}: {e}", file=sys.stderr)
            continue
        img = to_tensor(Image.open(png).convert("RGB"))
        writer.add_image(f"sample/{idx:02d}", img, global_step=step)
        count += 1
    writer.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge kohya sample PNGs → TensorBoard.")
    parser.add_argument(
        "run_dir", type=Path, help="Experiment directory (contains weights/ and logs/)."
    )
    args = parser.parse_args()

    n = bridge_samples(args.run_dir)
    if n:
        print(f"[tb-bridge] wrote {n} images → {args.run_dir / 'logs'}")
    else:
        print(f"[tb-bridge] no samples found in {args.run_dir / 'weights' / 'sample'}")


if __name__ == "__main__":
    main()
