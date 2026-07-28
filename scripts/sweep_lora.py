"""
Sweep all LoRA experiments: train → generate samples → evaluate → log HParams to TensorBoard.

For each config, the sweep:
  1. Trains the LoRA (calls train_lora.py, which also bridges samples → TensorBoard).
  2. Generates sample images (generate_lora_samples.py), unless --skip-generate.
  3. Evaluates and writes metrics.json (evaluate_lora.py).
  4. Logs one HParams row per experiment into a shared sweep TensorBoard logdir.

After all configs, prints a summary ranked by final_loss.

Usage:
    uv run python scripts/sweep_lora.py --configs experiments/lora/*/config.yaml
    uv run python scripts/sweep_lora.py --configs experiments/lora/*/config.yaml --skip-generate
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> int:
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep all LoRA experiments.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--skip-generate", action="store_true", help="Skip sample generation.")
    args = parser.parse_args()

    sweep_dir = _ROOT / "experiments" / "lora" / "_sweeps" / time.strftime("%Y%m%d-%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSweep logdir: {sweep_dir}")

    results: list[tuple[str, str, dict]] = []

    for cfg_path in args.configs:
        cfg_path = cfg_path.resolve()
        print(f"\n{'=' * 60}")
        print(f"Experiment: {cfg_path.parent.name}  [{cfg_path}]")
        print(f"{'=' * 60}")

        # 1. Train (includes TB bridge for sample images)
        rc = _run([sys.executable, "scripts/train_lora.py", "--config", str(cfg_path)])
        status = "OK" if rc == 0 else "TRAIN_FAILED"

        # 2. Generate samples
        if status == "OK" and not args.skip_generate:
            rc = _run(
                [sys.executable, "scripts/generate_lora_samples.py", "--config", str(cfg_path)]
            )
            if rc != 0:
                status = "GENERATE_FAILED"

        # 3. Evaluate → metrics.json
        if status == "OK":
            rc = _run([sys.executable, "scripts/evaluate_lora.py", "--config", str(cfg_path)])
            if rc != 0:
                status = "EVAL_FAILED"

        # 4. Load config + metrics for HParams
        cfg = LoRAExperimentConfig.from_yaml(cfg_path)
        metrics_path = cfg_path.parent / "metrics.json"
        metrics: dict = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                pass

        from torch.utils.tensorboard import SummaryWriter

        hparams = {
            "experiment": cfg.experiment_name,
            "dataset_version": cfg.data.dataset_version,
            "network_dim": cfg.network.network_dim,
            "network_alpha": cfg.network.network_alpha,
            "learning_rate": cfg.training.learning_rate,
            "max_train_steps": cfg.training.max_train_steps,
            "noise_offset": cfg.training.noise_offset,
            "min_snr_gamma": cfg.training.min_snr_gamma,
            "ip_noise_gamma": cfg.training.ip_noise_gamma,
        }
        metric_values = {
            f"hparam/{k}": v
            for k, v in metrics.items()
            if isinstance(v, int | float) and v is not None
        }
        with SummaryWriter(log_dir=str(sweep_dir / cfg.experiment_name)) as w:
            w.add_hparams(hparams, metric_values or {"hparam/status": 0.0})

        results.append((cfg.experiment_name, status, metrics))

    # Summary ranked by final_loss (None sorts last)
    print(f"\n\n{'=' * 60}")
    print(f"Sweep complete → {sweep_dir}")
    print(f"{'=' * 60}")
    results.sort(
        key=lambda x: (x[2].get("final_loss") is None, x[2].get("final_loss", float("inf")))
    )
    print(f"{'Experiment':<24} {'Status':<18} {'final_loss':>12} {'min_loss':>10}")
    print("-" * 68)
    for name, status, m in results:
        fl = m.get("final_loss")
        ml = m.get("min_loss")
        print(
            f"{name:<24} {status:<18} "
            f"{fl if fl is not None else 'N/A':>12} "
            f"{ml if ml is not None else 'N/A':>10}"
        )
    print(f"\nView HParams:  tensorboard --logdir {sweep_dir}")


if __name__ == "__main__":
    main()
