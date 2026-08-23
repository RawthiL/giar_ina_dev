"""
Optuna hyperparameter search over LoRA configs, scored with the Phase 2 metric stack.

Each trial writes a full experiment under `experiments/lora/_studies/<study>/trial_XXX/`,
trains it, generates samples, evaluates, and returns a composite score. Trials are therefore
ordinary experiments — inspectable, re-runnable, and comparable with the hand-built ones.

**Resumable.** State lives in a SQLite file, so the study survives Ctrl-C, a reboot, or a
multi-day gap: re-run the same command and it continues from the trials already recorded. This
is why Optuna rather than scikit-optimize — at ~1h per trial, a sweep that cannot resume is
unusable, and the space has conditional dimensions (a trick is present/absent, then valued)
that skopt handles poorly.

**No pruning.** The plan floated pruning on mid-training samples, but kohya emits exactly one
image per prompt per sample event — far too few for a usable intermediate metric — and staged
train/resume costs more complexity than the ~1.5x throughput it would buy. Instead
`max_train_steps` is a searched dimension, so Optuna can find cheap-and-good regions itself.

**The dataset is part of the objective.** `--dataset-version` (default `per_phase_nofill`) is
recorded in the study's `user_attrs` and a mismatch on resume is a hard error. This was hardcoded
to `per_phase` until 2026-08-15, which silently pinned the first 21-trial study to the dataset
carrying the dark-red rotation wedges; the fix for that was worth +0.085 phase_consistency,
larger than most of the gaps the study was ranking.

Usage:
    uv run python scripts/optimize_lora.py --study-name lora_nofill --n-trials 30
    uv run python scripts/optimize_lora.py --study-name lora_nofill --n-trials 30 --timeout 43200
    uv run python scripts/optimize_lora.py --study-name lora_nofill --report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import yaml

from allium_cepa_classifier.config.base_config import find_project_root

_ROOT = find_project_root()
_STUDY_ROOT = _ROOT / "experiments/lora/_studies"

# --- Objective weights -------------------------------------------------------
# Weights are set against each metric's observed SPREAD, so no term silently dominates.
# Intended priority: phase_consistency contributes ~2x kid and ~2x coverage, because a
# phase-inaccurate generation is label noise in the very class the project exists to fix,
# whereas a slightly-off texture is merely a worse training image.
#
# Spreads measured across all 30 re-scored experiments on 2026-08-11:
#   phase_consistency 0.635 (0.215-0.850) | kid_classifier 2.343 (1.024-3.368)
#   coverage          0.375 (0.055-0.430)
# Weights below give contributions of 0.635 / 0.318 / 0.318 — exactly the intended 2:1:1.
#
# **The kid weight is scale-critical.** Fixing the KID estimator on 2026-08-10 (L2-normalised
# features) moved its scale from ~0.2 to ~1-3.4. The weight of 1.0 that had been balanced for
# the old scale left kid dominating phase_consistency ~20:1 and ranked a memoriser above an
# honest run — caught by test_score_penalises_memorisation_below_a_weaker_but_honest_run.
# If the feature space or estimator changes again, RE-DERIVE these from the new spreads.
W_PHASE_CONSISTENCY = 1.0
W_KID = 0.135
W_COVERAGE = 0.847
# Memorisation is a guard, not a goal: zero cost until generated images sit closer to the
# training set than real crops sit to each other, then a steep penalty.
MEMORIZATION_CEILING = 0.0
W_MEMORIZATION = 5.0


# Mirror of the space declared in build_config, needed to replay finished trials into a new
# study via `create_trial`. `_check_space_matches` asserts the two never drift apart.
DISTRIBUTIONS: dict[str, optuna.distributions.BaseDistribution] = {
    "network_dim": optuna.distributions.CategoricalDistribution([16, 32, 64, 128, 256]),
    "alpha_ratio": optuna.distributions.CategoricalDistribution([0.25, 0.5, 1.0]),
    "learning_rate": optuna.distributions.FloatDistribution(1e-6, 1e-4, log=True),
    "te_lr_ratio": optuna.distributions.FloatDistribution(0.5, 10.0, log=True),
    "lr_scheduler": optuna.distributions.CategoricalDistribution(
        ["cosine", "constant_with_warmup"]
    ),
    "max_train_steps": optuna.distributions.CategoricalDistribution([800, 1200, 1600, 2000]),
    "use_noise_offset": optuna.distributions.CategoricalDistribution([False, True]),
    "noise_offset_value": optuna.distributions.FloatDistribution(0.02, 0.15),
    "use_min_snr": optuna.distributions.CategoricalDistribution([True, False]),
    "min_snr_gamma_value": optuna.distributions.FloatDistribution(1.0, 20.0),
    "use_ip_noise": optuna.distributions.CategoricalDistribution([False, True]),
    "ip_noise_gamma_value": optuna.distributions.FloatDistribution(0.02, 0.2),
    "caption_dropout_rate": optuna.distributions.FloatDistribution(0.0, 0.3),
}


def build_config(trial: optuna.Trial, name: str, dataset_version: str) -> dict[str, Any]:
    """Sample one point in the search space and render it as a LoRAExperimentConfig dict."""
    network_dim = trial.suggest_categorical("network_dim", [16, 32, 64, 128, 256])
    alpha_ratio = trial.suggest_categorical("alpha_ratio", [0.25, 0.5, 1.0])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    te_lr_ratio = trial.suggest_float("te_lr_ratio", 0.5, 10.0, log=True)
    lr_scheduler = trial.suggest_categorical("lr_scheduler", ["cosine", "constant_with_warmup"])
    max_train_steps = trial.suggest_categorical("max_train_steps", [800, 1200, 1600, 2000])

    # Conditional dimensions: each trick is present/absent, and only then valued. Expressing it
    # this way (rather than "0.0 means off") keeps the off case a single point for the sampler
    # instead of an arbitrary boundary of a continuous range.
    noise_offset = (
        trial.suggest_float("noise_offset_value", 0.02, 0.15)
        if trial.suggest_categorical("use_noise_offset", [False, True])
        else None
    )
    min_snr_gamma = (
        trial.suggest_float("min_snr_gamma_value", 1.0, 20.0)
        if trial.suggest_categorical("use_min_snr", [True, False])
        else None
    )
    ip_noise_gamma = (
        trial.suggest_float("ip_noise_gamma_value", 0.02, 0.2)
        if trial.suggest_categorical("use_ip_noise", [False, True])
        else None
    )
    # Unconditional, unlike the tricks above: 0.0 genuinely is "no dropout" and sits on the same
    # continuum as 0.05, so there is no separate off-mode to model.
    caption_dropout_rate = trial.suggest_float("caption_dropout_rate", 0.0, 0.3)

    training: dict[str, Any] = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 15,
        "max_train_steps": max_train_steps,
        "learning_rate": learning_rate,
        "unet_lr": learning_rate,
        "text_encoder_lr": learning_rate * te_lr_ratio,
        "lr_scheduler": lr_scheduler,
        "optimizer_type": "AdamW8bit",
        "mixed_precision": "fp16",
        "enable_bucket": True,
        "gradient_checkpointing": False,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "caption_dropout_rate": caption_dropout_rate,
    }
    if noise_offset is not None:
        training["noise_offset"] = noise_offset
    if min_snr_gamma is not None:
        training["min_snr_gamma"] = min_snr_gamma
    if ip_noise_gamma is not None:
        training["ip_noise_gamma"] = ip_noise_gamma

    return {
        "experiment_name": name,
        "model": {
            "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "model_family": "sd15",
            "resolution": 512,
        },
        "network": {
            "network_module": "networks.lora",
            "network_dim": network_dim,
            "network_alpha": network_dim * alpha_ratio,
        },
        "training": training,
        "data": {"train_data_dir": "img", "dataset_version": dataset_version},
    }


def score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """
    Composite objective to MAXIMIZE, plus the breakdown for logging.

    Returns -inf when a required metric is missing, so a broken trial is never mistaken for a
    good one.
    """
    pc = metrics.get("phase_consistency")
    kid = metrics.get("kid_classifier")
    cov = metrics.get("coverage")
    mem = metrics.get("memorization_excess_p95")
    if pc is None or kid is None or cov is None:
        return float("-inf"), {}

    penalty = W_MEMORIZATION * max(0.0, (mem or 0.0) - MEMORIZATION_CEILING)
    total = W_PHASE_CONSISTENCY * pc - W_KID * kid + W_COVERAGE * cov - penalty
    return total, {
        "phase_consistency": pc,
        "kid_classifier": kid,
        "coverage": cov,
        "memorization_excess_p95": mem or 0.0,
        "memorization_penalty": penalty,
    }


def params_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Invert `build_config`: recover the Optuna params that produced a written config."""
    net, tr = cfg["network"], cfg["training"]
    params: dict[str, Any] = {
        "network_dim": net["network_dim"],
        "alpha_ratio": round(net["network_alpha"] / net["network_dim"], 4),
        "learning_rate": tr["learning_rate"],
        "te_lr_ratio": tr["text_encoder_lr"] / tr["learning_rate"],
        "lr_scheduler": tr["lr_scheduler"],
        "max_train_steps": tr["max_train_steps"],
        # Absent in configs written before caption dropout entered the space; 0.0 is what those
        # runs effectively used, since kohya's default is 0.0.
        "caption_dropout_rate": tr.get("caption_dropout_rate", 0.0),
    }
    for flag, key, value_name in (
        ("use_noise_offset", "noise_offset", "noise_offset_value"),
        ("use_min_snr", "min_snr_gamma", "min_snr_gamma_value"),
        ("use_ip_noise", "ip_noise_gamma", "ip_noise_gamma_value"),
    ):
        present = key in tr and tr[key] is not None
        params[flag] = present
        if present:
            params[value_name] = tr[key]
    return params


def reseed_study(source_dir: Path, target: optuna.Study) -> int:
    """
    Replay finished trials from an existing study directory into `target`, rescoring them.

    Used after the KID estimator was fixed on 2026-08-10: the training and the generated samples
    from those trials remain perfectly valid, only the score derived from them was wrong. This
    recovers roughly 28 GPU-hours that would otherwise be re-run to learn nothing new.
    """
    added = 0
    for cfg_path in sorted(source_dir.glob("trial_*/config.yaml")):
        metrics_path = cfg_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        value, breakdown = score(metrics)
        if value == float("-inf"):
            continue

        params = params_from_config(yaml.safe_load(cfg_path.read_text()))
        unknown = set(params) - set(DISTRIBUTIONS)
        if unknown:
            raise KeyError(f"{cfg_path}: params absent from DISTRIBUTIONS: {sorted(unknown)}")

        target.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions={k: DISTRIBUTIONS[k] for k in params},
                value=value,
                user_attrs={**breakdown, "reseeded_from": str(cfg_path.parent)},
            )
        )
        added += 1
    return added


def run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def make_objective(study_dir: Path, samples: int, dataset_version: str):
    def objective(trial: optuna.Trial) -> float:
        name = f"trial_{trial.number:03d}"
        run_dir = study_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = build_config(trial, name, dataset_version)
        cfg_path = run_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        print(f"\n{'=' * 70}\n{name}: {trial.params}\n{'=' * 70}")

        run([sys.executable, "scripts/train_lora.py", "--config", str(cfg_path)], _ROOT)
        run(
            [
                sys.executable,
                "scripts/evaluate_lora.py",
                "--config",
                str(cfg_path),
                "--samples",
                str(samples),
            ],
            _ROOT,
        )

        metrics = json.loads((run_dir / "metrics.json").read_text())
        total, breakdown = score(metrics)
        for key, value in breakdown.items():
            trial.set_user_attr(key, value)
        # Keep the secondary metrics too, so a Pareto analysis is possible after the fact
        # without re-reading every experiment directory.
        for key in ("kid_vqgan", "density", "vqgan_recon_ratio", "final_loss"):
            if metrics.get(key) is not None:
                trial.set_user_attr(key, metrics[key])
        trial.set_user_attr(
            "phase_consistency_per_phase", metrics.get("phase_consistency_per_phase")
        )

        print(f"{name}: score={total:.4f}  {breakdown}")
        return total

    return objective


def report(study: optuna.Study) -> None:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nStudy '{study.study_name}': {len(done)} complete, {len(study.trials)} total")
    if not done:
        return

    header = f"{'trial':<12}{'score':>9}{'phase_c':>9}{'kid':>8}{'cover':>8}  params"
    print(header)
    print("-" * 100)
    for t in sorted(done, key=lambda t: -t.value)[:15]:
        ua = t.user_attrs
        print(
            f"trial_{t.number:03d}{'':<3}{t.value:>9.4f}"
            f"{ua.get('phase_consistency', float('nan')):>9.3f}"
            f"{ua.get('kid_classifier', float('nan')):>8.3f}"
            f"{ua.get('coverage', float('nan')):>8.3f}  {t.params}"
        )

    print(f"\nBest: trial_{study.best_trial.number:03d}  score={study.best_value:.4f}")
    print(f"Params: {json.dumps(study.best_params, indent=2)}")

    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter importances:")
        for key, value in importances.items():
            print(f"  {key:<24} {value:.3f}")
    except Exception as exc:  # needs enough completed trials to be meaningful
        print(f"\n(importances unavailable: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search over LoRA hyperparameters.")
    parser.add_argument("--study-name", default="lora_phase")
    parser.add_argument(
        "--dataset-version",
        default="per_phase_nofill",
        help="LoRA dataset version every trial trains on. Was hardcoded to `per_phase`, which "
        "silently pinned the first 21-trial study to the dataset carrying the dark-red rotation "
        "wedges; the fix was worth +0.085 phase_consistency, larger than most gaps that study "
        "was ranking. It is recorded in the study's user_attrs so a study can never be "
        "compared against one built on different data.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Target TOTAL completed trials for the study, not an increment. Resuming with the "
        "same value finishes the study instead of starting another batch on top of it.",
    )
    parser.add_argument(
        "--reset-stale",
        action="store_true",
        help="Mark trials stuck in RUNNING (from a hard kill) as FAIL so they can be re-sampled.",
    )
    parser.add_argument("--timeout", type=int, default=None, help="Wall-clock cap in seconds.")
    parser.add_argument("--samples", type=int, default=100, help="Eval images per phase.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true", help="Summarise and exit.")
    parser.add_argument(
        "--reseed-from",
        type=Path,
        default=None,
        help="Replay finished trials from another study dir into this one, rescored with the "
        "current objective. Use after a metric fix to keep the training already paid for.",
    )
    args = parser.parse_args()

    study_dir = _STUDY_ROOT / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.db'}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        # multivariate=True models parameter interactions. Default TPE fits independent 1-D
        # densities and cannot represent "a high LR only works at low rank", which the first
        # study hinted at (best: rank 16 @ 3.9e-5; runner-up: rank 64 @ 4.4e-6).
        sampler=optuna.samplers.TPESampler(
            seed=args.seed, n_startup_trials=8, multivariate=True, group=True
        ),
        load_if_exists=True,
    )

    if args.report:
        print(f"dataset_version: {study.user_attrs.get('dataset_version', '<unrecorded>')}")
        report(study)
        return

    # Pin the dataset to the study. A study is a set of observations of one objective function,
    # and the training data is part of that function: the first 21-trial study silently mixed
    # nothing only because the version was hardcoded, but the moment `per_phase_nofill` existed,
    # resuming it would have compared runs on different data. Refuse rather than warn.
    recorded = study.user_attrs.get("dataset_version")
    if recorded is None:
        study.set_user_attr("dataset_version", args.dataset_version)
    elif recorded != args.dataset_version:
        sys.exit(
            f"Study '{args.study_name}' was built on dataset_version='{recorded}' but "
            f"--dataset-version='{args.dataset_version}' was requested. Those are observations "
            f"of different objective functions; mixing them corrupts the sampler. "
            f"Start a new study instead."
        )

    if args.reseed_from is not None:
        if study.trials:
            sys.exit(
                f"Study '{args.study_name}' already has {len(study.trials)} trials; "
                "reseed into a fresh study name instead."
            )
        added = reseed_study(args.reseed_from, study)
        print(f"Reseeded {added} trial(s) from {args.reseed_from}.")
        report(study)
        return

    # Seed the search with what Phase 1 and 2 already established, so TPE does not spend its
    # startup trials rediscovering it. First: p3_per_phase's exact configuration. Second: the
    # same thing without noise_offset — the Phase 2 re-scoring separated all seven p2 runs
    # perfectly on that flag, and p3 inherited it from the wrong carry-forward base.
    if not study.trials:
        common = {
            "network_dim": 256,
            "alpha_ratio": 0.5,
            "learning_rate": 1e-5,
            "te_lr_ratio": 5.0,
            "lr_scheduler": "cosine",
            "max_train_steps": 2000,
            "use_min_snr": True,
            "min_snr_gamma_value": 5.0,
            "use_ip_noise": True,
            "ip_noise_gamma_value": 0.1,
        }
        study.enqueue_trial({**common, "use_noise_offset": True, "noise_offset_value": 0.05})
        study.enqueue_trial({**common, "use_noise_offset": False})
        # Same again minus ip_noise, which scored neutral-to-negative in the p2 re-ranking.
        study.enqueue_trial({**common, "use_noise_offset": False, "use_ip_noise": False})
        print("Enqueued 3 seed trials from the Phase 1/2 findings.")

    # A hard kill (SIGKILL, power loss) leaves trials stuck in RUNNING forever, which inflates
    # the completed count and hides the fact that their GPU time produced nothing.
    stale = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if stale:
        if args.reset_stale:
            for t in stale:
                study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
            print(f"Marked {len(stale)} stale RUNNING trial(s) as FAIL.")
        else:
            print(
                f"WARNING: {len(stale)} trial(s) stuck in RUNNING "
                f"({[t.number for t in stale]}). If no other process is working on them, "
                f"re-run with --reset-stale."
            )

    # --n-trials is a TARGET TOTAL, not an increment: resuming with the same command finishes
    # the study rather than starting another full batch on top of it.
    finished = [
        t
        for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    remaining = max(0, args.n_trials - len(finished))
    print(f"Study has {len(finished)} completed trial(s); running {remaining} more.")
    if remaining == 0:
        report(study)
        return

    study.optimize(
        make_objective(study_dir, args.samples, args.dataset_version),
        n_trials=remaining,
        timeout=args.timeout,
        catch=(RuntimeError,),  # a crashed trial marks itself failed; the study continues
    )
    report(study)


if __name__ == "__main__":
    main()
