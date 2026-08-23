# LoRA Testing Suite Implementation Plan

## Overview

Expand the existing LoRA fine-tuning workflow into a **testing suite**: support multiple
*named dataset versions* under one folder, multiple experiments, a single-command **sweep**
across all of them, an **extensible evaluation** step (loss now, pluggable metrics later),
**validation images generated during training and surfaced in TensorBoard**, and a
**TensorBoard HParams** dashboard for cross-experiment comparison.

## Current State Analysis

- **Single shared dataset, no version axis.** `LoRADataConfig.dataset_dir` defaults to
  `datasets/crops/lora` ([lora_config.py:69](src/allium_cepa_classifier/config/lora_config.py#L69)),
  and experiments only override `train_data_dir: img`. `lora_dataset.py` writes one fixed layout
  `datasets/crops/lora/img/10_allium mitosis/` with no version concept
  ([lora_dataset.py:72](scripts/utils/lora_dataset.py#L72)).
- **Experiments overwrite each other.** rank16/32/64/128 config dirs exist, but every produced
  weight file is `sd15_rank16.safetensors` — `experiment_name` (and thus `--output_name`,
  [train_lora.py:53](scripts/train_lora.py#L53)) was copy-pasted across configs, and the rank/alpha
  values were never differentiated. The DVC `train_lora` matrix only lists `[sd15_rank16]`
  ([dvc.yaml:171](dvc.yaml#L171)).
- **No LoRA sweep.** [sweep.py](scripts/sweep.py) is hardcoded to `train_classifier.py`
  ([sweep.py:28](scripts/sweep.py#L28)).
- **No quantitative metrics, no `metrics.json`.** LoRA's only artifact is the visual
  `lora_samples.png` grid ([generate_lora_samples.py:98](scripts/generate_lora_samples.py#L98)).
  Every other stage emits a `metrics.json` (`cache: false`) — LoRA has none.
- **No samples during training.** The wrapper never passes kohya's `--sample_*` flags, and
  `LoRATrainingConfig` has no sample fields ([lora_config.py:39-65](src/allium_cepa_classifier/config/lora_config.py#L39-L65)).
- **kohya logs samples to wandb only, not TensorBoard.** `sample_images_common` writes PNGs to
  `<output_dir>/sample/` (`weights/sample/`) and logs only to wandb
  (`scripts/vendor/sd-scripts/library/sampling.py:320,491`). Filenames encode the step/epoch:
  `{output_name}_{num}_{i:02d}_{ts}{_seed}.png` (`sampling.py:488`). So a **bridge** is required
  to put them in the TensorBoard IMAGES tab.

## Desired End State

```bash
# Build two dataset versions
dvc repro prepare_lora_dataset@baseline prepare_lora_dataset@heavy_aug
#  → datasets/crops/lora/baseline/img/...   datasets/crops/lora/heavy_aug/img/...

# Sweep every experiment in one command
uv run python scripts/sweep_lora.py --configs experiments/lora/*/config.yaml
#  → trains each, generates samples, evaluates → metrics.json per experiment
#  → writes a shared TensorBoard HParams run + per-experiment validation-image timelines

# Compare
tensorboard --logdir experiments/lora/_sweeps/<timestamp>   # HParams tab + IMAGES tab
```

Verification: each experiment dir has distinct `weights/<experiment_name>.safetensors`, a
`metrics.json`, a TensorBoard logdir containing scalar loss + sampled IMAGES, and the shared sweep
logdir shows one HParams row per experiment.

### Key Discoveries
- kohya sample save dir: `args.output_dir + "/sample"` (`sampling.py:320`) → `weights/sample/`.
- kohya sample filename carries the step/epoch number (`sampling.py:488`) — parseable for TB steps.
- kohya only logs samples to wandb (`sampling.py:491-498`) — TB bridge needed.
- `BaseConfig.from_yaml` is plain Pydantic v2; adding fields with defaults is backward compatible
  ([base_config.py:26](src/allium_cepa_classifier/config/base_config.py#L26)).
- `dataset_dir` is resolved in `train_lora.build_cmd` as `dataset_dir / train_data_dir`
  ([train_lora.py:44](scripts/train_lora.py#L44)) — version slots in cleanly as a path segment.

## What We're NOT Doing

- No FID/KID/CLIP/classifier-as-judge metrics now — only **loss**. The evaluator is built with a
  pluggable registry so these can be added later without touching the sweep driver.
- Not changing inference (`AlliumCepaModel`) — LoRA remains a standalone generator.
- Not modifying the vendored `sd-scripts` source (keep it ruff-excluded and re-syncable).
- Not switching the sweep to `dvc exp`; DVC stays a thin reproducibility wrapper, the sweep script
  is the primary driver (per decision).
- Not auto-migrating the existing miswritten rank32/64/128 configs beyond fixing their
  `experiment_name`/rank fields as part of Phase 4.

## Implementation Approach

Five additive phases: (1) dataset versioning, (2) config extension for samples + dataset version,
(3) wrapper emits sample flags and bridges samples → TensorBoard, (4) extensible evaluator writing
`metrics.json`, (5) sweep driver that ties train→generate→evaluate together and logs HParams.

---

## Phase 1: Named dataset versions

### Overview
Let `lora_dataset.py` build into a named version subfolder and parameterize the DVC stage.

### Changes Required

#### 1. `scripts/utils/lora_dataset.py`
**Changes**: Add `--version` (default keeps current behavior). Output to
`<out>/<version>/img/<repeats>_allium mitosis/`.

```python
parser.add_argument("--version", default="baseline",
                    help="Named dataset version subfolder under --out.")
...
concept_dir = args.out / args.version / "img" / f"{args.repeats}_allium mitosis"
```

#### 2. `dvc.yaml` — `prepare_lora_dataset` becomes a matrix
**Changes**: Convert to `foreach` over version names; output per-version dir.

```yaml
  prepare_lora_dataset:
    foreach: [baseline, heavy_aug]
    do:
      cmd: .venv/bin/python scripts/utils/lora_dataset.py
        --vae-dir datasets/crops/vae --out datasets/crops/lora --version ${item}
      deps:
      - scripts/utils/lora_dataset.py
      - datasets/crops/vae
      outs:
      - datasets/crops/lora/${item}
```

(If `heavy_aug` needs different augmentation, that is a follow-up flag on `lora_dataset.py`; the
version axis itself is the deliverable here.)

### Success Criteria

#### Automated Verification:
- [x] `uv run python scripts/utils/lora_dataset.py --version baseline` creates
      `datasets/crops/lora/baseline/img/10_allium mitosis/` with `.png`/`.txt` pairs.
- [x] `dvc repro prepare_lora_dataset@baseline` succeeds.
- [x] Ruff clean: `uv run ruff check scripts/utils/lora_dataset.py`

#### Manual Verification:
- [ ] Captions and phase coverage match the previous single-folder output.

---

## Phase 2: Config extension (dataset version + sampling)

### Overview
Add a `dataset_version` field and kohya sample-generation fields to the config.

### Changes Required

#### 1. `src/allium_cepa_classifier/config/lora_config.py`
**Changes**: `LoRADataConfig` gains `dataset_version`; the resolved train dir becomes
`dataset_dir / dataset_version / train_data_dir`. Add a `LoRASamplingConfig`.

```python
class LoRADataConfig(BaseModel):
    dataset_dir: Path = _ROOT / "datasets/crops/lora"
    dataset_version: str = "baseline"
    train_data_dir: str = "img"
    caption_extension: str = ".txt"
    dataset_config: Path | None = None

class LoRASamplingConfig(BaseModel):
    enabled: bool = True
    every_n_steps: int | None = None
    every_n_epochs: int | None = 1
    sampler: str = "euler_a"
    at_first: bool = True
    # one prompt per mitotic phase by default; rendered into a kohya sample_prompts file
    prompts: list[str] = [
        "micrograph of allium cepa root tip mitotic cell in prophase phase --w 512 --h 512 --s 25",
        "micrograph of allium cepa root tip mitotic cell in metaphase phase --w 512 --h 512 --s 25",
        "micrograph of allium cepa root tip mitotic cell in anaphase phase --w 512 --h 512 --s 25",
        "micrograph of allium cepa root tip mitotic cell in telophase phase --w 512 --h 512 --s 25",
    ]

class LoRAExperimentConfig(BaseConfig):
    experiment_name: str
    model: LoRAModelConfig = LoRAModelConfig()
    network: LoRANetworkConfig = LoRANetworkConfig()
    training: LoRATrainingConfig = LoRATrainingConfig()
    data: LoRADataConfig = LoRADataConfig()
    sampling: LoRASamplingConfig = LoRASamplingConfig()
```

#### 2. `scripts/train_lora.py` — resolve version in `build_cmd`
**Changes**: [train_lora.py:44](scripts/train_lora.py#L44).

```python
train_dir = (cfg.data.dataset_dir / cfg.data.dataset_version / cfg.data.train_data_dir).resolve()
```

Apply the same version segment in the dry-run existence check ([train_lora.py:190,198](scripts/train_lora.py#L190)).

### Success Criteria

#### Automated Verification:
- [x] `uv run python scripts/train_lora.py --config experiments/lora/sd15_rank16/config.yaml --dry-run`
      prints a command whose `--train_data_dir` includes `/baseline/img`.
- [x] Ruff clean on the config + wrapper.

#### Manual Verification:
- [ ] Existing configs still load (defaults preserve `dataset_version: baseline`).

---

## Phase 3: Samples during training → TensorBoard

### Overview
Emit kohya `--sample_*` flags, then bridge `weights/sample/*.png` into the experiment's TensorBoard
logdir (kohya itself only logs samples to wandb).

### Changes Required

#### 1. `scripts/train_lora.py` — append sample flags in `build_cmd`
**Changes**: When `cfg.sampling.enabled`, write a `sample_prompts.txt` into `run_dir` and pass it.

```python
if cfg.sampling.enabled:
    prompts_file = run_dir / "sample_prompts.txt"
    prompts_file.write_text("\n".join(cfg.sampling.prompts))
    args += [f"--sample_prompts={prompts_file}", f"--sample_sampler={cfg.sampling.sampler}"]
    if cfg.sampling.at_first:
        args.append("--sample_at_first")
    if cfg.sampling.every_n_steps is not None:
        args.append(f"--sample_every_n_steps={cfg.sampling.every_n_steps}")
    if cfg.sampling.every_n_epochs is not None:
        args.append(f"--sample_every_n_epochs={cfg.sampling.every_n_epochs}")
```

#### 2. New `scripts/utils/lora_tb_bridge.py`
**Changes**: Parse `weights/sample/*.png`, extract the step/epoch number from the filename
(`{output_name}_{num}_{i:02d}_{ts}{_seed}.png`, `sampling.py:488`), and write each to a
`SummaryWriter(log_dir=run_dir/"logs")` via `add_image(tag=f"sample/{i}", img, global_step=num)`.

```python
def bridge_samples(run_dir: Path) -> int:
    sample_dir = run_dir / "weights" / "sample"
    if not sample_dir.exists():
        return 0
    writer = SummaryWriter(log_dir=str(run_dir / "logs"))
    for png in sorted(sample_dir.glob("*.png")):
        step, idx = _parse(png.name)          # num → step, i → tag index
        writer.add_image(f"sample/{idx}", to_tensor(Image.open(png).convert("RGB")), step)
    writer.close()
    return count
```

#### 3. `scripts/train_lora.py` — call the bridge after a successful run
**Changes**: After `run(...)` returns 0 ([train_lora.py:208-211](scripts/train_lora.py#L208-L211)),
call `bridge_samples(run_dir)`.

### Success Criteria

#### Automated Verification:
- [ ] Smoke run with `training.max_train_steps: 5`, `sampling.every_n_steps: 2` produces PNGs in
      `weights/sample/` and a non-empty `logs/` event file.
- [x] `lora_tb_bridge` unit test: a temp `sample/` dir with named PNGs yields the expected
      `(step, idx)` parse.
- [x] Ruff clean.

#### Manual Verification:
- [ ] `tensorboard --logdir experiments/lora/<name>/logs` shows sampled phase images in the
      IMAGES tab on a step/epoch timeline.

---

## Phase 4: Extensible evaluator → `metrics.json`

### Overview
A pluggable metric registry computing **loss** now (from the kohya TensorBoard event files), writing
`experiments/lora/<name>/metrics.json`, leaving an obvious seam for FID/CLIP/classifier-judge later.
Also fix the miswritten rank configs.

### Changes Required

#### 1. New `scripts/evaluate_lora.py`
**Changes**: Registry mapping metric-name → callable `(cfg, run_dir) -> dict[str, float]`. Default
runs `["loss"]`. The `loss` metric reads scalar tags from `run_dir/logs` (TB event files via
`tensorboard.backend.event_processing.EventAccumulator`) and reports `final_loss` / `min_loss` /
`avg_loss`. Writes `metrics.json` (sorted, stable) — same `cache: false` convention as other stages.

```python
METRICS: dict[str, Callable] = {"loss": loss_from_tb}

def main():
    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    out = {}
    for name in args.metrics:           # default ["loss"]
        out.update(METRICS[name](cfg, run_dir))
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True))
```

#### 2. Fix experiment configs
**Changes**: In `experiments/lora/sd15_rank{32,64,128}/config.yaml`, set `experiment_name`
accordingly and the matching `network.network_dim`/`network_alpha`. Add `data.dataset_version`
where a non-default version is intended.

### Success Criteria

#### Automated Verification:
- [x] `uv run python scripts/evaluate_lora.py --config experiments/lora/sd15_rank16/config.yaml`
      writes a `metrics.json` containing a numeric `final_loss`.
- [x] Adding a dummy metric to the registry surfaces in output with no sweep-script change
      (registry test).
- [x] Each `experiments/lora/*/config.yaml` has a unique `experiment_name` (assert in test).
- [x] Ruff clean.

#### Manual Verification:
- [ ] Loss values are plausible vs the TensorBoard scalar curves.

---

## Phase 5: Sweep driver + HParams comparison

### Overview
`scripts/sweep_lora.py`: for each config, run train → generate samples → evaluate, then log one
HParams entry per experiment to a shared sweep TensorBoard logdir.

### Changes Required

#### 1. New `scripts/sweep_lora.py`
**Changes**: Mirror [sweep.py](scripts/sweep.py) but LoRA-aware and including evaluation + HParams.

```python
parser.add_argument("--configs", nargs="+", type=Path, required=True)
parser.add_argument("--skip-generate", action="store_true")
sweep_dir = _ROOT / "experiments/lora/_sweeps" / time.strftime("%Y%m%d-%H%M%S")

for cfg_path in args.configs:
    run([sys.executable, "scripts/train_lora.py", "--config", str(cfg_path)])
    if not args.skip_generate:
        run([sys.executable, "scripts/generate_lora_samples.py", "--config", str(cfg_path)])
    run([sys.executable, "scripts/evaluate_lora.py", "--config", str(cfg_path)])
    cfg = LoRAExperimentConfig.from_yaml(cfg_path)
    metrics = json.loads((cfg_path.parent / "metrics.json").read_text())
    hparams = {
        "experiment": cfg.experiment_name,
        "dataset_version": cfg.data.dataset_version,
        "network_dim": cfg.network.network_dim,
        "network_alpha": cfg.network.network_alpha,
        "learning_rate": cfg.training.learning_rate,
        "max_train_epochs": cfg.training.max_train_epochs,
    }
    with SummaryWriter(log_dir=str(sweep_dir / cfg.experiment_name)) as w:
        w.add_hparams(hparams, {f"hparam/{k}": v for k, v in metrics.items()})

# print best-to-worst summary by final_loss
```

#### 2. `dvc.yaml` — expand the `train_lora`/`generate_lora_samples` foreach lists
**Changes**: List all experiment names; add a `metrics:` block pointing at each
`experiments/lora/${item}/metrics.json` (`cache: false`), mirroring the classifier stage.

#### 3. CLAUDE.md
**Changes**: Document `sweep_lora.py`, `evaluate_lora.py`, dataset versions, and the HParams/IMAGES
TensorBoard comparison under the LoRA section.

### Success Criteria

#### Automated Verification:
- [ ] `uv run python scripts/sweep_lora.py --configs experiments/lora/*/config.yaml`
      (with tiny `max_train_steps`) completes, writing `metrics.json` for each and a populated
      `experiments/lora/_sweeps/<ts>/`.
- [ ] Sweep prints a summary ranked by `final_loss`.
- [x] Ruff clean: `uv run ruff check scripts/`.
- [x] `uv run pytest` passes (22 tests: 9 LoRA + 13 VAE).

#### Manual Verification:
- [ ] `tensorboard --logdir experiments/lora/_sweeps/<ts>` shows one HParams row per experiment and
      the per-experiment sample IMAGES.

---

## Testing Strategy

### Unit Tests:
- `lora_tb_bridge._parse` filename → `(step, idx)`.
- evaluator registry: dummy metric appears in output; `loss` reads a synthetic TB event file.
- config: every `experiments/lora/*/config.yaml` loads and has a unique `experiment_name`.

### Integration Tests:
- End-to-end smoke sweep with `max_train_steps: 5` over two tiny configs → two `metrics.json` + one
  sweep dir.

### Manual Testing Steps:
1. Build `baseline` + a second dataset version; confirm distinct folders.
2. Run a short training; confirm samples appear in TensorBoard IMAGES during/after the run.
3. Run the sweep; open the HParams tab and confirm rows compare correctly.

## Performance Considerations

Sampling every N steps adds inference overhead per checkpoint — keep `every_n_epochs` modest
(default 1) and allow disabling via `sampling.enabled: false` for long runs. The TB bridge runs once
post-training, negligible cost.

## Migration Notes

- Existing `datasets/crops/lora/img/...` is superseded by `datasets/crops/lora/baseline/img/...`;
  rebuild via `dvc repro prepare_lora_dataset@baseline` (default `dataset_version: baseline` keeps
  old configs working once data is rebuilt).
- The rank32/64/128 weights currently mislabeled `sd15_rank16.safetensors` will be regenerated with
  correct names after the Phase 4 config fix.

## References

- Wrapper: [scripts/train_lora.py](scripts/train_lora.py)
- Sample generation: [scripts/generate_lora_samples.py](scripts/generate_lora_samples.py)
- Dataset prep: [scripts/utils/lora_dataset.py](scripts/utils/lora_dataset.py)
- Config: [src/allium_cepa_classifier/config/lora_config.py](src/allium_cepa_classifier/config/lora_config.py)
- Existing sweep: [scripts/sweep.py](scripts/sweep.py)
- kohya sampling (wandb-only, save path, filename): `scripts/vendor/sd-scripts/library/sampling.py:320,488,491`
- DVC LoRA stages: [dvc.yaml:54,170,190](dvc.yaml#L54)
- Prior LoRA plan: [thoughts/shared/plans/2026-06-21-lora-finetuning-kohya.md](thoughts/shared/plans/2026-06-21-lora-finetuning-kohya.md)
