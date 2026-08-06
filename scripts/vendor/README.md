# Vendored third-party code

Files here are copied verbatim (plus a provenance header) from upstream
projects. They are **excluded from ruff** (`pyproject.toml` → `extend-exclude`)
so the diff against upstream stays minimal and auditable.

## `sd-scripts/` (git submodule)

kohya_ss LoRA / DreamBooth trainer — headless `sd-scripts` library.

- **Source**: https://github.com/kohya-ss/sd-scripts
- **Branch**: `sd3` (superset of `main`; also carries `train_network.py` for SD1.x/2.x and `sdxl_train_network.py`)
- **Pinned commit**: see `.gitmodules` / `git submodule status`

The `sd3` branch covers SD1.5, SD2.x, SDXL, and SD3.x with a single checkout.
We use it headlessly via `accelerate launch scripts/vendor/sd-scripts/<entrypoint>.py <args>`;
the GUI (`bmaltais/kohya_ss`) is not involved.

### Re-syncing / bumping to a newer sd3-branch commit

```bash
cd scripts/vendor/sd-scripts
git fetch origin sd3
git checkout <new-commit-sha>
cd -
git add scripts/vendor/sd-scripts
git commit -m "chore: bump kohya sd-scripts to <new-commit-sha>"
```

**Do not** use `git submodule update --remote` — bumps must be explicit and reviewed.

### Entrypoints

| model family | script |
|---|---|
| SD1.5 / SD2.x | `train_network.py` |
| SDXL | `sdxl_train_network.py` |
| SD3 / SD3.5 | `sd3_train_network.py` |

### Dependency note

kohya's `requirements.txt` pins `diffusers==0.32.1`, which conflicts with the repo's
`diffusers>=0.35.1,<0.36`. Since kohya's `setup.py` declares no package deps, installing
the submodule as an editable package (`{path = "scripts/vendor/sd-scripts", editable = true}`)
does **not** re-pin diffusers. The repo's version is used at runtime; SD1.5 training is
verified to work with it.

---

## `train_controlnet.py`

ControlNet SD1.5 fine-tuning script.

- **Source**: https://github.com/Nictauro98/diffusers
  (`examples/controlnet/train_controlnet.py`)
- **Commit**: `c696ea555a717daaa672d25ce832e7b1874d3c26`
- **Upstream**: fork of `huggingface/diffusers`

### Differences from stock diffusers

1. Adds `import os` and `from PIL import Image`.
2. Pins `check_min_version("0.35.1")` (upstream had `"0.36.0.dev0"`).
3. In `make_train_dataset.preprocess_train`, conditioning images are loaded
   from disk by relative path rather than from an already-decoded image column:
   ```python
   conditioning_images = [
       Image.open(os.path.join(args.dataset_name, path)).convert("RGB")
       for path in examples[conditioning_image_column]
   ]
   ```

### Re-syncing from upstream

```bash
curl -sSL -o /tmp/train_controlnet.py \
  https://raw.githubusercontent.com/Nictauro98/diffusers/c696ea555a717daaa672d25ce832e7b1874d3c26/examples/controlnet/train_controlnet.py
# then re-apply the provenance header block (see the top of the vendored file)
```

This script is **not** edited in-repo. It is launched by
`scripts/train_controlnet.py` via `accelerate launch`.
