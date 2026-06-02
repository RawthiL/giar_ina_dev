# Vendored third-party code

Files here are copied verbatim (plus a provenance header) from upstream
projects. They are **excluded from ruff** (`pyproject.toml` → `extend-exclude`)
so the diff against upstream stays minimal and auditable.

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
