"""
Relabel the noisy ``prophase`` crops of ``datasets/crops/vae_256`` with the OpenAI Decisions API.

Report-only: nothing in ``vae_256`` is modified. Each crop gets a stage
(interphase / early_prophase / mid_prophase / late_prophase / other), an ``other`` reason, and the
API's per-class probabilities. Results go to a new folder, one subfolder per predicted class::

    datasets/crops/prophase_llm/<run_id>/
        <stage>/*.png        hardlinks (copy fallback) of the source crops
        results.csv          one row per crop (label, probabilities, confidence, tokens, cost)
        review.csv           low-confidence / refused / other rows to eyeball
        raw.jsonl            raw answers, appended as the run goes (this is what --resume reads)
        summary.json         counts, tokens, cache hit rate, cost

Cost levers: POST /v1/decisions charges no output tokens; the rubric and the few-shot reference
images are the shared prefix of every request and are cached with an explicit breakpoint (the
query crop comes last, so it is never part of the cached prefix). A hard ``--budget`` aborts the run.

The key is read from ``OPENAI_API_KEY`` (a gitignored ``.env`` at the repo root is loaded).

Usage:
    uv run python scripts/utils/relabel_prophase_llm.py --dry-run
    uv run python scripts/utils/relabel_prophase_llm.py --pilot 40
    uv run python scripts/utils/relabel_prophase_llm.py --run-id 20261112-1000 --resume
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "datasets/crops/vae_256/prophase"
DEFAULT_MANIFEST = ROOT / "datasets/crops/vae_256/manifest.csv"
DEFAULT_OUT = ROOT / "datasets/crops/prophase_llm"
DEFAULT_PROMPT = Path(__file__).with_name("prophase_llm_prompt.yaml")

MODEL = "gpt-6-luna"
STAGES = ["interphase", "early_prophase", "mid_prophase", "late_prophase", "other"]
OTHER_REASONS = [
    "prometaphase_or_later",
    "blurry",
    "cropped",
    "debris",
    "unclear",
    "not_applicable",
]
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# USD per 1M tokens, gpt-6-luna Standard, prompts <= 272K tokens (pricing page, checked 2026-11-12).
# Output is "not charged" on /v1/decisions per the guide; kept as a conservative upper bound.
PRICE_INPUT = 0.10
PRICE_CACHED = 0.01
PRICE_CACHE_WRITE = 0.125  # 30m TTL, the only one Decisions supports
PRICE_OUTPUT = 0.0
# Pessimistic per-image token guess used only by --dry-run (no API call is made there).
DRYRUN_TOKENS_PER_IMAGE = 1500
DRYRUN_TEXT_TOKENS = 2500


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------------------


def confidence_label(p: float) -> str:
    """Words for a probability: high >= 0.8, medium >= 0.5, else low."""
    if p >= 0.8:
        return "high"
    if p >= 0.5:
        return "medium"
    return "low"


def request_cost(usage: dict) -> float:
    """USD cost of one request from a usage dict (input_tokens includes cached + cache-write)."""
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    written = details.get("cache_write_tokens", 0)
    fresh = max(usage.get("input_tokens", 0) - cached - written, 0)
    return (
        fresh * PRICE_INPUT
        + cached * PRICE_CACHED
        + written * PRICE_CACHE_WRITE
        + usage.get("output_tokens", 0) * PRICE_OUTPUT
    ) / 1e6


def image_part(path: Path, detail: str, breakpoint_: bool = False) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    part = {"type": "input_image", "image_url": f"data:{mime};base64,{data}", "detail": detail}
    if breakpoint_:
        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return part


def build_shared_input(prompt_cfg: dict, src: Path, detail: str) -> list[dict]:
    """Developer rubric + labelled reference images. The last reference image carries the cache
    breakpoint, so everything up to and including it is the cached prefix."""
    items: list[dict] = [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": prompt_cfg["developer_prompt"]}],
        }
    ]
    refs = prompt_cfg["references"]
    for i, ref in enumerate(refs):
        label = ref["label"] + (f" (reason: {ref['reason']})" if ref.get("reason") else "")
        items.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Reference example. Correct label: {label}"},
                    image_part(src / ref["file"], detail, breakpoint_=i == len(refs) - 1),
                ],
            }
        )
    return items


def build_query_message(prompt_cfg: dict, path: Path, detail: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt_cfg["query_text"]},
            image_part(path, detail),
        ],
    }


def build_questions(prompt_cfg: dict) -> list[dict]:
    q = prompt_cfg["questions"]
    return [
        {
            "type": "choice",
            "name": "stage",
            "instructions": q["stage"]["instructions"],
            "choices": [{"value": s} for s in STAGES],
        },
        {
            "type": "choice",
            "name": "other_reason",
            "instructions": q["other_reason"]["instructions"],
            "choices": [{"value": r} for r in OTHER_REASONS],
        },
    ]


def parse_answers(answers: list[dict]) -> dict:
    """Flatten the Decisions answers (as dicts) into one result row."""
    by_name = {a.get("name"): a for a in answers}
    stage = by_name.get("stage") or {}
    reason = by_name.get("other_reason") or {}
    if stage.get("type") != "choice":
        return {
            "label": "other",
            "other_reason": "refused",
            "p_label": 0.0,
            "api_confidence": 0.0,
            "conf_label": "low",
            "refused": True,
            "probs": {},
        }
    probs = {p["value"]: p["probability"] for p in stage["probabilities"]}
    label = stage["choice"]
    p_label = probs.get(label, 0.0)
    other_reason = ""
    if label == "other":
        other_reason = reason.get("choice", "unclear") if reason.get("type") == "choice" else ""
        if other_reason == "not_applicable":
            other_reason = "unclear"
    return {
        "label": label,
        "other_reason": other_reason,
        "p_label": p_label,
        "api_confidence": stage["confidence"],
        "conf_label": confidence_label(p_label),
        "refused": False,
        "probs": probs,
    }


def load_manifest_index(manifest: Path) -> dict[str, dict]:
    """basename of `output` -> manifest row. The prophase folder is flat, so split/legacy come
    from here."""
    index: dict[str, dict] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dropped") or not row.get("output"):
                continue
            index[Path(row["output"]).name] = row
    return index


def pilot_sample(files: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample over (split, legacy), proportional, at least one per non-empty stratum."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = {}
    for item in files:
        strata.setdefault((item["split"], item["legacy"]), []).append(item)
    total = len(files)
    chosen: list[dict] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda x: x["file"])
        k = min(len(group), max(1, round(n * len(group) / total)))
        chosen += rng.sample(group, k)
    rng.shuffle(chosen)
    return chosen[:n]


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------


def collect_files(
    src: Path, index: dict[str, dict], refs: set[str]
) -> tuple[list[dict], list[str]]:
    files, unmatched = [], []
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in IMG_EXTS or p.name.startswith("."):
            continue
        if p.name in refs:
            continue
        row = index.get(p.name)
        if row is None:
            unmatched.append(p.name)
        files.append(
            {
                "file": p.name,
                "path": p,
                "split": row["split"] if row else "unknown",
                "legacy": row["legacy"] if row else "",
            }
        )
    return files, unmatched


def read_done(raw_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["file"]] = rec
    return done


def classify_one(client, item: dict, shared: list[dict], prompt_cfg: dict, questions, args) -> dict:
    decision = client.decisions.create(
        model=args.model,
        input=[*shared, build_query_message(prompt_cfg, item["path"], args.detail)],
        questions=questions,
        prompt_cache_key=args.cache_key,
        prompt_cache_options={"mode": "explicit"},
    )
    answers = [a.model_dump() for a in decision.answers]
    usage = decision.usage.model_dump()
    parsed = parse_answers(answers)
    return {
        "file": item["file"],
        "split": item["split"],
        "legacy": item["legacy"],
        **parsed,
        "model": decision.model,
        "usage": usage,
        "cost_usd": request_cost(usage),
        "answers": answers,
    }


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def finalize(run_dir: Path, src: Path, raw_path: Path) -> dict:
    records = list(read_done(raw_path).values())
    for stage in STAGES:
        (run_dir / stage).mkdir(exist_ok=True)
    cols = [
        "file",
        "split",
        "legacy",
        "label",
        "other_reason",
        "p_label",
        "conf_label",
        "api_confidence",
        "refused",
        "probs",
        "input_tokens",
        "cached_tokens",
        "cost_usd",
    ]
    rows = []
    for r in sorted(records, key=lambda r: r["file"]):
        details = r["usage"].get("input_tokens_details") or {}
        rows.append(
            {
                **{k: r[k] for k in cols if k in r},
                "probs": json.dumps(r["probs"]),
                "input_tokens": r["usage"].get("input_tokens", 0),
                "cached_tokens": details.get("cached_tokens", 0),
            }
        )
        link_or_copy(src / r["file"], run_dir / r["label"] / r["file"])
    for name, subset in (
        ("results.csv", rows),
        ("review.csv", [x for x in rows if x["conf_label"] == "low" or x["refused"]]),
    ):
        with (run_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(subset)
    tokens_in = sum(x["input_tokens"] for x in rows)
    tokens_cached = sum(x["cached_tokens"] for x in rows)
    summary = {
        "n": len(rows),
        "label_counts": dict(Counter(x["label"] for x in rows)),
        "confidence_counts": dict(Counter(x["conf_label"] for x in rows)),
        "other_reason_counts": dict(Counter(x["other_reason"] for x in rows if x["other_reason"])),
        "input_tokens": tokens_in,
        "cached_tokens": tokens_cached,
        "cache_hit_rate": round(tokens_cached / tokens_in, 4) if tokens_in else 0.0,
        "cost_usd": round(sum(float(r["cost_usd"]) for r in records), 6),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--resume", action="store_true", help="continue an existing --run-id")
    ap.add_argument("--pilot", type=int, default=0, help="only N stratified crops")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--detail", default="auto", choices=["low", "high", "auto", "original"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--budget", type=float, default=5.0, help="hard USD cap; aborts when reached")
    ap.add_argument("--cache-key", default="prophase-llm-v1", help="prompt_cache_key (<= 64 chars)")
    ap.add_argument("--dry-run", action="store_true", help="count + estimate, no API call")
    args = ap.parse_args()

    prompt_cfg = yaml.safe_load(args.prompt.read_text())
    ref_names = {r["file"] for r in prompt_cfg["references"]}
    missing = sorted(n for n in ref_names if not (args.src / n).exists())
    if missing:
        print(f"ERROR: reference crops not found in {args.src}: {missing}", file=sys.stderr)
        return 2

    index = load_manifest_index(args.manifest)
    files, unmatched = collect_files(args.src, index, ref_names)
    if unmatched:
        print(
            f"WARNING: {len(unmatched)} crops not in the manifest (split='unknown'): {unmatched[:5]}"
        )
    if args.pilot:
        files = pilot_sample(files, args.pilot, args.seed)

    run_dir = args.out_root / args.run_id
    raw_path = run_dir / "raw.jsonl"
    done = read_done(raw_path)
    if run_dir.exists() and not args.resume and not args.dry_run:
        print(f"ERROR: {run_dir} exists; pass --resume or a new --run-id", file=sys.stderr)
        return 2
    todo = [f for f in files if f["file"] not in done]

    print(f"crops: {len(files)} (+{len(ref_names)} reference crops excluded) | done: {len(done)}")
    print("by split:", dict(Counter(f["split"] for f in files)))
    print(f"to classify: {len(todo)} | model: {args.model} | budget: ${args.budget:.2f}")

    if args.dry_run:
        n_img = len(ref_names) + 1
        per_req = n_img * DRYRUN_TOKENS_PER_IMAGE + DRYRUN_TEXT_TOKENS
        worst = len(todo) * per_req * PRICE_INPUT / 1e6
        print(
            f"dry-run upper bound (NO caching, ~{per_req} tokens/request): ${worst:.2f}. "
            "Real cost should be far lower once the prefix is cached; run --pilot to measure."
        )
        return 0

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (put it in .env, see .env.example)", file=sys.stderr)
        return 2

    client = OpenAI(max_retries=5)
    shared = build_shared_input(prompt_cfg, args.src, args.detail)
    questions = build_questions(prompt_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    spent = sum(r["cost_usd"] for r in done.values())
    lock = threading.Lock()
    stop = threading.Event()
    failures: list[str] = []

    def record(rec: dict) -> None:
        nonlocal spent
        with lock:
            with raw_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            spent += rec["cost_usd"]
            if spent >= args.budget:
                stop.set()

    def work(item: dict) -> None:
        if stop.is_set():
            return
        record(classify_one(client, item, shared, prompt_cfg, questions, args))

    if todo:
        # First call alone: it writes the cache, so the parallel calls after it can hit it.
        work(todo[0])
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(work, it): it for it in todo[1:]}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001 - keep going, report at the end
                    failures.append(f"{futures[fut]['file']}: {type(e).__name__}: {e}")
                if i % 25 == 0:
                    print(f"  {i}/{len(futures)} done, spent ${spent:.4f}")

    summary = finalize(run_dir, args.src, raw_path)
    print(json.dumps(summary, indent=2))
    print(f"outputs: {run_dir}")
    if failures:
        print(f"{len(failures)} failures (re-run with --resume):", *failures[:10], sep="\n  ")
    if stop.is_set():
        print(f"ABORTED: budget ${args.budget:.2f} reached; re-run with --resume to continue")
        return 3
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
