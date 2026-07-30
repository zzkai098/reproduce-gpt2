"""Sweep eval benchmarks across training snapshots -> a score-vs-step curve.

Loads every `log/snap_*.pt` weights-only snapshot (written by scripts/train.py),
runs a small representative subset of the eval.py benchmarks on each, and writes
one row per snapshot to a CSV. Reuses eval.py's dataset loaders and the two
evaluators in gpt2/eval.py unchanged.

Datasets are loaded once and reused across every snapshot — the expensive part is
the per-snapshot forward passes, not reading the data. The CSV is flushed after
each snapshot, so an interrupted sweep keeps the rows it already computed.

Keep the tracked set SMALL (this runs once per snapshot); the full 8-benchmark
head-to-head lives in eval.py and is meant for the final table, not the curve.

Usage
-----
    python scripts/eval_sweep.py                          # all snapshots in log/
    python scripts/eval_sweep.py --limit 500              # cap MC examples (speed)
    python scripts/eval_sweep.py --snap-dir log --out log/sweep.csv
"""
import argparse
import csv
import glob
import os
import re

import torch
import tiktoken

from gpt2.utils import get_device
from gpt2.eval import evaluate_perplexity, evaluate_multiple_choice
from eval import load_reproduced, PERPLEXITY, MULTIPLE_CHOICE


# Subset of eval.py's registries to track over training (PIQA omitted — its Hub
# dataset is a loading script that recent `datasets` no longer supports).
PPL_BENCHMARKS = ["WikiText-103"]
MC_BENCHMARKS = ["HellaSwag", "ARC-Easy", "ARC-Challenge", "OpenBookQA", "Winogrande"]


def snapshot_step(path):
    """Pull the integer step out of a 'snap_01000.pt' filename (for sort / x-axis)."""
    m = re.search(r"snap_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", default="log", help="dir holding snap_*.pt snapshots")
    ap.add_argument("--out", default=None, help="output CSV (default <snap-dir>/sweep.csv)")
    ap.add_argument("--limit", type=int, default=500, help="cap examples per MC benchmark; 0 = full")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.snap_dir, "sweep.csv")
    limit = args.limit or None

    device = get_device()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    block_size = 1024

    # find + sort snapshots by step (so the CSV / curve is in training order)
    snaps = sorted(glob.glob(os.path.join(args.snap_dir, "snap_*.pt")), key=snapshot_step)
    assert snaps, f"no snap_*.pt found in {args.snap_dir}"
    print(f"found {len(snaps)} snapshots: steps {[snapshot_step(s) for s in snaps]}")

    # load each dataset ONCE, reuse across all snapshots; skip any that fail to load
    ppl_data, mc_data = {}, {}
    for name in PPL_BENCHMARKS:
        try:
            ppl_data[name] = PERPLEXITY[name](enc)
        except Exception as e:
            print(f"skip {name}: {type(e).__name__}: {e}")
    for name in MC_BENCHMARKS:
        try:
            mc_data[name] = MULTIPLE_CHOICE[name](enc, limit)
        except Exception as e:
            print(f"skip {name}: {type(e).__name__}: {e}")

    # columns: step, then one column per benchmark that actually loaded
    ppl_cols = [f"{name}_ppl" for name in ppl_data]
    mc_cols = [f"{name}_acc_norm" for name in mc_data]
    fieldnames = ["step"] + ppl_cols + mc_cols

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for path in snaps:
            step = snapshot_step(path)
            model = load_reproduced(path, device)   # reuses eval.py loader (strips _orig_mod.)
            row = {"step": step}

            for name, toks in ppl_data.items():
                ppl = evaluate_perplexity(model, toks, block_size, device, device_type)
                row[f"{name}_ppl"] = round(ppl, 3)
            for name, ex in mc_data.items():
                r = evaluate_multiple_choice(model, ex, block_size, device, device_type)
                row[f"{name}_acc_norm"] = round(r["acc_norm"], 4)

            writer.writerow(row)
            f.flush()                               # persist after each snapshot
            print(f"step {step:5d} | " + " | ".join(f"{k} {row[k]}" for k in fieldnames[1:]))

            del model                               # free the model before the next one
            if device_type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
