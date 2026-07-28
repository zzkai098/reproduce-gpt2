"""Head-to-head evaluation: reproduced GPT-2 vs official GPT-2.

Runs a suite of zero-shot benchmarks on both models and prints a comparison table.

  perplexity (lower better) : WikiText-103
  multiple choice (higher)  : HellaSwag, PIQA, ARC-Easy, ARC-Challenge,
                              OpenBookQA, Winogrande

Add a benchmark by dropping one loader into the PERPLEXITY / MULTIPLE_CHOICE
registries below — the two evaluators in gpt2/eval.py are reused as-is.

Needs `datasets` (pip install datasets).

Usage
-----
    python scripts/eval.py --ckpt log/model_19072.pt   # your model vs official gpt2
    python scripts/eval.py                              # official gpt2 only (validate the harness)
    python scripts/eval.py --limit 500                 # cap examples per MC benchmark (speed)
"""
import argparse
import torch
import tiktoken
from datasets import load_dataset

from gpt2.model import GPT
from gpt2.utils import get_device
from gpt2.eval import evaluate_perplexity, evaluate_multiple_choice


# ---- models ----------------------------------------------------------------
def load_reproduced(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"])
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}  # strip compile prefix
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_official(device):
    return GPT.from_pretrained("gpt2").to(device).eval()


# ---- perplexity datasets ---------------------------------------------------
def _wikitext(enc, name):
    ds = load_dataset("wikitext", name, split="test")
    return torch.tensor(enc.encode_ordinary("\n\n".join(ds["text"])), dtype=torch.long)

PERPLEXITY = {
    # WikiText-2 shares the exact same test set as WikiText-103 (only the train
    # split differs), so it would report an identical number — not listed here.
    "WikiText-103": lambda enc: _wikitext(enc, "wikitext-103-raw-v1"),
}


# ---- multiple-choice datasets ----------------------------------------------
def _mc(ctx_ids, ending_ids_list, label):
    return {"context": ctx_ids, "endings": ending_ids_list, "label": label}

def _sub(ds, limit):
    return ds.select(range(min(limit, len(ds)))) if limit else ds

def hellaswag_examples(enc, limit):
    ds = _sub(load_dataset("hellaswag", split="validation"), limit)
    return [_mc(enc.encode_ordinary(e["ctx"]),
               [enc.encode_ordinary(" " + x) for x in e["endings"]],
               int(e["label"])) for e in ds]

def piqa_examples(enc, limit):
    ds = _sub(load_dataset("piqa", split="validation", trust_remote_code=True), limit)
    return [_mc(enc.encode_ordinary(e["goal"]),
               [enc.encode_ordinary(" " + e["sol1"]), enc.encode_ordinary(" " + e["sol2"])],
               int(e["label"])) for e in ds]

def _arc(enc, limit, config):
    ds = _sub(load_dataset("ai2_arc", config, split="test"), limit)
    out = []
    for e in ds:
        labels, texts = e["choices"]["label"], e["choices"]["text"]
        if e["answerKey"] not in labels:            # a few rows have odd keys
            continue
        out.append(_mc(enc.encode_ordinary(e["question"]),
                       [enc.encode_ordinary(" " + t) for t in texts],
                       labels.index(e["answerKey"])))
    return out

def openbookqa_examples(enc, limit):
    ds = _sub(load_dataset("openbookqa", "main", split="test"), limit)
    return [_mc(enc.encode_ordinary(e["question_stem"]),
               [enc.encode_ordinary(" " + t) for t in e["choices"]["text"]],
               e["choices"]["label"].index(e["answerKey"])) for e in ds]

def winogrande_examples(enc, limit):
    ds = _sub(load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True), limit)
    out = []
    for e in ds:
        prefix, suffix = e["sentence"].split("_", 1)      # blank marks the two-option split
        out.append(_mc(enc.encode_ordinary(prefix),
                       [enc.encode_ordinary(e["option1"] + suffix),
                        enc.encode_ordinary(e["option2"] + suffix)],
                       int(e["answer"]) - 1))
    return out

MULTIPLE_CHOICE = {
    "HellaSwag":     hellaswag_examples,
    "PIQA":          piqa_examples,
    "ARC-Easy":      lambda enc, limit: _arc(enc, limit, "ARC-Easy"),
    "ARC-Challenge": lambda enc, limit: _arc(enc, limit, "ARC-Challenge"),
    "OpenBookQA":    openbookqa_examples,
    "Winogrande":    winogrande_examples,
}


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="reproduced-model checkpoint; omit to eval official gpt2 only")
    ap.add_argument("--limit", type=int, default=1000, help="cap examples per MC benchmark; 0 = full")
    args = ap.parse_args()

    device = get_device()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    block_size = 1024
    limit = args.limit or None

    models = {"official gpt2": load_official(device)}
    if args.ckpt:
        models["reproduced"] = load_reproduced(args.ckpt, device)

    # perplexity
    for name, loader in PERPLEXITY.items():
        toks = loader(enc)
        print(f"\n=== {name} perplexity ({len(toks):,} tokens, lower is better) ===")
        for mname, model in models.items():
            ppl = evaluate_perplexity(model, toks, block_size, device, device_type)
            print(f"  {mname:16s} : {ppl:.2f}")

    # multiple choice
    for name, loader in MULTIPLE_CHOICE.items():
        ex = loader(enc, limit)
        print(f"\n=== {name} ({len(ex)} examples, higher is better) ===")
        for mname, model in models.items():
            r = evaluate_multiple_choice(model, ex, block_size, device, device_type)
            print(f"  {mname:16s} : acc {r['acc']*100:.1f}%   acc_norm {r['acc_norm']*100:.1f}%")


if __name__ == "__main__":
    main()
