"""Evaluation — zero-shot perplexity and multiple-choice accuracy.

Two reusable evaluators, meant to be run head-to-head against the official GPT-2:

    evaluate_perplexity       sliding-window perplexity over a token stream
                              (WikiText-103, WikiText-2, PTB, LAMBADA, 1BW, ...)
    evaluate_multiple_choice  pick the most likely completion; accuracy
                              (HellaSwag, Winogrande, PIQA, ARC, ...)

Both assume a model whose forward is `logits, loss = model(idx, targets=None)` and
returns `logits` of shape (B, T, vocab) when `targets` is None.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_perplexity(model, token_ids, block_size, device, device_type, stride=None):
    """Sliding-window perplexity over one long token stream.

    token_ids : 1D LongTensor — the whole corpus tokenized and concatenated.
    stride    : how far the window moves each step (defaults to block_size // 2).
                smaller stride -> more context per token -> more accurate, slower.

    Each window scores only its *new* tokens (the overlapping prefix is masked with
    -100 = ignore_index), so every token is scored once with full left-context.
    Returns perplexity = exp(total_nll / tokens_scored).
    """
    model.eval()
    if stride is None:
        stride = block_size // 2

    n = token_ids.size(0)
    nll_sum = 0.0     # summed cross-entropy over scored tokens
    n_tokens = 0      # number of scored tokens
    prev_end = 0      # corpus index scored up to so far

    for begin in range(0, n - 1, stride):
        end = min(begin + block_size, n - 1)   # window covers [begin, end)
        trg_len = end - prev_end               # how many *new* tokens to score

        x = token_ids[begin:end].unsqueeze(0).to(device)                  # (1, L)
        y = token_ids[begin + 1:end + 1].clone().unsqueeze(0).to(device)  # shifted targets
        y[:, :-trg_len] = -100                 # mask the overlapping prefix -> score only new tokens

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(x)               # (1, L, vocab)  (no targets -> not reshaped)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (L, vocab)
            y.view(-1),                        # (L,)
            ignore_index=-100,
            reduction="sum",                   # sum here; we divide by n_tokens ourselves
        )
        nll_sum += loss.item()
        n_tokens += trg_len
        prev_end = end
        if end == n - 1:
            break

    avg_nll = nll_sum / n_tokens
    return math.exp(avg_nll)


@torch.no_grad()
def _completion_nll(model, context_ids, completion_ids, block_size, device, device_type):
    """Summed cross-entropy of `completion_ids` given `context_ids`, and its length.

    Concatenates context + completion, scores only the completion tokens (context
    positions masked with -100), left-truncating the context if the pair exceeds
    block_size. Returns (nll_sum, n_completion_tokens).
    """
    full = list(context_ids) + list(completion_ids)
    n_comp = len(completion_ids)
    if len(full) > block_size:
        full = full[-block_size:]              # keep the tail (completion stays intact)

    full = torch.tensor(full, dtype=torch.long, device=device)
    x = full[:-1].unsqueeze(0)                 # (1, L-1)
    y = full[1:].clone()                       # (L-1,)  shifted targets
    y[:-n_comp] = -100                         # score only the completion tokens (the tail)
    y = y.unsqueeze(0)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(x)

    nll = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return nll.item(), n_comp


@torch.no_grad()
def evaluate_multiple_choice(model, examples, block_size, device, device_type):
    """Multiple-choice accuracy: pick the completion the model finds most likely.

    examples : list of dicts, each
        {"context": [int ids], "endings": [[ids], [ids], ...], "label": int}

    For every ending we compute the completion NLL, then predict the ending with
    the lowest NLL. Reports two standard metrics:
        acc      : lowest total NLL   (raw likelihood)
        acc_norm : lowest average NLL (length-normalized; the HellaSwag headline)
    """
    model.eval()
    correct = 0
    correct_norm = 0
    total = 0

    for ex in examples:
        ctx = ex["context"]
        nll_sums, nll_avgs = [], []
        for ending in ex["endings"]:
            s, m = _completion_nll(model, ctx, ending, block_size, device, device_type)
            nll_sums.append(s)
            nll_avgs.append(s / m)             # per-token average

        pred = min(range(len(nll_sums)), key=lambda i: nll_sums[i])       # lowest total NLL
        pred_norm = min(range(len(nll_avgs)), key=lambda i: nll_avgs[i])  # lowest avg NLL

        correct += int(pred == ex["label"])
        correct_norm += int(pred_norm == ex["label"])
        total += 1

    return {"acc": correct / total, "acc_norm": correct_norm / total, "n": total}
