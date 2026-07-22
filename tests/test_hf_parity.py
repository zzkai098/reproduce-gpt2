"""Hugging Face weight parity.

The whole point of a *faithful* reproduction: once our GPT loads the official
GPT-2 weights via `GPT.from_pretrained`, a forward pass must produce the same
logits as Hugging Face's `GPT2LMHeadModel` on the same input. If this passes,
the architecture (attention, MLP, embeddings, weight tying, the Conv1D
transpose in from_pretrained) is correct down to the numbers.

Run:  pytest tests/test_hf_parity.py -s
(needs `transformers`; installed via the [dev] extra)
"""
import pytest
import torch

from gpt2.model import GPT


def test_hf_logits_parity(monkeypatch, tmp_path):
    # The repo has a package dir named `gpt2/`. Hugging Face's from_pretrained
    # treats "gpt2" as a *local path* if such a dir exists in the cwd, and fails.
    # Run from an empty tmp dir so "gpt2" resolves to the Hub id instead.
    monkeypatch.chdir(tmp_path)

    transformers = pytest.importorskip("transformers")  # skip if dev deps missing
    from transformers import GPT2LMHeadModel

    # 1) our model, loaded with the official weights
    model = GPT.from_pretrained("gpt2")
    model.eval()

    # 2) the reference Hugging Face model
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    model_hf.eval()

    # 3) feed both the exact same token ids: (B, T)
    torch.manual_seed(0)
    idx = torch.randint(0, 50257, (2, 16))

    with torch.no_grad():
        logits_ours, _ = model(idx)          # our forward returns (logits, loss)
        logits_hf = model_hf(idx).logits     # (B, T, vocab_size)

    # shapes must match
    assert logits_ours.shape == logits_hf.shape == (2, 16, 50257)

    # semantic check: the predicted next token is identical at every position
    assert torch.equal(logits_ours.argmax(dim=-1), logits_hf.argmax(dim=-1)), \
        "argmax (next-token predictions) differ from Hugging Face"

    # numeric check: logits agree to a tight tolerance (fp32; tiny diffs come
    # only from op ordering, not from the model being different)
    max_diff = (logits_ours - logits_hf).abs().max().item()
    print(f"\nmax abs logit diff vs HF: {max_diff:.2e}")
    assert max_diff < 1e-3, f"logits diverge from HF: max abs diff = {max_diff}"
