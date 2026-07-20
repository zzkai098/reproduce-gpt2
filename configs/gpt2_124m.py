"""Config for reproducing GPT-2 small (124M).

The architecture and training hyper-parameters that match OpenAI's GPT-2 124M:
12 layers, 12 heads, n_embd 768, block_size 1024, vocab 50257 — plus the training
recipe (token batch size ~0.5M via gradient accumulation, cosine LR schedule,
weight decay, warmup, total tokens ~10B for one epoch of the corpus).

Intended for a multi-GPU run; this is the "full reproduction" target.
"""
