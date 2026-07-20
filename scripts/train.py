"""Pretrain GPT-2 from scratch.

The main training loop: load a config, build the model and data loaders, and run
the GPT-2 recipe — AdamW with weight decay, cosine LR schedule with warmup,
gradient accumulation to reach the target token batch size, gradient clipping,
and mixed precision (bf16/fp16). Periodically evaluates validation loss and saves
checkpoints.

Runs on a single GPU, or on multiple GPUs with DistributedDataParallel via
`torchrun`.

Example
-------
    python scripts/train.py configs/gpt2_124m.py             # single GPU
    torchrun --standalone --nproc_per_node=8 scripts/train.py configs/gpt2_124m.py
"""
