"""Shared training utilities.

Small helpers used across the training scripts, kept out of the model:

    get_device      pick cuda / mps / cpu.
    set_seed        seed python / numpy / torch for reproducibility.
    get_lr          cosine learning-rate schedule with linear warmup (as in the
                    GPT-2 / GPT-3 training recipe).
    configure_optimizers
                    build AdamW with decoupled weight decay applied only to 2D
                    weights (not to biases / LayerNorm), the GPT-2 convention.

Logging / checkpoint-io helpers live here too so the training loop stays readable.
"""
