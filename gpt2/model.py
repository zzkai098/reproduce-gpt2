"""The GPT-2 model — a faithful decoder-only Transformer.

Mirrors OpenAI's GPT-2 architecture closely enough that the official Hugging Face
weights load into it unchanged (see tests/test_hf_parity.py). Built up from the
smallest piece:

    GPTConfig            hyper-parameters (n_layer / n_head / n_embd / block_size /
                         vocab_size); the 124M preset matches GPT-2 small.
    CausalSelfAttention  multi-head causal attention, QKV in one fused Linear.
    MLP                  the feed-forward block, 4x hidden, with GELU (not ReLU).
    Block                pre-LayerNorm + residual around attention and MLP.
    GPT                  wte + wpe embeddings -> N x Block -> final LayerNorm ->
                         lm_head, with the lm_head weight tied to the token
                         embedding (GPT-2's weight tying).

GPT-2-specific details that differ from a toy char-GPT: learned positional
embeddings (`wpe`), GELU activations, weight tying, and the scaled residual
initialization from the GPT-2 paper. A `from_pretrained` classmethod loads the
official weights from Hugging Face.

Shape convention: B = batch, T = sequence length, C = n_embd.
"""
