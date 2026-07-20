"""A tiny config for fast local / Colab sanity checks.

A scaled-down model (few layers, small n_embd, short block_size) and a handful of
steps — enough to confirm the whole pipeline runs end to end (data → forward →
backward → checkpoint → sample) in a minute or two on CPU or a free GPU, before
committing to an expensive full run.
"""
