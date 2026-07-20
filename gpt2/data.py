"""Data loading — tokenized text served as fixed-length training batches.

Two scales share one interface:

    a small in-memory dataset (a single tokenized text file) for local / Colab
    experiments, and a sharded loader that streams pre-tokenized `.npy` shards
    (e.g. FineWeb / OpenWebText) for a full reproduction run without holding the
    whole corpus in memory.

Each batch is (x, y) of shape (B, T), where y is x shifted by one — the standard
next-token-prediction setup. A DDP-aware loader strides shards across ranks so
each GPU sees a different slice.

`scripts/prepare_data.py` produces the token shards this module reads.
"""
