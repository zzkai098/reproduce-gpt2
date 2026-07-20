"""Download a corpus, tokenize it with the GPT-2 BPE, and write token shards.

Turns raw text into the pre-tokenized `.npy` shards that `gpt2.data` streams
during training. Small runs can point at a single text file; a full reproduction
uses a large web corpus (e.g. FineWeb-Edu / OpenWebText).

Example
-------
    python scripts/prepare_data.py --dataset fineweb --out data/
"""
