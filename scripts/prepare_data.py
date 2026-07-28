"""Download DCLM-baseline, tokenize it with the GPT-2 BPE, and write token shards.

Streams the DCLM-baseline dataset from Hugging Face (~4T tokens, far too large to
download in full), tokenizes documents in parallel, and writes fixed-size `.npy`
shards of uint16 token ids that `gpt2.data` streams during training. Stops once
~10B tokens have been written (one epoch for the 124M reproduction).

Shard 0 is the validation split; shards 1+ are training shards.

Needs the `datasets` and `tqdm` packages (pip install datasets tqdm).

Example
-------
    python scripts/prepare_data.py
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# ------------------------------------------------------------------------------
# config
local_dir = "dclm_10B"
shard_size = int(1e8)        # 100M tokens per shard
target_tokens = int(1e10)    # 10B tokens total, then stop

# output dir: <repo>/data/dclm_10B/  (created relative to this file, not the cwd)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# tokenizer  (module-level so spawned workers get it when they re-import this file)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end-of-text token: document delimiter


def tokenize(doc):
    """Tokenize one document -> uint16 numpy array, prefixed with <|endoftext|>."""
    tokens = [eot]                                  # start each doc with the delimiter
    tokens.extend(enc.encode_ordinary(doc["text"]))  # 'text' is DCLM's field; ignore special tokens
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token id out of uint16 range"
    return tokens_np.astype(np.uint16)              # uint16: vocab < 65536, halves disk


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# ------------------------------------------------------------------------------
def main():
    # stream the dataset (streaming=True -> never downloads the whole 4T; reads on the fly)
    ds = load_dataset("mlfoundations/dclm-baseline-1.0-parquet", split="train", streaming=True)

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # buffer for current shard
        token_count = 0            # how many tokens already in the buffer
        total_written = 0          # total tokens written to disk so far
        progress_bar = None

        # imap: main process streams docs from `ds`, workers tokenize them in parallel
        for tokens in pool.imap(tokenize, ds, chunksize=16):

            if token_count + len(tokens) < shard_size:
                # fits in the current shard -> just append
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # shard is full -> fill it up, write it, spill the remainder into a new shard
                split = "val" if shard_index == 0 else "train"   # shard 0 = validation
                filename = os.path.join(DATA_CACHE_DIR, f"dclm_{split}_{shard_index:06d}.npy")
                remainder = shard_size - token_count             # room left in this shard
                if progress_bar is not None:
                    progress_bar.update(remainder)
                all_tokens_np[token_count:] = tokens[:remainder] # top up the shard
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                total_written += shard_size
                progress_bar = None
                # leftover tokens start the next shard
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

            if total_written >= target_tokens:  # got enough -> stop streaming
                break

        # flush the last partial shard (if we stopped before it filled)
        if token_count != 0 and total_written < target_tokens:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"dclm_{split}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
