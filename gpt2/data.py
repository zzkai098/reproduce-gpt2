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

import os
import torch
import tiktoken

    
class DataLoaderLite:
    
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')
        with open(data_path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        self.current_position = self.B * self.T * self.process_rank # each process starts at its own offset (staggered by rank)
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        
        # advance by the full stride so processes never read each other's chunks
        self.current_position += B*T * self.num_processes # update the pointer (+ chunk_size * stride)
        if self.current_position + (B*T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
            
        return x, y
