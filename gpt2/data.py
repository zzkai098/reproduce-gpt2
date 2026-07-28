"""Sharded, DDP-aware data loading.

Streams pre-tokenized `.npy` token shards (from scripts/prepare_data.py) one
shard at a time, so the full corpus never has to fit in memory. next_batch
returns (x, y) of shape (B, T) with y = x shifted by one (next-token
prediction); DDP ranks read interleaved slices so each GPU sees different data.
"""

import os
import torch
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename) # uint16
    npt = npt.astype(np.int32) # int32（token id）
    return torch.tensor(npt, dtype=torch.long)
    
    
class DataLoaderLite:
    
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        data_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'dclm_10B')
        shards = sorted([s for s in os.listdir(data_root) if split in s]) # based on train or val get the shards
        self.shards = [os.path.join(data_root, s) for s in shards]        
        assert len(self.shards) > 0, f"no shards for split {split}"
        if self.process_rank == 0:
            print(f"found {len(self.shards)} shards for split {split}")
            
        self.reset()
            
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        
        # advance by the full stride so processes never read each other's chunks
        self.current_position += B*T * self.num_processes # update the pointer (+ chunk_size * stride)
        if self.current_position + (B*T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
            
        return x, y
