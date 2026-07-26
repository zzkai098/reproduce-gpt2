"""Shared training utilities.

Small helpers used across the training scripts, kept out of the model:

    get_device      pick cuda / cpu.
    set_seed        seed python / numpy / torch for reproducibility.
    get_lr          cosine learning-rate schedule with linear warmup (as in the
                    GPT-2 / GPT-3 training recipe).

Logging / checkpoint-io helpers live here too so the training loop stays readable.
"""

import math
import torch


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    
    # warmup
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    #cosin decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps) # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 1-> 0
    return min_lr + coeff * (max_lr - min_lr)
    
