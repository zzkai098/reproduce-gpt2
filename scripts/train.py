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

import os
import time
import math
import torch

# DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# mygpt2 
from gpt2.model import GPT, GPTConfig
from gpt2.data import DataLoaderLite
from gpt2.utils import get_lr, get_device, set_seed


# ddp setup ------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "cuda not found"
    
    # join the distributed process group so all ranks can do collective ops
    # (all-reduce / broadcast); 'nccl' is NVIDIA's GPU-to-GPU backend (over NVLink)    
    init_process_group(backend='nccl')
    
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])    
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device) # pin this process to its assigned GPU
    master_process = (ddp_rank == 0)
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    master_process = True; device = get_device()
    
device_type = 'cuda' if device.startswith('cuda') else 'cpu'


# train setup ------------------------------------------------------------------------
set_seed(1337 + ddp_rank)
total_batch_size = 524288 # 2**19 = 0.5m
B, T = 16, 1024 
assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch should be divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # 524288 // (16 * 1024 * 8) = 4, each gpu now has 4 accum steps
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"grad_accum_steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)


# model ------------------------------------------------------------------------------
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 375e6 /524288 = 715.2557373047 gpt3 paper use 375m tokens for warmup 
max_steps = 19073 # 10B / 0.5M = 10e9 / 524288 ≈ 19073 use 10b tokens for 1 epoch

optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, 
    learning_rate=max_lr, 
    betas=(0.9, 0.95), 
    device_type=device_type,
    )


# training ------------------------------------------------------------------------------
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()    
    loss_accum = 0.0
    
    # inner loop: forward and backward, grad accumulation
    for micro_step in range(grad_accum_steps):
        xb, yb = train_loader.next_batch()
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # all_reduce when final micro_step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) 
        loss.backward()
        
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # grad clip
    
    # update parameters
    lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    optimizer.step()
    
    if device_type == 'cuda':
        torch.cuda.synchronize()
    
    dt = time.time() - t0
    tokens_per_sec = total_batch_size / dt
    if master_process:
        print(f"step {step:4d} | loss {loss_accum.item():.4f} | norm {norm:.4f} | lr {lr:.2e} | dt {dt*1000:.0f}ms | tok/s {tokens_per_sec:.0f}")
    
if ddp:
    destroy_process_group()
    



