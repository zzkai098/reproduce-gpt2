"""Generate text from a checkpoint or from the official GPT-2 weights.

Loads a model — either a local checkpoint or a pretrained GPT-2 pulled from
Hugging Face — and samples continuations with temperature and top-k control.

Example
-------
    python scripts/sample.py --init gpt2 --prompt "Hello, I'm a language model,"
    python scripts/sample.py --ckpt out/ckpt.pt --tokens 200
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken

from gpt2.model import GPT
from gpt2.utils import get_device


def load_model(args, device):
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = GPT(ckpt["config"])
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt['model'].items()}
        model.load_state_dict(sd)
    else:
        model = GPT.from_pretrained(args.init)
    
    return model.to(device).eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature, top_k):
    for _ in range(max_new_tokens):
        # idx is (B, T)
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond) # (B, T, C)
        logits = logits[:, -1, :] / temperature # (B, C)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k) # (B, topk)
            logits[logits < v[:, [-1]]] = -float('inf') # (B, C)
        
        probs = F.softmax(logits, -1) # (B, C)
        next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
        idx = torch.cat([idx, next_token], dim=1) # (B, T + 1)
        yield next_token
            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="local checkpoint, use --init if None")
    ap.add_argument("--init", default="gpt2", help="if ckpt is None: use gpt2 from hf")
    ap.add_argument("--prompt", default="Hello, I'm a language model,")
    ap.add_argument("--tokens", type=int, default=100, help="how much tokens want to be generated")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True, help="streaming output (use --no-stream to disable)")
    args = ap.parse_args()
    
    device = get_device()
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args, device)
    block_size = model.config.block_size
    
    tokens = enc.encode(args.prompt) # (T)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    
    #streaming
    if args.stream:
        print(args.prompt, end="", flush=True)
        ids = list(tokens)
        prev = enc.decode(ids)
        for next_token in generate(model, idx, args.tokens, block_size, args.temperature, args.top_k):
            ids.append(next_token[0, 0].item())
            text = enc.decode(ids)
            print(text[len(prev):], end="", flush=True)
            prev = text
        print()
    else:
        idx = idx.repeat(args.num_samples, 1)  # (B, T)
        out = list(generate(model, idx, args.tokens, block_size, args.temperature, args.top_k)) # (B, args.tokens)
        seqs = torch.cat([idx] + out, dim=-1) # (B, T + tokens)
        for i in range(args.num_samples):
            print((f"\n--- sample {i+1} ---"))
            print(enc.decode(seqs[i].tolist()))
        
        
if __name__ == "__main__":
    main()
    