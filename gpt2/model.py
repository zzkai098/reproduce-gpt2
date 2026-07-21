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
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024 #T
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 #C


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # key, query, Value projections for all heads        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # projection
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)) # (T, T) ->  (1,1,T,T)
        
    def forward(self, x):
        B, T, C = x.size() # (B, T, n_embd)
        
        qkv = self.c_attn(x) # (B, T, 2304(3 * n_embd))
        q, k, v = qkv.split(self.n_embd, dim=2) #3 * (B, T, 768(n_embd))) split in dim2, which is C dim         
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs) (B, 12, T, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs) (B, 12, T, 64)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs) (B, 12, T, 64)

        # (B,nh,T,hs) @ (B,nh,hs,T) -> (B,nh,T,T), then sqrt(hs)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # (B,nh,T,T)
        y = att @ v # (B,nh,T,T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y) # output projection
        return y


class MLP(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x) # (B,T, n_embd) ->  (B,T, 4 * n_embd) 
        x = self.gelu(x) 
        x = self.c_proj(x) # (B,T, 4 * n_embd) ->  (B,T, n_embd) 
        return x 
        

class Block(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # (B,T, n_embd) -> (B,T, n_embd)
        x = x + self.mlp(self.ln_2(x))  # (B,T, n_embd) -> (B,T, n_embd)
        return x 

    
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd), # weight token embedding 
            wpe  = nn.Embedding(config.block_size, config.n_embd), # position token embedding 
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #
            ln_f = nn.LayerNorm(config.n_embd), #layernorm final
        ))        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx):
        #idx is B, T
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        # embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape(T)  [0,1,2,...,T-1]
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # transformer
        for block in self.transformer.h: # self attention and MLP
            x = block(x) # (B, T, n_embd)
        
        # final layer norm and the classifier
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        return logits
                
    @classmethod
    def from_pretrained(cls, model_type):
        """ load pretrained GPT2 model weights from huggingface """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 1) Based on model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257   # alwas 50257 for GPT-2 
        config_args['block_size'] = 1024    # alwas 1024 for GPT-2 

        # 2 create a initialized gpt model
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()        
        # attn.bias(down tril mask) is buffer, no need to learn
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 3 HF model，
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # These are need .t()
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                    'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape   
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    


# ------------------------------------------------------------------------


    
