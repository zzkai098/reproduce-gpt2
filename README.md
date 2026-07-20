# reproduce-gpt2

A faithful, **from-scratch reproduction of OpenAI's GPT-2 (124M)** in PyTorch —
built to match the real architecture closely enough that the official Hugging Face
weights load in unchanged. A step up from a toy character-level GPT to the actual
model: byte-level BPE, learned positional embeddings, GELU, weight tying, the
GPT-2 initialization, mixed-precision training, gradient accumulation, and
multi-GPU (DDP) training.

Inspired by Andrej Karpathy's ["Let's reproduce GPT-2
(124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) and
[build-nanogpt](https://github.com/karpathy/build-nanogpt).

## What "reproduce GPT-2" means here

Four levels, from free to full — you can go as far as your compute budget allows:

| Level | What | Compute |
|---|---|---|
| 1. **Architecture + weight parity** | Build GPT-2 and load the official 124M weights; verify logits match Hugging Face exactly | Free (CPU) |
| 2. **Inference & fine-tuning** | Generate from pretrained GPT-2; fine-tune it on a small custom corpus | Free (Colab GPU) |
| 3. **Small pretraining run** | Train a scaled-down model to watch the full recipe work (loss drops, machinery runs) | Free (Colab GPU) |
| 4. **Full 124M reproduction** | Pretrain 124M on ~10B tokens (FineWeb) to GPT-2's eval loss | Multi-GPU cloud (~$20–60) |

Levels 1–3 need no paid hardware. Level 4 is the optional endgame on rented GPUs.

## How GPT-2 differs from a toy GPT

The architecture is the same decoder-only Transformer, but with the details that
make it the *real* GPT-2:

- **Byte-level BPE** (50,257-token vocab) via `tiktoken`, instead of characters.
- **Learned positional embeddings** (`wpe`), a trainable table like the token
  embeddings.
- **GELU** activation in the MLP (not ReLU).
- **Weight tying** — the output `lm_head` shares weights with the token embedding.
- **Scaled residual init** from the GPT-2 paper for stable deep training.
- **Training recipe**: AdamW with decoupled weight decay on 2D weights only,
  cosine LR schedule with warmup, gradient accumulation to a ~0.5M-token batch,
  gradient clipping, and bf16/fp16 mixed precision.

## Project layout

```
reproduce-gpt2/
├── gpt2/                     # the library (importable)
│   ├── model.py              # GPT / Block / CausalSelfAttention / MLP + GPTConfig, from_pretrained
│   ├── tokenizer.py          # GPT-2 BPE via tiktoken
│   ├── data.py               # in-memory + sharded, DDP-aware batch loaders
│   └── utils.py              # lr schedule, optimizer config, device, seeding, io
├── scripts/                  # entry points (run these)
│   ├── prepare_data.py       # download + tokenize a corpus into token shards
│   ├── train.py              # pretrain from scratch (single-GPU or torchrun DDP)
│   ├── finetune.py           # fine-tune pretrained GPT-2 on a small dataset
│   └── sample.py             # generate text from a checkpoint or from gpt2
├── configs/                  # hyper-parameters as Python files
│   ├── gpt2_124m.py          # the full 124M reproduction target
│   └── debug_small.py        # tiny config for fast local/Colab sanity checks
├── tests/                    # shapes, causal masking, and Hugging Face weight parity
├── notebooks/                # colab_train.ipynb
├── assets/                   # figures
├── pyproject.toml
└── README.md
```

## License

MIT © 2026 Zhankai Zhang
