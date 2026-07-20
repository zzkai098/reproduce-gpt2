"""Generate text from a checkpoint or from the official GPT-2 weights.

Loads a model — either a local checkpoint or a pretrained GPT-2 pulled from
Hugging Face — and samples continuations with temperature and top-k control.

Example
-------
    python scripts/sample.py --init gpt2 --prompt "Hello, I'm a language model,"
    python scripts/sample.py --ckpt out/ckpt.pt --tokens 200
"""
