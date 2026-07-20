"""Fine-tune a pretrained GPT-2 on a smaller, custom dataset.

Starts from the official GPT-2 weights (`GPT.from_pretrained`) instead of random
init, then continues training on a small corpus with a low learning rate. This is
the cheap, GPU-light path — minutes on a free Colab GPU — and reuses most of the
training loop from `scripts/train.py`.

Example
-------
    python scripts/finetune.py --init gpt2 --data data/shakespeare.txt
"""
