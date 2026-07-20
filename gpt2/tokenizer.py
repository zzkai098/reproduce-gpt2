"""Tokenizer — a thin wrapper over OpenAI's GPT-2 BPE via `tiktoken`.

GPT-2 uses byte-level BPE with a 50,257-token vocabulary. Rather than reimplement
it here, this wraps `tiktoken.get_encoding("gpt2")` behind a small encode/decode
interface so the rest of the codebase doesn't depend on tiktoken directly.

(Building BPE from scratch is a separate project — quillbpe.)
"""
