"""
Document chunking. Splits raw text into overlapping windows sized in tokens
(not characters). Token-aware sizing matters because the downstream LLM and
the embedding model both work in tokens.

LEARNING CHECKPOINT 1
=====================
You implement this. Things to think about:

1. Why tokens, not characters? Run a quick experiment: take a paragraph,
   count its chars, count its tokens with tiktoken. The ratio varies by
   language (English ~4 chars/token, Turkish closer to 2).

2. Why overlap? Without overlap, a sentence straddling a chunk boundary
   gets split across two embeddings, and neither chunk fully represents it.
   Overlap (usually 10-20% of chunk size) preserves context.

3. Why not sentence-aware splitting? You could split on sentence boundaries.
   That is better, but adds complexity. Start with fixed-size token windows,
   then upgrade to recursive character splitter or semantic chunking later.

When you're done, write 1 paragraph in NOTES.md: what did you pick for chunk
size and overlap, and why?
"""
from __future__ import annotations

import tiktoken

# cl100k_base is the tokenizer used by GPT-4 / text-embedding-3 and is a
# reasonable proxy for token counts even when generating with Claude.
# Claude has its own tokenizer but the counts are close enough for chunking.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Split `text` into overlapping windows of `chunk_size_tokens`, with
    `overlap_tokens` of overlap between consecutive chunks.

    Returns a list of chunk strings (decoded back from tokens).

    TODO: implement.

    Hints:
    - Encode the whole text to token ids with _TOKENIZER.encode(text)
    - Slide a window of size chunk_size_tokens, step = chunk_size - overlap
    - Decode each window back to a string with _TOKENIZER.decode(ids)
    - Edge case: text shorter than chunk_size should produce 1 chunk
    - Edge case: overlap >= chunk_size is invalid, raise ValueError
    """
    raise NotImplementedError("you implement chunking, see docstring")
