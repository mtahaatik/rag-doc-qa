# Design Notes

## Checkpoint 1: Chunking

- Strategy: fixed-size sliding window over token IDs (tiktoken cl100k_base).
- Defaults: chunk_size=400 tokens (~300 words), overlap=50 tokens (~12%).
- Why overlap: sentences spanning a boundary stay recoverable from at least one chunk.
- Why token-aware over character-based: token count is what matters for both the embedding model's context limit and downstream LLM cost. Character counts mislead, especially for non-English text.
- Edge cases: text shorter than chunk_size returns a single chunk; empty text returns empty list; overlap >= chunk_size raises ValueError.
- Next iteration: try recursive character splitter or semantic chunking for better boundary respect.

## Checkpoint 2: Embedding

- Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim).
- Normalization: enabled at embed time (`normalize_embeddings=True`).
  - One source of truth, no risk of forgetting to normalize at query time.
  - Cosine similarity with normalized vectors reduces to dot product, faster at scale.
- Why local embeddings over an API: no per-token cost, predictable latency, no third-party dependency in the hot path. Trade-off is slightly lower retrieval quality vs OpenAI ada-002.
- Next iteration: benchmark all-mpnet-base-v2 (768-dim) and decide if quality lift justifies 3x latency.