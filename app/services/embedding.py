"""
Embedding service. Loads the sentence-transformers model once and exposes
a function to embed a batch of texts.

The model is loaded lazily on first use. In production you would warm it up
during startup so the first user request is not slow.

LEARNING CHECKPOINT 2
=====================
The skeleton here loads the model and exposes embed(). What's left for you:

1. Batching. The function takes a list of strings. Sentence-transformers
   handles batching internally if you pass a list, but you should think
   about: what happens if I pass 10,000 strings? You may want to chunk
   the batch and emit progress logs.

2. Normalization. Cosine similarity wants L2-normalized vectors.
   sentence-transformers can do this for you via normalize_embeddings=True.
   Decide: normalize at embed-time, or at query-time in SQL? There is a
   correct answer here. Write your reasoning in NOTES.md.

3. Try changing the model to a larger one (e.g. all-mpnet-base-v2, 768-dim).
   What changes do you have to make? (Hint: DB schema, EMBEDDING_DIM env,
   probably re-ingest everything.) This is a real production headache and
   worth feeling once.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    log.info("loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[np.ndarray]:
    """
    Embed a batch of strings. Returns one numpy array per input.

    TODO: decide on normalization (see learning checkpoint 2). Currently
    returns raw (non-normalized) vectors.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        # normalize_embeddings=True,  # uncomment after you decide
    )
    return [v for v in vectors]
