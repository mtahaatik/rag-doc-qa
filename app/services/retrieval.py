"""
Retrieval: given an embedded query vector, find the top-k most similar
chunks from the database using pgvector.

LEARNING CHECKPOINT 3
=====================
You implement the SQL.

pgvector exposes 3 distance operators:
  <->   L2 (euclidean)  distance
  <#>   negative inner product
  <=>   cosine distance

For normalized vectors, all three rank the same. Without normalization,
they don't. You decided about normalization in checkpoint 2; the operator
you pick here has to match that decision.

Things to think about:
1. SELECT 1 - (embedding <=> query) AS similarity. Why 1 minus?
   (Hint: pgvector returns DISTANCE, not similarity.)
2. ORDER BY embedding <=> $1 LIMIT $2 lets the index be used. ORDER BY
   similarity DESC does NOT use the ivfflat index. Why?
3. If you want a similarity threshold ("only return chunks with sim > 0.7"),
   add a WHERE on the distance, not on the computed similarity. Same reason.

Write your reasoning in NOTES.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.core.db import get_pool

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    source: str
    chunk_index: int
    content: str
    similarity: float


async def retrieve_similar(
    query_embedding: np.ndarray,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Return the top-k chunks most similar to query_embedding.

    TODO: write the SQL. The skeleton below shows the shape of the result
    your caller expects. Replace the NotImplementedError with a real query.
    """
    pool = get_pool()

    raise NotImplementedError(
        "write the SQL using pgvector's <=> operator, return RetrievedChunk list"
    )

    # Example shape (delete this comment when implementing):
    # async with pool.connection() as conn:
    #     async with conn.cursor() as cur:
    #         await cur.execute(
    #             """
    #             SELECT c.id, c.document_id, d.source, c.chunk_index, c.content,
    #                    1 - (c.embedding <=> %s) AS similarity
    #             FROM chunks c JOIN documents d ON d.id = c.document_id
    #             ORDER BY c.embedding <=> %s
    #             LIMIT %s
    #             """,
    #             (query_embedding, query_embedding, top_k),
    #         )
    #         rows = await cur.fetchall()
    #         return [RetrievedChunk(*row) for row in rows]
