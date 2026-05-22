"""
POST /ingest: take raw text, chunk it, embed each chunk, store in pgvector.
"""
import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.db import get_pool
from app.models.schemas import IngestRequest, IngestResponse
from app.services.chunking import chunk_text
from app.services.embedding import embed

router = APIRouter(tags=["ingest"])
log = logging.getLogger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    chunks = chunk_text(
        req.content,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="document produced no chunks")

    log.info("ingesting source=%s chunks=%d", req.source, len(chunks))

    embeddings = embed(chunks)

    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO documents (source) VALUES (%s) RETURNING id",
                (req.source,),
            )
            row = await cur.fetchone()
            document_id = row[0]

            # Bulk insert chunks. For larger batches, use copy or executemany.
            for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
                await cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_id, idx, content, emb),
                )
        await conn.commit()

    return IngestResponse(document_id=document_id, chunks_created=len(chunks))
