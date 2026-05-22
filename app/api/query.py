"""
POST /query: embed the question, retrieve top-k similar chunks, generate
an answer with Claude.

LEARNING CHECKPOINT 5
=====================
Add Server-Sent Events streaming. Anthropic's SDK supports streaming:
async with client.messages.stream(...) as stream:
    async for text in stream.text_stream:
        yield text

Wire that to fastapi.responses.StreamingResponse with media_type="text/event-stream".

The hard part is shaping events so the client (curl or a small HTML page)
can consume them. Standard SSE format is:
    data: {"token": "..."}\n\n
"""
import logging

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.embedding import embed
from app.services.generation import generate_answer
from app.services.retrieval import retrieve_similar

router = APIRouter(tags=["query"])
log = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    top_k = req.top_k or settings.top_k_retrieval

    # Embed the question. embed() takes a list, returns a list.
    query_vector = embed([req.question])[0]

    # Retrieve relevant chunks from pgvector.
    chunks = await retrieve_similar(query_vector, top_k=top_k)

    # Generate the answer.
    answer_text = await generate_answer(req.question, chunks)

    return QueryResponse(
        answer=answer_text,
        sources=[
            SourceChunk(
                document_id=c.document_id,
                source=c.source,
                chunk_index=c.chunk_index,
                similarity=c.similarity,
                snippet=c.content[:200],
            )
            for c in chunks
        ],
    )
