"""
Request and response models. Pydantic gives us validation + OpenAPI for free.
"""
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = Field(..., description="Label for this document (filename, URL, note id)")
    content: str = Field(..., min_length=1, description="Raw text content to ingest")


class IngestResponse(BaseModel):
    document_id: int
    chunks_created: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, description="Override default top-k for retrieval")


class SourceChunk(BaseModel):
    document_id: int
    source: str
    chunk_index: int
    similarity: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
