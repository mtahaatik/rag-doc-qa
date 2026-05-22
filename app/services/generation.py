"""
Generation: take the user question + retrieved chunks, build a prompt,
call Claude, return the answer.

LEARNING CHECKPOINT 4
=====================
You implement the prompt assembly and the API call.

Things that separate a junior RAG from a senior one:

1. Source attribution. Every chunk you put in the prompt should be tagged
   so the model can cite it ("according to source #2..."). Otherwise the
   user has no way to verify the answer.

2. Context budget. Claude can take 200K tokens but you shouldn't dump
   everything. You retrieve top-K, but K=20 with 400-token chunks = 8000
   tokens of context. Decide your budget and enforce it.

3. Instruction discipline. Tell the model what to do when the context does
   not contain the answer. Default behavior is to hallucinate. The fix is
   one line in the system prompt: "If the context does not contain the
   answer, say so. Do not make things up."

4. Streaming vs non-streaming. The Anthropic SDK supports both. For a
   chatbot UI, streaming feels much better. Skeleton uses non-streaming
   first to keep things simple; add streaming in query.py.

Write your prompt template choices in NOTES.md and explain WHY.
"""
from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.services.retrieval import RetrievedChunk

log = logging.getLogger(__name__)

_client = AsyncAnthropic(api_key=settings.anthropic_api_key)


SYSTEM_PROMPT = """\
You are a retrieval-augmented assistant. You answer the user's question
using ONLY the information in the provided context blocks. Each context
block is numbered. When you use information from a block, cite it inline
like [#1] or [#2].

If the context does not contain enough information to answer the question,
say so explicitly. Do not invent facts.
"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as a numbered context section."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[#{i}] (source: {c.source}, similarity: {c.similarity:.3f})\n{c.content}")
    return "\n\n".join(parts)


async def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Build the prompt and call Claude. Returns the answer text.

    TODO: refine the prompt, add max_tokens budget, handle the empty-chunks
    case (right now if chunks is [], we still call the API with empty
    context; you might want to short-circuit instead).
    """
    context = _build_context_block(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    log.info("calling claude, %d context chunks", len(chunks))

    response = await _client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    # response.content is a list of content blocks; for text-only responses
    # there is one TextBlock. In production you would handle tool_use blocks
    # too. Not needed here.
    return response.content[0].text
