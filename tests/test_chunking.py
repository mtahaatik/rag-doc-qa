"""
Basic tests. The chunking tests will fail until you implement chunk_text.
Use them as a spec: when they pass, your chunking is correct.
"""
import pytest

from app.services.chunking import chunk_text


class TestChunking:
    """These tests describe the contract for chunk_text. Implement until green."""

    def test_short_text_returns_one_chunk(self):
        result = chunk_text("hello world", chunk_size_tokens=100, overlap_tokens=10)
        assert len(result) == 1
        assert "hello world" in result[0]

    def test_long_text_returns_multiple_chunks(self):
        # 1000 words is much more than 50 tokens, so we should get multiple chunks
        text = " ".join(["word"] * 1000)
        result = chunk_text(text, chunk_size_tokens=50, overlap_tokens=10)
        assert len(result) > 1

    def test_overlap_greater_than_chunk_size_is_rejected(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size_tokens=10, overlap_tokens=20)

    def test_empty_text_returns_empty_or_single(self):
        result = chunk_text("", chunk_size_tokens=100, overlap_tokens=10)
        # Either behavior is defensible; pick one and document it.
        assert isinstance(result, list)


def test_health_endpoint_is_importable():
    # Sanity: the app module loads without raising
    from app.main import app
    assert app is not None
