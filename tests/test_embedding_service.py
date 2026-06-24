"""Test embedding service."""

from unittest.mock import MagicMock, patch

import pytest

from slopometry.solo.services.embedding_service import EmbeddingService


def test_compute_similarity_same_vector() -> None:
    """Similarity of vector to itself should be 1.0."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    vector = [0.1, 0.2, 0.3, 0.4]
    similarity = service.compute_similarity(vector, vector)

    assert similarity == pytest.approx(1.0)


def test_compute_similarity_different_vectors() -> None:
    """Different vectors should have similarity less than 1.0."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    vector1 = [0.1, 0.2, 0.3, 0.4]
    vector2 = [0.4, 0.3, 0.2, 0.1]
    similarity = service.compute_similarity(vector1, vector2)

    assert similarity < 1.0
    assert similarity > -1.0


def test_compute_similarity_zero_magnitude() -> None:
    """Should handle zero vectors gracefully."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    vector = [0.0, 0.0, 0.0, 0.0]
    other = [0.1, 0.2, 0.3, 0.4]

    result = service.compute_similarity(vector, other)
    assert result == 0.0

    result2 = service.compute_similarity(other, vector)
    assert result2 == 0.0


def test_compute_uniqueness_score_no_existing() -> None:
    """Should return 1.0 when no existing embeddings."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    vector = [0.1, 0.2, 0.3, 0.4]
    score = service.compute_uniqueness_score(vector, [])

    assert score == 1.0


def test_compute_uniqueness_score_with_existing() -> None:
    """Should return lower score when similar embeddings exist."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    new_vector = [1.0, 0.0, 0.0, 0.0]
    existing = [
        [0.5, 0.5, 0.5, 0.5],
        [0.6, 0.4, 0.4, 0.4],
    ]

    score = service.compute_uniqueness_score(new_vector, existing)

    assert score < 1.0
    assert score >= 0.0


def test_get_embedding_raises_on_failure() -> None:
    """Should raise RuntimeError on API failure."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="Failed to get embedding"):
            service.get_embedding("test text")


def test_get_embedding_success() -> None:
    """Test successful embedding retrieval."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.return_value = mock_response

        result = service.get_embedding("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="embedding-model",
            input="test text",
        )


def test_get_embedding_raises_on_missing_package() -> None:
    """Should raise RuntimeError if openai package is not installed."""
    service = EmbeddingService(
        endpoint="http://localhost:11434/v1",
        model="embedding-model",
        api_key="test-key",
    )

    original_import = __builtins__["__import__"]

    def failing_import(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=failing_import):
            with pytest.raises(RuntimeError, match="openai package required"):
                service.get_embedding("test text")
    finally:
        pass

