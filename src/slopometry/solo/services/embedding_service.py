"""Embedding service for memory similarity calculations."""

import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings for memory content using an OpenAI-compatible API."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for a text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package required for embeddings. Install with: pip install openai")

        try:
            client = OpenAI(base_url=self.endpoint, api_key=self.api_key)

            response = client.embeddings.create(
                model=self.model,
                input=text,
            )

            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            raise RuntimeError("Empty response from embedding endpoint")

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding: {e}") from e

    def compute_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        import math

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def compute_uniqueness_score(
        self,
        embedding: list[float],
        existing_embeddings: list[list[float]],
    ) -> float:
        """Compute uniqueness score (1 - avg similarity to existing).

        Higher score = more unique compared to existing memories.
        Score of 1.0 = completely unique (no similarity to any existing).
        Score of 0.0 = identical to existing memories.

        Args:
            embedding: New embedding to score
            existing_embeddings: List of existing embeddings to compare against

        Returns:
            Uniqueness score between 0.0 and 1.0
        """
        if not existing_embeddings:
            return 1.0

        similarities = [self.compute_similarity(embedding, existing) for existing in existing_embeddings]
        avg_similarity = sum(similarities) / len(similarities)

        uniqueness = 1.0 - max(0.0, min(1.0, avg_similarity))
        return round(uniqueness, 2)
