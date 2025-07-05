"""
Embeddings Module - Fashion AI Chatbot

This module handles the generation and caching of text embeddings using OpenAI's
text-embedding-ada-002 model. Embeddings are vector representations of text that
capture semantic meaning, enabling similarity-based retrieval.

Key Concepts:
- Vector Embeddings: Dense numerical representations of text in high-dimensional space
- Semantic Search: Finding similar content based on meaning, not just keywords
- Batch Processing: Efficiently processing multiple texts in batches to reduce API calls
- Caching: Storing embeddings to avoid redundant API calls and reduce latency
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import OpenAI


class EmbeddingGenerator:
    """
    Generates and manages text embeddings for semantic search.

    This class handles:
    1. Batch embedding generation via OpenAI API
    2. Embedding normalization for cosine similarity
    3. Caching to disk for faster subsequent loads
    """

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        cache_path: Optional[str] = None
    ):
        """
        Initialize the embedding generator.

        Args:
            model: OpenAI embedding model to use
            cache_path: Path to cache embeddings (optional)
        """
        self.model = model
        self.cache_path = cache_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts with batching.

        Batching improves efficiency by processing multiple texts per API call.
        Normalization enables cosine similarity via dot product (faster computation).

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per API call
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = []

        # Process in batches to optimize API usage
        for start_idx in range(0, len(texts), batch_size):
            batch = texts[start_idx:start_idx + batch_size]

            # Call OpenAI embeddings API
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            # Extract embeddings from response
            batch_embeddings = [
                np.array(item.embedding, dtype=np.float32)
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            embeddings.extend(batch_embeddings)

        # Stack into matrix
        embedding_matrix = np.vstack(embeddings)

        # Normalize for cosine similarity (optional but recommended)
        if normalize:
            embedding_matrix = self._normalize_embeddings(embedding_matrix)

        return embedding_matrix

    def generate_single_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single text string.

        Args:
            text: Text string to embed
            normalize: Whether to L2-normalize the embedding

        Returns:
            numpy array of shape (embedding_dim,)
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        if normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings for cosine similarity.

        After normalization, cosine similarity = dot product, which is faster.

        Args:
            embeddings: Matrix of embeddings

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Clip to avoid division by zero
        norms = np.clip(norms, a_min=1e-10, a_max=None)
        return embeddings / norms

    def save_to_cache(
        self,
        embeddings: np.ndarray,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Save embeddings to disk cache for faster loading.

        Args:
            embeddings: Embedding matrix to cache
            metadata: Optional metadata to store with embeddings
        """
        if self.cache_path is None:
            return

        cache_dir = Path(self.cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings and metadata
        save_dict = {"embeddings": embeddings}
        if metadata:
            save_dict.update(metadata)

        np.savez_compressed(self.cache_path, **save_dict)
        print(f"Embeddings cached to: {self.cache_path}")

    def load_from_cache(self) -> Optional[np.ndarray]:
        """
        Load embeddings from disk cache.

        Returns:
            Cached embeddings if found, None otherwise
        """
        if self.cache_path is None or not Path(self.cache_path).exists():
            return None

        try:
            data = np.load(self.cache_path)
            embeddings = data["embeddings"]
            print(f"Loaded {len(embeddings)} embeddings from cache")
            return embeddings
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None


def create_embedding_generator(cache_embeddings: bool = True) -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator with default settings.

    Args:
        cache_embeddings: Whether to enable caching

    Returns:
        Configured EmbeddingGenerator instance
    """
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    cache_path = os.getenv("EMBEDDINGS_CACHE_PATH", "data/embeddings_cache.npz")

    if not cache_embeddings:
        cache_path = None

    return EmbeddingGenerator(model=model, cache_path=cache_path)
