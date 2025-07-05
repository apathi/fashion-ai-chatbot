"""
Retrieval Module - Fashion AI Chatbot

This module implements the retrieval component of RAG (Retrieval-Augmented Generation).
It handles data loading, preprocessing, and semantic search using vector embeddings.

Key Concepts:
- Cosine Similarity: Measures similarity between vectors (normalized dot product)
- Semantic Search: Finding documents based on meaning rather than exact keyword matching
- Top-K Retrieval: Returning the K most relevant documents for a query
- Context Window: Selected documents that provide context to the language model
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from src.embeddings import EmbeddingGenerator


class FashionDataLoader:
    """
    Loads and preprocesses the 2023 fashion trends dataset.

    Handles:
    1. CSV data loading and validation
    2. Text preprocessing and cleaning
    3. Document construction for embedding
    """

    def __init__(self, data_path: str):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the fashion trends CSV file
        """
        self.data_path = Path(data_path)
        self.df = None

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Load CSV data and preprocess for RAG.

        Returns:
            Preprocessed DataFrame with trend information
        """
        # Load CSV
        self.df = pd.read_csv(self.data_path)

        # Validate required columns
        required_columns = ["URL", "Trends", "Source"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        # Clean data
        self.df = self.df[required_columns].copy()
        self.df = self.df.fillna("").astype(str)
        self.df = self.df.apply(lambda col: col.str.strip())

        # Remove empty trends and duplicates
        self.df = self.df[self.df["Trends"] != ""].copy()
        self.df = self.df.drop_duplicates(subset=["Trends", "Source"])

        # Extract trend names and compose searchable text
        self.df["trend_name"] = self.df["Trends"].apply(self._extract_trend_name)
        self.df["text"] = self.df.apply(self._compose_document_text, axis=1)

        print(f"Loaded {len(self.df)} fashion trends")
        return self.df

    @staticmethod
    def _extract_trend_name(text: str) -> str:
        """
        Extract a concise trend name from the full description.

        Args:
            text: Full trend description

        Returns:
            Extracted trend name
        """
        prefix = "2023 Fashion Trend:"
        first_sentence = text.split(".")[0].strip()

        if prefix in first_sentence:
            return first_sentence.replace(prefix, "").strip()

        return first_sentence if first_sentence else "Featured fashion trend"

    @staticmethod
    def _compose_document_text(row: pd.Series) -> str:
        """
        Compose a rich text representation for embedding.

        This combines trend name, description, source, and URL into a single
        searchable document that captures all relevant information.

        Args:
            row: DataFrame row containing trend information

        Returns:
            Composed document text
        """
        trend_name = row["trend_name"]
        description = row["Trends"]
        source = row["Source"] or "Unknown source"
        url = row["URL"] or "No URL provided"

        return (
            f"{trend_name} is highlighted in 2023 fashion coverage. "
            f"Description: {description} "
            f"(Source: {source}; URL: {url})."
        )


class SemanticSearchEngine:
    """
    Performs semantic search over fashion trends using vector embeddings.

    This is the core retrieval component of RAG:
    1. Embed user query into vector space
    2. Compute similarity with all document embeddings
    3. Return top-K most relevant documents
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize the search engine.

        Args:
            df: DataFrame containing trend documents
            embeddings: Pre-computed document embeddings (normalized)
            embedding_generator: Generator for query embeddings
        """
        self.df = df
        self.embeddings = embeddings
        self.embedding_generator = embedding_generator

        # Validate dimensions
        if len(df) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(df)} documents but {len(embeddings)} embeddings"
            )

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Perform semantic search for a user query.

        Process:
        1. Generate embedding for query
        2. Compute cosine similarity with all documents
        3. Select top-K most similar documents
        4. Return query embedding and relevant documents

        Args:
            query: User's question or search query
            top_k: Number of top results to return

        Returns:
            Tuple of (query_embedding, top_documents_df)
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(
            query,
            normalize=True
        )

        # Compute cosine similarity scores
        # Since embeddings are normalized, cosine similarity = dot product
        similarity_scores = self.embeddings @ query_embedding

        # Get top-K indices
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

        # Retrieve top documents
        top_docs = self.df.iloc[top_indices].copy()
        top_docs["similarity"] = similarity_scores[top_indices]

        return query_embedding, top_docs

    def get_context_for_prompt(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[str, pd.DataFrame]:
        """
        Retrieve and format context for RAG prompt.

        This method:
        1. Performs semantic search
        2. Formats retrieved documents as context string
        3. Returns formatted context and source documents

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (formatted_context, source_documents)
        """
        _, top_docs = self.search(query, top_k)

        # Format context for LLM prompt
        context_parts = []
        for idx, row in top_docs.iterrows():
            trend_name = row["trend_name"] or "Fashion trend"
            source = row["Source"] or "Unknown source"
            url = row["URL"] or "No URL provided"
            description = row["Trends"]

            context_parts.append(
                f"{trend_name} — Source: {source}. "
                f"URL: {url}. Summary: {description}"
            )

        context = "\n\n".join(context_parts)

        return context, top_docs


class RAGRetriever:
    """
    High-level RAG retrieval system that combines data loading,
    embedding generation, and semantic search.
    """

    def __init__(
        self,
        data_path: str,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        use_cache: bool = True
    ):
        """
        Initialize the RAG retriever.

        Args:
            data_path: Path to fashion trends CSV
            embedding_generator: Optional custom embedding generator
            use_cache: Whether to use embedding cache
        """
        self.data_path = data_path
        self.use_cache = use_cache

        # Create embedding generator if not provided
        if embedding_generator is None:
            from src.embeddings import create_embedding_generator
            self.embedding_generator = create_embedding_generator(use_cache)
        else:
            self.embedding_generator = embedding_generator

        # Initialize components
        self.data_loader = FashionDataLoader(data_path)
        self.df = None
        self.embeddings = None
        self.search_engine = None

    def initialize(self) -> None:
        """
        Load data and generate/load embeddings.

        This must be called before using the retriever.
        """
        # Load and preprocess data
        self.df = self.data_loader.load_and_preprocess()
        documents = self.df["text"].tolist()

        # Try to load from cache
        if self.use_cache:
            self.embeddings = self.embedding_generator.load_from_cache()

        # Generate embeddings if not cached
        if self.embeddings is None:
            print("Generating embeddings (this may take a minute)...")
            self.embeddings = self.embedding_generator.generate_embeddings(
                documents,
                normalize=True
            )

            # Cache for future use
            if self.use_cache:
                self.embedding_generator.save_to_cache(
                    self.embeddings,
                    metadata={"num_documents": len(documents)}
                )

        # Create search engine
        self.search_engine = SemanticSearchEngine(
            self.df,
            self.embeddings,
            self.embedding_generator
        )

        print(f"RAG system initialized with {len(self.df)} trends")

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[str, pd.DataFrame]:
        """
        Retrieve context for a query.

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (formatted_context, source_documents)
        """
        if self.search_engine is None:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")

        return self.search_engine.get_context_for_prompt(query, top_k)
