"""
Test Suite for Fashion AI Chatbot

Tests the core functionality of embeddings, retrieval, and chat generation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from dotenv import load_dotenv

from src.embeddings import EmbeddingGenerator
from src.retrieval import FashionDataLoader, RAGRetriever
from src.chatbot import FashionChatbot

# Load environment variables
load_dotenv()

# Test data path
DATA_PATH = "data/2023_fashion_trends.csv"


class TestEmbeddings:
    """Test embedding generation."""

    def test_embedding_generation(self):
        """Test that embeddings are generated correctly."""
        generator = EmbeddingGenerator()
        texts = ["fashion trend", "style guide"]

        embeddings = generator.generate_embeddings(texts, normalize=False)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Embedding dimension
        assert embeddings.dtype == np.float32

    def test_embedding_normalization(self):
        """Test that normalized embeddings have unit length."""
        generator = EmbeddingGenerator()
        texts = ["fashion trend"]

        embeddings = generator.generate_embeddings(texts, normalize=True)

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_single_embedding(self):
        """Test single text embedding."""
        generator = EmbeddingGenerator()
        text = "2023 fashion trends"

        embedding = generator.generate_single_embedding(text)

        assert embedding.ndim == 1
        assert len(embedding) > 0


class TestDataLoader:
    """Test data loading and preprocessing."""

    def test_load_data(self):
        """Test CSV data loading."""
        loader = FashionDataLoader(DATA_PATH)
        df = loader.load_and_preprocess()

        assert len(df) > 0
        assert "trend_name" in df.columns
        assert "text" in df.columns
        assert "URL" in df.columns

    def test_trend_name_extraction(self):
        """Test trend name extraction."""
        text = "2023 Fashion Trend: Bold Colors. Bright hues are trending."
        name = FashionDataLoader._extract_trend_name(text)

        assert "Bold Colors" in name

    def test_no_empty_trends(self):
        """Test that empty trends are filtered out."""
        loader = FashionDataLoader(DATA_PATH)
        df = loader.load_and_preprocess()

        assert (df["Trends"] == "").sum() == 0


class TestRetrieval:
    """Test semantic search and retrieval."""

    @pytest.fixture
    def retriever(self):
        """Create a retriever instance."""
        ret = RAGRetriever(DATA_PATH, use_cache=False)
        ret.initialize()
        return ret

    def test_retrieval_initialization(self, retriever):
        """Test that retriever initializes correctly."""
        assert retriever.df is not None
        assert retriever.embeddings is not None
        assert retriever.search_engine is not None

    def test_semantic_search(self, retriever):
        """Test semantic search returns relevant results."""
        query = "utilitarian fashion trends"
        context, sources = retriever.retrieve(query, top_k=3)

        assert len(sources) == 3
        assert "similarity" in sources.columns
        assert context != ""

    def test_similarity_scores(self, retriever):
        """Test that similarity scores are in valid range."""
        query = "red color trends"
        _, sources = retriever.retrieve(query, top_k=5)

        similarities = sources["similarity"].values
        assert all(0 <= s <= 1 for s in similarities)

        # Scores should be sorted in descending order
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))


class TestChatbot:
    """Test chatbot functionality."""

    @pytest.fixture
    def chatbot(self):
        """Create a chatbot instance."""
        retriever = RAGRetriever(DATA_PATH, use_cache=False)
        retriever.initialize()
        return FashionChatbot(retriever=retriever)

    def test_basic_response(self, chatbot):
        """Test basic mode response."""
        question = "What are the fashion trends for 2023?"
        response = chatbot.ask_basic(question)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_rag_response(self, chatbot):
        """Test RAG mode response with sources."""
        question = "Which trends involve bold colors?"
        response, sources = chatbot.ask_with_rag(question, top_k=3)

        assert isinstance(response, str)
        assert len(response) > 0
        assert len(sources) == 3

    def test_compare_modes(self, chatbot):
        """Test mode comparison."""
        question = "What denim trends are popular?"
        result = chatbot.compare_modes(question, top_k=3)

        assert "basic_response" in result
        assert "rag_response" in result
        assert "sources" in result
        assert len(result["sources"]) == 3


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Fashion AI Chatbot Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
