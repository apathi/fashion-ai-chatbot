"""
Chatbot Module - Fashion AI Chatbot

This module implements the generation component of RAG (Retrieval-Augmented Generation).
It handles chat completions using OpenAI's GPT models with optional retrieval context.

Key Concepts:
- RAG (Retrieval-Augmented Generation): Enhancing LLM responses with retrieved context
- Prompt Engineering: Crafting effective prompts to guide model behavior
- Knowledge Grounding: Using retrieved documents to reduce hallucination
- Temperature: Controls randomness in generation (lower = more focused)
"""

import os
from typing import List, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI

from src.retrieval import RAGRetriever


class FashionChatbot:
    """
    Fashion trend advisor chatbot with RAG capabilities.

    Supports two modes:
    1. Basic mode: Standard chat without retrieval (may hallucinate)
    2. RAG mode: Context-aware responses using retrieved trends
    """

    # System prompt template for RAG mode
    RAG_SYSTEM_TEMPLATE = (
        "You are a fashion-savvy assistant who recommends 2023 style trends. "
        "Use ONLY the provided context to answer questions. "
        "Reference trend names, design details, sources, and URLs when helpful. "
        "If the context doesn't contain relevant information, say so honestly."
    )

    # System prompt for basic mode
    BASIC_SYSTEM_PROMPT = (
        "You are a helpful fashion assistant who provides advice on 2023 trends."
    )

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        chat_model: Optional[str] = None
    ):
        """
        Initialize the chatbot.

        Args:
            retriever: RAG retriever for context-aware responses
            chat_model: OpenAI chat model to use
        """
        self.retriever = retriever
        self.chat_model = chat_model or os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ask_basic(
        self,
        question: str,
        temperature: float = 0.7
    ) -> str:
        """
        Ask the chatbot without retrieval context (basic mode).

        This mode may produce generic or hallucinated responses since it
        doesn't have access to the specific 2023 trends dataset.

        Args:
            question: User's question
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Chat response
        """
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": self.BASIC_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=temperature
        )

        return response.choices[0].message.content.strip()

    def ask_with_rag(
        self,
        question: str,
        top_k: int = 3,
        temperature: float = 0.2
    ) -> Tuple[str, pd.DataFrame]:
        """
        Ask the chatbot with retrieval context (RAG mode).

        This mode:
        1. Retrieves relevant fashion trends via semantic search
        2. Injects them as context in the system prompt
        3. Generates a grounded, citation-backed response

        Args:
            question: User's question
            top_k: Number of trends to retrieve as context
            temperature: Sampling temperature (lower for factual responses)

        Returns:
            Tuple of (response, source_documents)
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not configured for RAG mode")

        # Retrieve relevant context
        context, source_docs = self.retriever.retrieve(question, top_k)

        # Construct RAG prompt
        system_prompt = f"{self.RAG_SYSTEM_TEMPLATE}\n\nContext:\n{context}"

        # Generate response
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=temperature
        )

        answer = response.choices[0].message.content.strip()

        return answer, source_docs

    def compare_modes(
        self,
        question: str,
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Compare basic vs RAG responses for the same question.

        Useful for demonstrating the value of RAG.

        Args:
            question: User's question
            top_k: Number of trends to retrieve for RAG mode

        Returns:
            Dictionary with both responses and metadata
        """
        # Get basic response
        basic_response = self.ask_basic(question)

        # Get RAG response
        rag_response, source_docs = self.ask_with_rag(question, top_k)

        return {
            "question": question,
            "basic_response": basic_response,
            "rag_response": rag_response,
            "sources": source_docs[["trend_name", "Source", "URL", "similarity"]].to_dict("records")
        }


class ConversationManager:
    """
    Manages multi-turn conversations with context history.

    Note: For simplicity, this implementation doesn't maintain full
    conversation history. Each question is treated independently.
    For production, you would extend this to maintain chat history.
    """

    def __init__(self, chatbot: FashionChatbot):
        """
        Initialize conversation manager.

        Args:
            chatbot: FashionChatbot instance
        """
        self.chatbot = chatbot
        self.history: List[Dict[str, str]] = []

    def send_message(
        self,
        message: str,
        mode: str = "rag",
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Send a message and get a response.

        Args:
            message: User's message
            mode: "rag" or "basic"
            top_k: Number of context documents (for RAG mode)

        Returns:
            Response dictionary
        """
        if mode == "rag":
            response, sources = self.chatbot.ask_with_rag(message, top_k)
            result = {
                "message": message,
                "response": response,
                "mode": "rag",
                "sources": sources[["trend_name", "Source", "URL", "similarity"]].to_dict("records")
            }
        else:
            response = self.chatbot.ask_basic(message)
            result = {
                "message": message,
                "response": response,
                "mode": "basic",
                "sources": []
            }

        # Store in history
        self.history.append(result)

        return result

    def get_history(self) -> List[Dict[str, any]]:
        """
        Get conversation history.

        Returns:
            List of conversation turns
        """
        return self.history

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []


def create_chatbot(
    data_path: str = "data/2023_fashion_trends.csv",
    use_cache: bool = True
) -> FashionChatbot:
    """
    Factory function to create a fully initialized chatbot.

    Args:
        data_path: Path to fashion trends data
        use_cache: Whether to cache embeddings

    Returns:
        Initialized FashionChatbot with RAG capabilities
    """
    # Create and initialize retriever
    retriever = RAGRetriever(data_path, use_cache=use_cache)
    retriever.initialize()

    # Create chatbot
    chatbot = FashionChatbot(retriever=retriever)

    return chatbot
