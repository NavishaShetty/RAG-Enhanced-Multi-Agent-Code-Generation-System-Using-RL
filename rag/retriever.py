"""
RAG Retriever - main interface for the RAG system.

This module provides:
- High-level retrieval API
- Context formatting for LLM prompts
- Relevance filtering
"""

import logging
from typing import List, Dict, Optional, Any

from .vector_store import VectorStore
from .knowledge_base import KnowledgeBaseLoader
from .config import RAGConfig

logger = logging.getLogger(__name__)


class Retriever:
    """Main RAG retriever interface."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: Optional pre-configured VectorStore
            config: Optional RAGConfig for settings
        """
        self.config = config or RAGConfig()
        self.vector_store = vector_store
        self._initialized = False

        # Set relevance threshold from config
        self.relevance_threshold = self.config.relevance_threshold

    def initialize(
        self,
        knowledge_base_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Initialize the retriever by loading knowledge base.

        Call this once at startup to populate the vector store.

        Args:
            knowledge_base_path: Path to knowledge base directory
            force_reload: If True, clear existing data and reload
        """
        kb_path = knowledge_base_path or self.config.knowledge_base_path

        # Initialize vector store if not provided
        if self.vector_store is None:
            self.vector_store = VectorStore(config=self.config)

        # Check if already populated
        if self.vector_store.count() > 0 and not force_reload:
            logger.info(
                f"Vector store already has {self.vector_store.count()} documents. "
                "Skipping initialization. Use force_reload=True to reload."
            )
            self._initialized = True
            return

        # Clear if force reloading
        if force_reload:
            logger.info("Force reloading: clearing existing documents")
            self.vector_store.clear()

        # Load knowledge base
        logger.info(f"Loading knowledge base from: {kb_path}")
        loader = KnowledgeBaseLoader(kb_path, config=self.config)
        documents = loader.load_all_documents()

        if not documents:
            logger.warning("No documents loaded from knowledge base")
            self._initialized = True
            return

        # Add to vector store
        self.vector_store.add_documents(
            documents=[d["content"] for d in documents],
            metadatas=[d["metadata"] for d in documents],
            ids=[f"doc_{i}" for i in range(len(documents))],
        )

        logger.info(f"Initialized RAG with {len(documents)} document chunks")
        self._initialized = True

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query (task description)
            n_results: Maximum number of results
            filter_threshold: Optional custom relevance threshold

        Returns:
            List of relevant documents with scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to retriever")
            return []

        # Ensure vector store exists
        if self.vector_store is None:
            logger.warning("Vector store not initialized. Call initialize() first.")
            return []

        # Query vector store
        results = self.vector_store.query(query, n_results=n_results)

        # Filter by relevance threshold
        threshold = filter_threshold or self.relevance_threshold
        filtered_results = [r for r in results if r.get("score", 0) >= threshold]

        logger.debug(
            f"Retrieved {len(filtered_results)}/{len(results)} documents "
            f"above threshold {threshold}"
        )

        return filtered_results

    def format_context(
        self,
        query: str,
        n_results: int = 3,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Retrieve and format context for LLM prompt.

        This is the primary method to use for RAG integration.
        Returns a formatted string ready to inject into the prompt.

        Args:
            query: The task description / search query
            n_results: Maximum number of results to include
            max_length: Maximum length of formatted context

        Returns:
            Formatted string ready to inject into prompt,
            or empty string if no relevant context found
        """
        max_length = max_length or self.config.max_context_length

        documents = self.retrieve(query, n_results)

        if not documents:
            logger.debug("No relevant context found for query")
            return ""

        context_parts = [
            "## Relevant Context from Knowledge Base:\n",
            "Use the following reference examples and patterns to help with the task:\n",
        ]

        current_length = sum(len(p) for p in context_parts)

        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0)

            # Format this reference
            source = metadata.get("source", "unknown")
            section = metadata.get("section", "")

            reference_header = f"\n### Reference {i} (from {source}"
            if section:
                reference_header += f" - {section}"
            reference_header += f", relevance: {score:.2f}):\n"

            # Check if adding this would exceed max length
            if current_length + len(reference_header) + len(content) > max_length:
                # Try to add truncated version
                available_space = max_length - current_length - len(reference_header) - 20
                if available_space > 100:
                    content = content[:available_space] + "..."
                    context_parts.append(reference_header)
                    context_parts.append(content)
                break

            context_parts.append(reference_header)
            context_parts.append(content)
            current_length += len(reference_header) + len(content)

        context = "\n".join(context_parts)
        logger.info(
            f"Formatted RAG context with {len(documents)} references "
            f"({len(context)} chars)"
        )

        return context

    def is_initialized(self) -> bool:
        """Check if retriever has been initialized."""
        return self._initialized and self.vector_store is not None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever.

        Returns:
            Dict with document count, initialization status, etc.
        """
        stats = {
            "initialized": self._initialized,
            "relevance_threshold": self.relevance_threshold,
        }

        if self.vector_store:
            stats["document_count"] = self.vector_store.count()
            stats["collection_name"] = self.vector_store.collection_name
        else:
            stats["document_count"] = 0

        return stats


# Singleton instance for easy import
_default_retriever: Optional[Retriever] = None


def get_retriever(config: Optional[RAGConfig] = None) -> Retriever:
    """
    Get the default retriever instance.

    Creates a new instance if none exists.

    Args:
        config: Optional configuration

    Returns:
        Retriever instance
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever(config=config)
    return _default_retriever


def reset_retriever() -> None:
    """Reset the default retriever instance."""
    global _default_retriever
    _default_retriever = None
