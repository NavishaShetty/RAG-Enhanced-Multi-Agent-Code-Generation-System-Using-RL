"""
RAG Configuration settings.

This module provides configuration for the RAG system including:
- Embedding model settings
- Vector store configuration
- Retrieval parameters
"""

from dataclasses import dataclass
import os


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    # Vector store settings
    collection_name: str = "code_knowledge"
    persist_directory: str = "./chroma_db"

    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Knowledge base settings
    knowledge_base_path: str = "./knowledge_base"

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval settings
    top_k: int = 5
    relevance_threshold: float = 0.3

    # Context formatting
    max_context_length: int = 2000

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create config from environment variables."""
        return cls(
            collection_name=os.getenv("RAG_COLLECTION_NAME", "code_knowledge"),
            persist_directory=os.getenv("RAG_PERSIST_DIR", "./chroma_db"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            knowledge_base_path=os.getenv("RAG_KB_PATH", "./knowledge_base"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            relevance_threshold=float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.3")),
        )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 <= self.relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if self.chunk_size < 100:
            raise ValueError("chunk_size should be at least 100")
