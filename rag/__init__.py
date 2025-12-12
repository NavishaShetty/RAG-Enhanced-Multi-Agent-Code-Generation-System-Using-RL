"""
RAG (Retrieval-Augmented Generation) module for code generation.

This module provides:
- Vector store for semantic search using ChromaDB
- Knowledge base loading and processing
- Retrieval pipeline for LLM context augmentation
"""

from .vector_store import VectorStore
from .knowledge_base import KnowledgeBaseLoader
from .retriever import Retriever
from .config import RAGConfig

__all__ = [
    "VectorStore",
    "KnowledgeBaseLoader",
    "Retriever",
    "RAGConfig",
]
