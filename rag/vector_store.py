"""
Vector store implementation using ChromaDB.

This module handles:
- Creating/loading ChromaDB collections
- Adding documents with embeddings
- Querying for similar documents
"""

import logging
from typing import List, Dict, Optional, Any
import os

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "chromadb is required. Install it with: pip install chromadb"
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. Install it with: pip install sentence-transformers"
    )

from .config import RAGConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for RAG."""

    def __init__(
        self,
        collection_name: str = "code_knowledge",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model for embeddings
            config: Optional RAGConfig object to use instead of individual params
        """
        if config:
            collection_name = config.collection_name
            persist_directory = config.persist_directory
            embedding_model = config.embedding_model

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client with persistence
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            f"Vector store initialized. Collection '{collection_name}' has "
            f"{self.collection.count()} documents."
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document strings
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of unique IDs for each document
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Generate empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self._embed(documents)

        # Add to collection in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
            )

        logger.info(f"Added {len(documents)} documents to vector store")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents.

        Args:
            query_text: The query string
            n_results: Number of results to return

        Returns:
            List of dicts with 'content', 'metadata', 'score' keys
        """
        if not query_text.strip():
            logger.warning("Empty query provided")
            return []

        # Check if collection has documents
        if self.collection.count() == 0:
            logger.warning("Vector store is empty")
            return []

        # Generate query embedding
        query_embedding = self._embed([query_text])[0]

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity score (cosine distance to similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                # Convert to similarity: 1 - (distance / 2)
                similarity = 1 - (distance / 2)

                formatted_results.append(
                    {
                        "content": doc,
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"]
                            else {}
                        ),
                        "score": similarity,
                        "distance": distance,
                    }
                )

        logger.debug(f"Query returned {len(formatted_results)} results")
        return formatted_results

    def clear(self) -> None:
        """Clear all documents from the collection."""
        logger.info(f"Clearing collection '{self.collection_name}'")

        # Delete the collection and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Collection cleared")

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def get_all_ids(self) -> List[str]:
        """Get all document IDs in the collection."""
        result = self.collection.get()
        return result["ids"] if result else []
