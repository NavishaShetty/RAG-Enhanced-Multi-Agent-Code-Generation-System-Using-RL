"""
Tests for RAG (Retrieval-Augmented Generation) system.

Run with: pytest tests/test_rag.py -v
"""

import pytest
import os
import sys
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig
from rag.vector_store import VectorStore
from rag.knowledge_base import KnowledgeBaseLoader
from rag.retriever import Retriever


class TestRAGConfig:
    """Tests for RAG configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAGConfig()
        assert config.collection_name == "code_knowledge"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.top_k == 5
        assert 0 <= config.relevance_threshold <= 1

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid top_k
        with pytest.raises(ValueError):
            RAGConfig(top_k=0)

        # Invalid relevance_threshold
        with pytest.raises(ValueError):
            RAGConfig(relevance_threshold=1.5)

        # Invalid chunk_size
        with pytest.raises(ValueError):
            RAGConfig(chunk_size=50)

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        os.environ["RAG_TOP_K"] = "10"
        config = RAGConfig.from_env()
        assert config.top_k == 10
        del os.environ["RAG_TOP_K"]


class TestVectorStore:
    """Tests for vector store implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for vector store."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create a vector store instance."""
        return VectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_model="all-MiniLM-L6-v2"
        )

    def test_add_and_query_documents(self, vector_store):
        """Test adding documents and querying."""
        # Add documents
        documents = [
            "Python list comprehension is a concise way to create lists.",
            "The reverse function can be implemented using slicing: s[::-1]",
            "Binary search requires a sorted array for efficient searching.",
        ]

        vector_store.add_documents(
            documents=documents,
            ids=["doc_0", "doc_1", "doc_2"]
        )

        # Query
        results = vector_store.query("How to reverse a string?", n_results=2)

        assert len(results) > 0
        assert "content" in results[0]
        assert "score" in results[0]
        # Should find the reverse-related document
        assert any("reverse" in r["content"].lower() for r in results)

    def test_query_empty_store(self, vector_store):
        """Test querying an empty vector store."""
        results = vector_store.query("test query")
        assert len(results) == 0

    def test_clear_collection(self, vector_store):
        """Test clearing the collection."""
        # Add some documents
        vector_store.add_documents(
            documents=["Test document"],
            ids=["doc_0"]
        )
        assert vector_store.count() == 1

        # Clear
        vector_store.clear()
        assert vector_store.count() == 0

    def test_empty_query(self, vector_store):
        """Test handling empty queries."""
        results = vector_store.query("")
        assert len(results) == 0

        results = vector_store.query("   ")
        assert len(results) == 0


class TestKnowledgeBaseLoader:
    """Tests for knowledge base loading."""

    @pytest.fixture
    def temp_kb(self):
        """Create a temporary knowledge base directory."""
        temp_path = tempfile.mkdtemp()

        # Create sample markdown files
        content1 = """# Python Basics

## Topic: String Reversal

### Description
How to reverse a string in Python.

### Example
```python
def reverse_string(s):
    return s[::-1]
```
"""
        content2 = """# Python Patterns

## Topic: List Comprehension

### Description
Creating lists concisely.

### Example
```python
squares = [x**2 for x in range(10)]
```
"""
        with open(os.path.join(temp_path, "basics.md"), "w") as f:
            f.write(content1)

        with open(os.path.join(temp_path, "patterns.md"), "w") as f:
            f.write(content2)

        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_load_documents(self, temp_kb):
        """Test loading documents from knowledge base."""
        loader = KnowledgeBaseLoader(temp_kb)
        documents = loader.load_all_documents()

        assert len(documents) > 0
        assert all("content" in d for d in documents)
        assert all("metadata" in d for d in documents)

    def test_chunk_preservation(self, temp_kb):
        """Test that code blocks are preserved in chunks."""
        loader = KnowledgeBaseLoader(temp_kb)
        documents = loader.load_all_documents()

        # Check that we have code-containing chunks
        code_chunks = [d for d in documents if "```" in d["content"] or "def " in d["content"]]
        assert len(code_chunks) > 0

    def test_metadata_extraction(self, temp_kb):
        """Test that metadata is correctly extracted."""
        loader = KnowledgeBaseLoader(temp_kb)
        documents = loader.load_all_documents()

        # Check that source files are recorded
        sources = {d["metadata"].get("source") for d in documents}
        assert "basics.md" in sources or any("basics" in s for s in sources if s)

    def test_nonexistent_directory(self):
        """Test handling of nonexistent directory."""
        loader = KnowledgeBaseLoader("/nonexistent/path")
        documents = loader.load_all_documents()
        assert documents == []

    def test_get_stats(self, temp_kb):
        """Test knowledge base statistics."""
        loader = KnowledgeBaseLoader(temp_kb)
        stats = loader.get_stats()

        assert stats["file_count"] == 2
        assert stats["total_chunks"] > 0
        assert "basics.md" in stats["files"]


class TestRetriever:
    """Tests for the retriever interface."""

    @pytest.fixture
    def temp_setup(self):
        """Create temporary directories for testing."""
        temp_db = tempfile.mkdtemp()
        temp_kb = tempfile.mkdtemp()

        # Create sample knowledge base
        content = """# Python Reference

## Topic: Reverse String

### Description
Reverse a string using slicing.

### Example
```python
def reverse_string(s: str) -> str:
    return s[::-1]
```

## Topic: Find Maximum

### Description
Find the maximum value in a list.

### Example
```python
def find_max(nums: list) -> int:
    return max(nums)
```

## Topic: Prime Check

### Description
Check if a number is prime.

### Example
```python
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```
"""
        with open(os.path.join(temp_kb, "reference.md"), "w") as f:
            f.write(content)

        yield {"db": temp_db, "kb": temp_kb}

        shutil.rmtree(temp_db, ignore_errors=True)
        shutil.rmtree(temp_kb, ignore_errors=True)

    def test_retriever_initialization(self, temp_setup):
        """Test retriever initialization."""
        config = RAGConfig(
            persist_directory=temp_setup["db"],
            knowledge_base_path=temp_setup["kb"]
        )
        retriever = Retriever(config=config)

        assert not retriever.is_initialized()

        retriever.initialize()

        assert retriever.is_initialized()

    def test_retrieve_relevant_documents(self, temp_setup):
        """Test retrieving relevant documents."""
        config = RAGConfig(
            persist_directory=temp_setup["db"],
            knowledge_base_path=temp_setup["kb"],
            relevance_threshold=0.1  # Lower threshold for testing
        )
        retriever = Retriever(config=config)
        retriever.initialize()

        # Query for string reversal
        results = retriever.retrieve("How to reverse a string?", n_results=3)

        assert len(results) > 0
        # Top result should be relevant to string reversal
        top_content = results[0]["content"].lower()
        assert "reverse" in top_content or "string" in top_content

    def test_format_context(self, temp_setup):
        """Test context formatting for LLM prompts."""
        config = RAGConfig(
            persist_directory=temp_setup["db"],
            knowledge_base_path=temp_setup["kb"],
            relevance_threshold=0.1
        )
        retriever = Retriever(config=config)
        retriever.initialize()

        context = retriever.format_context("reverse a string")

        # Should have header
        assert "Relevant Context" in context or context == ""

        # Should have references if content found
        if context:
            assert "Reference" in context

    def test_empty_query(self, temp_setup):
        """Test handling of empty queries."""
        config = RAGConfig(
            persist_directory=temp_setup["db"],
            knowledge_base_path=temp_setup["kb"]
        )
        retriever = Retriever(config=config)
        retriever.initialize()

        results = retriever.retrieve("")
        assert len(results) == 0

        context = retriever.format_context("")
        assert context == ""

    def test_get_stats(self, temp_setup):
        """Test retriever statistics."""
        config = RAGConfig(
            persist_directory=temp_setup["db"],
            knowledge_base_path=temp_setup["kb"]
        )
        retriever = Retriever(config=config)

        stats = retriever.get_stats()
        assert "initialized" in stats
        assert not stats["initialized"]

        retriever.initialize()

        stats = retriever.get_stats()
        assert stats["initialized"]
        assert stats["document_count"] > 0


class TestIntegration:
    """Integration tests for the RAG system."""

    @pytest.fixture
    def full_setup(self):
        """Create full test setup."""
        temp_db = tempfile.mkdtemp()
        temp_kb = tempfile.mkdtemp()

        # Create comprehensive knowledge base
        basics_content = """# Python Basics

## Topic: String Reversal

### Description
Reversing a string in Python can be done efficiently using slicing.

### Example
```python
def reverse_string(s: str) -> str:
    return s[::-1]
```

### Best Practice
Use slice notation for simplicity and performance.

## Topic: Finding Maximum

### Description
Find the maximum value in a list.

### Example
```python
def find_max(numbers: list) -> int:
    if not numbers:
        raise ValueError("List cannot be empty")
    return max(numbers)
```
"""
        patterns_content = """# Python Patterns

## Topic: List Comprehension

### Description
Concise way to create lists.

### Example
```python
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

## Topic: Dictionary Comprehension

### Example
```python
squares_dict = {x: x**2 for x in range(6)}
```
"""
        with open(os.path.join(temp_kb, "basics.md"), "w") as f:
            f.write(basics_content)

        with open(os.path.join(temp_kb, "patterns.md"), "w") as f:
            f.write(patterns_content)

        yield {"db": temp_db, "kb": temp_kb}

        shutil.rmtree(temp_db, ignore_errors=True)
        shutil.rmtree(temp_kb, ignore_errors=True)

    def test_full_rag_pipeline(self, full_setup):
        """Test the complete RAG pipeline."""
        # Configure
        config = RAGConfig(
            persist_directory=full_setup["db"],
            knowledge_base_path=full_setup["kb"],
            relevance_threshold=0.1
        )

        # Load knowledge base
        loader = KnowledgeBaseLoader(full_setup["kb"], config=config)
        documents = loader.load_all_documents()
        assert len(documents) > 0

        # Create vector store
        store = VectorStore(
            persist_directory=full_setup["db"],
            collection_name="test_full"
        )
        store.add_documents(
            documents=[d["content"] for d in documents],
            metadatas=[d["metadata"] for d in documents]
        )
        assert store.count() > 0

        # Create retriever
        retriever = Retriever(vector_store=store, config=config)
        retriever._initialized = True

        # Test retrieval
        results = retriever.retrieve("How do I reverse a string?", n_results=3)
        assert len(results) > 0

        # Test context formatting
        context = retriever.format_context("reverse a string")
        assert len(context) > 0

    def test_retriever_with_coder_agent_integration(self, full_setup):
        """Test integration between retriever and coder agent."""
        # This test verifies the interface but doesn't make actual LLM calls

        config = RAGConfig(
            persist_directory=full_setup["db"],
            knowledge_base_path=full_setup["kb"],
            relevance_threshold=0.1
        )

        retriever = Retriever(config=config)
        retriever.initialize()

        # Verify retriever provides context in expected format
        context = retriever.format_context("reverse a string")

        # Context should be a string
        assert isinstance(context, str)

        # If context found, should have proper structure
        if context:
            assert "Reference" in context or "Context" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
