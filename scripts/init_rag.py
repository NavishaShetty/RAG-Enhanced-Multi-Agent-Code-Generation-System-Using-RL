#!/usr/bin/env python3
"""
Initialize RAG system by loading knowledge base into vector store.

Run this once before starting the application:
    python scripts/init_rag.py

This script:
1. Loads all markdown documents from the knowledge_base/ directory
2. Chunks them appropriately for retrieval
3. Generates embeddings using sentence-transformers
4. Stores embeddings in ChromaDB for fast retrieval
5. Verifies the setup with a test query
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

from rag.retriever import Retriever
from rag.knowledge_base import KnowledgeBaseLoader
from rag.vector_store import VectorStore
from rag.config import RAGConfig


def setup_logging(verbose: bool = False):
    """Configure logging for the initialization script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """Main entry point for RAG initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize RAG system by loading knowledge base"
    )
    parser.add_argument(
        "--knowledge-base",
        default="./knowledge_base",
        help="Path to knowledge base directory (default: ./knowledge_base)"
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Path to ChromaDB persistence directory (default: ./chroma_db)"
    )
    parser.add_argument(
        "--collection",
        default="code_knowledge",
        help="ChromaDB collection name (default: code_knowledge)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reload even if vector store has existing data"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("RAG System Initialization")
    print("=" * 60)

    # Validate knowledge base path
    kb_path = Path(args.knowledge_base)
    if not kb_path.exists():
        logger.error(f"Knowledge base directory not found: {kb_path}")
        print(f"\nError: Knowledge base directory not found: {kb_path}")
        print("Please create the knowledge_base/ directory with markdown files.")
        sys.exit(1)

    # Count markdown files
    md_files = list(kb_path.glob("*.md"))
    if not md_files:
        logger.error(f"No markdown files found in {kb_path}")
        print(f"\nError: No markdown files found in {kb_path}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Knowledge Base: {kb_path.absolute()}")
    print(f"  Persist Directory: {args.persist_dir}")
    print(f"  Collection Name: {args.collection}")
    print(f"  Markdown Files: {len(md_files)}")
    print()

    # Create config
    config = RAGConfig(
        knowledge_base_path=str(kb_path),
        persist_directory=args.persist_dir,
        collection_name=args.collection
    )

    # Step 1: Load knowledge base
    print("Step 1: Loading knowledge base documents...")
    loader = KnowledgeBaseLoader(str(kb_path), config=config)
    documents = loader.load_all_documents()

    if not documents:
        logger.error("No documents loaded from knowledge base")
        print("Error: No documents could be loaded.")
        sys.exit(1)

    print(f"  Loaded {len(documents)} document chunks from {len(md_files)} files")

    # Show sample of what was loaded
    if args.verbose and documents:
        print("\n  Sample chunks:")
        for i, doc in enumerate(documents[:3]):
            content_preview = doc['content'][:100].replace('\n', ' ')
            print(f"    {i+1}. [{doc['metadata'].get('source', 'unknown')}] {content_preview}...")

    # Step 2: Initialize vector store
    print("\nStep 2: Initializing vector store...")
    store = VectorStore(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_model=config.embedding_model
    )

    existing_count = store.count()
    if existing_count > 0 and not args.force:
        print(f"  Vector store already has {existing_count} documents.")
        response = input("  Clear and reload? [y/N]: ").strip().lower()
        if response != 'y':
            print("  Keeping existing data. Use --force to skip this prompt.")
            print("\nInitialization skipped (existing data preserved).")
            return

    # Clear existing data
    print("  Clearing existing data...")
    store.clear()

    # Step 3: Add documents to vector store
    print("\nStep 3: Adding documents to vector store...")
    print("  Generating embeddings (this may take a moment)...")

    store.add_documents(
        documents=[d['content'] for d in documents],
        metadatas=[d['metadata'] for d in documents],
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    print(f"  Added {len(documents)} documents to vector store")

    # Step 4: Verify with test queries
    print("\nStep 4: Verifying setup with test queries...")

    test_queries = [
        "How to reverse a string in Python?",
        "Write a function to find the maximum value",
        "How to handle exceptions in Python?",
        "What is list comprehension?",
        "How to check if a number is prime?",
    ]

    retriever = Retriever(vector_store=store, config=config)

    all_passed = True
    for query in test_queries:
        results = retriever.retrieve(query, n_results=3)
        if results:
            print(f"  [OK] Query: '{query[:40]}...' -> {len(results)} results")
            if args.verbose and results:
                print(f"       Top result score: {results[0]['score']:.3f}")
        else:
            print(f"  [!!] Query: '{query[:40]}...' -> No results")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("RAG Initialization Complete!")
        print(f"  Total documents: {store.count()}")
        print(f"  Collection: {args.collection}")
        print(f"  Persist path: {Path(args.persist_dir).absolute()}")
    else:
        print("RAG Initialization Complete with Warnings")
        print("Some test queries returned no results.")

    print("=" * 60)

    # Final instructions
    print("\nNext steps:")
    print("  1. Run the Streamlit app: streamlit run app.py")
    print("  2. Or test the system: python demo/demo.py")


if __name__ == "__main__":
    main()
