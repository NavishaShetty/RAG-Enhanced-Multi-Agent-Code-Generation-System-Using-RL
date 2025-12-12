"""
Knowledge base document loading and processing.

This module handles:
- Loading markdown files from knowledge_base/
- Chunking documents appropriately
- Preparing documents for vector store
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .config import RAGConfig

logger = logging.getLogger(__name__)


class KnowledgeBaseLoader:
    """Load and process knowledge base documents."""

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize loader with path to knowledge base.

        Args:
            knowledge_base_path: Path to directory containing knowledge base files
            config: Optional RAGConfig for chunk settings
        """
        self.kb_path = Path(knowledge_base_path)
        self.config = config or RAGConfig()
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap

    def load_all_documents(self) -> List[Dict]:
        """
        Load all documents from knowledge base.

        Returns:
            List of dicts with 'content', 'metadata' keys
        """
        if not self.kb_path.exists():
            logger.warning(f"Knowledge base path does not exist: {self.kb_path}")
            return []

        documents = []
        md_files = list(self.kb_path.glob("*.md"))

        if not md_files:
            logger.warning(f"No markdown files found in {self.kb_path}")
            return []

        logger.info(f"Found {len(md_files)} markdown files in knowledge base")

        for md_file in md_files:
            try:
                file_docs = self._process_file(md_file)
                documents.extend(file_docs)
                logger.debug(f"Loaded {len(file_docs)} chunks from {md_file.name}")
            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")
                continue

        logger.info(f"Loaded {len(documents)} total document chunks")
        return documents

    def _process_file(self, file_path: Path) -> List[Dict]:
        """
        Process a single markdown file into chunks.

        Args:
            file_path: Path to the markdown file

        Returns:
            List of document chunks with metadata
        """
        content = file_path.read_text(encoding="utf-8")

        # Split by sections (## headers)
        sections = self._split_by_sections(content)

        chunks = []
        for section_title, section_content in sections:
            # Further chunk if section is too long
            section_chunks = self._chunk_text(section_content)

            for i, chunk in enumerate(section_chunks):
                if chunk.strip():
                    chunks.append(
                        {
                            "content": chunk.strip(),
                            "metadata": {
                                "source": file_path.name,
                                "section": section_title,
                                "chunk_index": i,
                                "total_chunks": len(section_chunks),
                            },
                        }
                    )

        return chunks

    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        Split content by markdown ## headers.

        Args:
            content: Full markdown content

        Returns:
            List of (section_title, section_content) tuples
        """
        # Pattern to match ## headers
        section_pattern = r"^##\s+(.+?)$"

        lines = content.split("\n")
        sections = []
        current_title = "Introduction"
        current_content = []

        for line in lines:
            match = re.match(section_pattern, line)
            if match:
                # Save previous section if it has content
                if current_content:
                    sections.append(
                        (current_title, "\n".join(current_content).strip())
                    )
                current_title = match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            sections.append((current_title, "\n".join(current_content).strip()))

        return sections

    def _chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Split text into chunks with overlap.

        Uses markdown-aware chunking to preserve code blocks.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        if len(text) <= chunk_size:
            return [text]

        # Try to split by code blocks first to keep them intact
        code_block_pattern = r"```[\s\S]*?```"
        parts = re.split(f"({code_block_pattern})", text)

        chunks = []
        current_chunk = ""

        for part in parts:
            # If it's a code block, try to keep it intact
            if re.match(code_block_pattern, part):
                if len(current_chunk) + len(part) <= chunk_size:
                    current_chunk += part
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # If code block itself is too large, we have to split it
                    if len(part) > chunk_size:
                        # Split the code block
                        for i in range(0, len(part), chunk_size - overlap):
                            chunk = part[i : i + chunk_size]
                            if chunk.strip():
                                chunks.append(chunk.strip())
                    else:
                        current_chunk = part
            else:
                # Regular text - split by sentences/paragraphs
                sentences = re.split(r"(?<=[.!?])\s+|\n\n", part)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def extract_code_blocks(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks with their descriptions.

        Args:
            content: Markdown content

        Returns:
            List of (description, code) tuples
        """
        code_blocks = []

        # Pattern to match code blocks with optional preceding description
        pattern = r"(?:([^\n]+)\n)?```(?:python)?\n([\s\S]*?)```"

        matches = re.findall(pattern, content)
        for description, code in matches:
            code_blocks.append((description.strip() if description else "", code.strip()))

        return code_blocks

    def get_stats(self) -> Dict:
        """
        Get statistics about the knowledge base.

        Returns:
            Dict with file count, total chunks, etc.
        """
        if not self.kb_path.exists():
            return {"error": "Knowledge base path does not exist"}

        md_files = list(self.kb_path.glob("*.md"))
        total_size = sum(f.stat().st_size for f in md_files)

        documents = self.load_all_documents()

        return {
            "file_count": len(md_files),
            "total_chunks": len(documents),
            "total_size_kb": total_size / 1024,
            "files": [f.name for f in md_files],
        }
