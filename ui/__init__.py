"""
UI Module for RAG-Enhanced Multi-Agent Code Generation System.

This module provides:
- Streamlit-based web interface
- Pipeline connector to multi-agent system
- Real-time visualization components
"""

from .pipeline import UIPipeline

__all__ = ["UIPipeline"]
