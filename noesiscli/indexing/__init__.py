"""
Embedding generation and local index persistence for NoesisCLI.
"""

from noesiscli.indexing.embedding import EmbeddingGenerator
from noesiscli.indexing.vector_store import ChromaVectorStore

__all__ = ["EmbeddingGenerator", "ChromaVectorStore"]
