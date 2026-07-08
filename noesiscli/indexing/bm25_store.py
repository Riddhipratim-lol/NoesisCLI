"""
Lexical BM25 Indexer — Phase 3.1
Tokenizes code chunks and builds/serializes a BM25 keyword search index.

The index is stored at `.noesis/bm25.pkl` inside the analyzed repository.
"""

import os
import re
import pickle
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """
    Tokenize a code string into a list of lowercase tokens.

    Splits on non-alphanumeric boundaries and also on camelCase / snake_case
    boundaries to maximize recall for code-specific vocabulary.

    Examples:
        "UserService.authenticate()" -> ["userservice", "authenticate"]
        "verify_token" -> ["verify", "token"]
        "MAX_RETRIES" -> ["max", "retries"]
    """
    # Split on camelCase boundaries (e.g. "getUserId" -> "get User Id")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Lowercase and split on any non-alphanumeric run
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    # Filter out empty tokens and pure numeric tokens under length 2
    return [t for t in tokens if t and len(t) >= 2]


class BM25Store:
    """
    Manages a BM25 keyword search index over a collection of Code Chunks.

    Attributes:
        chunks: The original list of Code Chunk dicts that were indexed.
        bm25: The underlying BM25Okapi index object.
    """

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of Code Chunk dicts.

        Each chunk's ``code_content`` field is used as the document text.
        Tokenization is done via :func:`_tokenize`.

        Args:
            chunks: A list of Code Chunk dicts (each must have at minimum
                    ``code_content``).
        """
        if not chunks:
            return

        self.chunks = list(chunks)
        self._tokenized_corpus = [_tokenize(c.get("code_content", "")) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_str: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a BM25 lexical search over the indexed code chunks.

        Args:
            query_str: The raw query string (natural language or code symbol).
            top_k: Maximum number of results to return.

        Returns:
            A list of (chunk, score) tuples sorted by descending BM25 score,
            where each chunk is a Code Chunk dict with an extra ``bm25_score``
            key injected for downstream fusion.
        """
        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = _tokenize(query_str)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair every chunk with its score and pick top_k by score
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue  # Skip chunks with zero BM25 relevance
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(score)
            results.append(chunk)

        return results

    def retrieve(self, query_str: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Alias for :meth:`query` — satisfies a common retriever interface."""
        return self.query(query_str, top_k=top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the BM25 index and chunk corpus to a pickle file.

        Args:
            path: Absolute or relative file path (e.g. ``.noesis/bm25.pkl``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "chunks": self.chunks,
            "tokenized_corpus": self._tokenized_corpus,
            "bm25": self.bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "BM25Store":
        """
        Deserialize a BM25Store from a previously saved pickle file.

        Args:
            path: Path to the ``.noesis/bm25.pkl`` file.

        Returns:
            A fully initialized :class:`BM25Store` instance.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 index not found at '{path}'. "
                "Run 'noesiscli analyze <path>' first."
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)

        store = cls()
        store.chunks = payload.get("chunks", [])
        store._tokenized_corpus = payload.get("tokenized_corpus", [])
        store.bm25 = payload.get("bm25")
        return store
