"""
Hybrid Retriever with Reciprocal Rank Fusion — Phase 3.2

Executes semantic search (ChromaDB) and lexical search (BM25) concurrently,
merges the result lists using Reciprocal Rank Fusion (RRF), de-duplicates
overlapping chunks, and returns a single ranked list of Code Chunk dicts.

RRF formula (per document d across all retrievers M):
    RRF(d) = Σ_{m ∈ M}  1 / (k + r_m(d))

where r_m(d) is the 1-indexed rank of document d in retriever m and k=60
is the standard constant that dampens the impact of top-ranked documents.
"""

from __future__ import annotations

import concurrent.futures
from typing import List, Dict, Any, Optional

# Reciprocal Rank Fusion constant (industry default)
_RRF_K = 60


def _chunk_key(chunk: Dict[str, Any]) -> str:
    """
    Build a deduplication key for a code chunk.

    Chunks from ChromaDB and BM25 that refer to the same code region share
    an identical key and are merged during fusion rather than returned twice.
    """
    file_path = chunk.get("file_path", "")
    start_line = chunk.get("start_line", 0)
    end_line = chunk.get("end_line", 0)
    node_type = chunk.get("node_type", "")
    return f"{file_path}:{start_line}:{end_line}:{node_type}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = _RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: A list of already-ranked result lists (each list is
                      ordered from most to least relevant).  Each element is
                      a Code Chunk dict.
        k: The RRF constant (default 60).

    Returns:
        A single deduplicated list of Code Chunk dicts sorted by descending
        RRF score.  The winning chunk dict is the one from whichever retriever
        first introduced the key (earlier retriever wins on tie).  An extra
        ``rrf_score`` key is injected for observability.
    """
    scores: Dict[str, float] = {}
    # Store the "best" (first-seen) chunk for each key
    canonical: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in canonical:
                canonical[key] = chunk

    # Sort by descending RRF score and inject the score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    result = []
    for key in sorted_keys:
        chunk = dict(canonical[key])
        chunk["rrf_score"] = round(scores[key], 6)
        result.append(chunk)
    return result


class HybridRetriever:
    """
    Hybrid Retriever that combines ChromaDB dense search with BM25 lexical
    search and merges results via Reciprocal Rank Fusion (RRF).

    Args:
        vector_store: A :class:`~noesiscli.indexing.vector_store.ChromaVectorStore`
                      instance (must expose a ``query(query_str, top_k)`` method).
        bm25_store:   A :class:`~noesiscli.indexing.bm25_store.BM25Store`
                      instance (must expose a ``query(query_str, top_k)`` method).
        top_k:        Number of candidates to request from each retriever before
                      fusion.  The final merged list may be shorter because
                      zero-score BM25 results are dropped.
    """

    def __init__(
        self,
        vector_store=None,
        bm25_store=None,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def retrieve(self, query_str: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval for a given query string.

        Both the dense vector search and the BM25 lexical search are launched
        concurrently in a thread pool so they execute in parallel.  Their
        ranked result lists are then fused via :func:`reciprocal_rank_fusion`.

        Args:
            query_str: The user's raw query (natural language or code symbol).
            top_k:     Overrides the instance-level ``top_k`` for this call.

        Returns:
            A merged, deduplicated, and RRF-ranked list of Code Chunk dicts.
        """
        effective_top_k = top_k if top_k is not None else self.top_k

        dense_results: List[Dict[str, Any]] = []
        lexical_results: List[Dict[str, Any]] = []

        # Run both searches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}

            if self.vector_store is not None:
                futures["dense"] = pool.submit(
                    self.vector_store.query, query_str, effective_top_k
                )

            if self.bm25_store is not None:
                futures["lexical"] = pool.submit(
                    self.bm25_store.query, query_str, effective_top_k
                )

            for label, future in futures.items():
                try:
                    result = future.result()
                    if label == "dense":
                        dense_results = result
                    else:
                        lexical_results = result
                except Exception as exc:  # pragma: no cover
                    # Degrade gracefully: log the failure and continue
                    print(
                        f"[HybridRetriever] {label} search raised an error: {exc}"
                    )

        # Build list of ranked lists to fuse (only include non-empty ones)
        ranked_lists = []
        if dense_results:
            ranked_lists.append(dense_results)
        if lexical_results:
            ranked_lists.append(lexical_results)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists)

        # Return at most effective_top_k results after fusion
        return fused[:effective_top_k]
