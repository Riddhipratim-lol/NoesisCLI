import pytest
from unittest.mock import MagicMock

# Try to import, otherwise skip tests or mock imports
try:
    from noesiscli.retrieval.fusion import HybridRetriever
except ImportError:
    HybridRetriever = None

try:
    from noesiscli.retrieval.pruner import ContextPruner
except ImportError:
    ContextPruner = None


@pytest.mark.skipif(HybridRetriever is None, reason="HybridRetriever not implemented")
def test_hybrid_retriever_rrf(mock_code_chunks):
    """Test Reciprocal Rank Fusion (RRF) fusion and deduplication algorithm."""
    # Chunk 1: Vector rank 1, BM25 rank 2
    # Chunk 2: Vector rank 2, BM25 rank 1
    # We should merge them and compute RRF scores.
    
    vector_results = [mock_code_chunks[0], mock_code_chunks[1]]
    bm25_results = [mock_code_chunks[1], mock_code_chunks[0]]
    
    retriever = HybridRetriever(k=60)
    merged_results = retriever.fuse(vector_results, bm25_results, top_k=2)
    
    assert len(merged_results) == 2
    # Ensure they have a fusion score and are deduplicated
    for item in merged_results:
        assert "rrf_score" in item or "score" in item
    assert merged_results[0]["file_path"] != merged_results[1]["file_path"]


@pytest.mark.skipif(ContextPruner is None, reason="ContextPruner not implemented")
def test_context_pruner(mock_code_chunks):
    """Test that context pruner extracts target structures and removes helper bodies."""
    # Mock Symbol Table and Dependency Graph
    symbol_table = {
        "UserService": [{
            "file_path": "/mock/project/src/user_service.py",
            "node_type": "class",
            "start_line": 1,
            "end_line": 10
        }]
    }
    
    # Simple dependency graph (using networkx-like interface)
    dependency_graph = MagicMock()
    dependency_graph.successors.return_value = [] # no dependencies
    
    pruner = ContextPruner(symbol_table=symbol_table, dependency_graph=dependency_graph)
    
    # Original code block
    raw_code = """
class UserService:
    def authenticate(self):
        # target logic
        return True
        
    def secret_helper(self):
        # non-target helper
        do_heavy_lifting()
"""
    
    # We want to keep 'authenticate' but prune 'secret_helper'
    target_symbols = ["UserService.authenticate"]
    pruned_code = pruner.prune_file(
        file_path="/mock/project/src/user_service.py",
        file_content=raw_code,
        target_symbols=target_symbols
    )
    
    # Pruned code should contain authenticate's body, but secret_helper's body should be pruned to ...
    assert "def authenticate" in pruned_code
    assert "return True" in pruned_code
    assert "def secret_helper" in pruned_code
    assert "do_heavy_lifting" not in pruned_code or "..." in pruned_code
