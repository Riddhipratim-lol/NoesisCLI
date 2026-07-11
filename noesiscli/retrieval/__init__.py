"""
Hybrid retrieval, context-aware structure pruning, and prompt construction for NoesisCLI.

Phases:
  Phase 3.2 — HybridRetriever  (fusion.py)
  Phase 6.1 — DependencyContextResolver  (pruner.py)
  Phase 6.2 — CodeStructurePruner        (pruner.py)
  Phase 6.3 — PromptConstructor          (pruner.py)
"""

from noesiscli.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion
from noesiscli.retrieval.pruner import (
    DependencyContextResolver,
    CodeStructurePruner,
    PromptConstructor,
    PrunedBlock,
    build_pruned_prompt,
)

__all__ = [
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "DependencyContextResolver",
    "CodeStructurePruner",
    "PromptConstructor",
    "PrunedBlock",
    "build_pruned_prompt",
]
