"""
Defines the State graph object for the LangGraph workflow.
Tracks properties such as:
  - query: Raw user prompt
  - is_valid: Boolean status from the Query Validation Layer
  - route: Selected route ('direct_llm' or 'repository_rag')
  - context_chunks: Retrieved code chunks / pruned context
  - response: Streamed/final response text
  - symbol_table: Global Symbol Table (Phase 4.1), available for pruning (Phase 6)
  - dep_graph: Codebase Dependency Graph (Phase 4.2), available for pruning (Phase 6)
"""

from typing import TypedDict, List, Dict, Any, Optional

class WorkflowState(TypedDict, total=False):
    query: str
    route: str
    context_chunks: List[Dict[str, Any]]
    response: str
    # Phase 4 relational structures — optional (absent for `ask` path)
    symbol_table: Optional[Any]   # noesiscli.parser.symbol_table.SymbolTable
    dep_graph: Optional[Any]      # noesiscli.parser.dependency_graph.DependencyGraph
