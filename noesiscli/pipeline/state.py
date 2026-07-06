"""
Defines the State graph object for the LangGraph workflow.
Tracks properties such as:
  - query: Raw user prompt
  - is_valid: Boolean status from the Query Validation Layer
  - route: Selected route ('direct_llm' or 'repository_rag')
  - context_chunks: Retrieved code chunks / pruned context
  - response: Streamed/final response text
"""

from typing import TypedDict, List, Dict, Any, Optional

class WorkflowState(TypedDict):
    query: str
    is_valid: bool
    route: str
    context_chunks: List[Dict[str, Any]]
    response: str
    feedback: Optional[str]
