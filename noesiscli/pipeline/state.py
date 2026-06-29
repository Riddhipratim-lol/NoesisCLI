"""
Defines the State graph object for the LangGraph workflow.
Tracks properties such as:
  - query: Raw user prompt
  - is_valid: Boolean status from the Query Validation Layer
  - route: Selected route ('direct_llm' or 'repository_rag')
  - context_chunks: Retrieved code chunks / pruned context
  - response: Streamed/final response text
"""
