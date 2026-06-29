"""
RAG Pipeline Orchestrator node.
Triggers hybrid search, fetches context from the Symbol Table & Dependency Graph,
prunes code context, constructs prompts, and runs reasoning over pruned context using Gemini 3.5 Flash.
"""
