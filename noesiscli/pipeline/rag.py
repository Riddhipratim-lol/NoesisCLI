"""
Repository RAG Node — Phase 6 Integration.

Retrieves context code chunks via the HybridRetriever (Phase 3.2), then
routes through the Context-Aware Pruning & Prompt Construction pipeline
(Phase 6.1 → 6.2 → 6.3) before streaming the reasoned response via the
Gemini LLM client (Phase 7.1).

Data flow:
  query
    → HybridRetriever.retrieve()          (Phase 3.2)
    → build_pruned_prompt()               (Phase 6.1–6.3)
    → GeminiClient.stream()               (Phase 7.1)
    → streamed token generator
"""

from typing import Generator, List, Dict, Any, Optional

from noesiscli.models.client import GeminiClient
from noesiscli.retrieval.pruner import build_pruned_prompt, PromptConstructor


class RAGNode:
    """
    Repository RAG reasoning node.

    Integrates retrieval (Phase 3.2), context-aware pruning (Phase 6.1–6.3),
    and LLM streaming (Phase 7.1) into a single execution step.

    Args:
        llm_client:   :class:`~noesiscli.models.client.GeminiClient` for
                      generating the streamed response.
        retriever:    Any object exposing ``retrieve(query_str) -> list``.
                      Typically :class:`~noesiscli.retrieval.fusion.HybridRetriever`.
        symbol_table: Optional loaded SymbolTable for Phase 6 pruning.
        dep_graph:    Optional loaded DependencyGraph for Phase 6 pruning.
    """

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        symbol_table=None,
        dep_graph=None,
    ) -> None:
        self.llm_client: GeminiClient = llm_client or GeminiClient()
        self.retriever = retriever
        self.symbol_table = symbol_table
        self.dep_graph = dep_graph

        # Pre-populated by graph.py's rag_node_node so retrieval is not
        # duplicated (the graph retrieves first, then hands chunks here).
        self.last_chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, query: str) -> Generator[str, None, None]:
        """
        Execute the RAG pipeline for *query* and yield streamed response tokens.

        If ``self.last_chunks`` has been pre-populated (by the graph node), those
        chunks are used directly.  Otherwise the retriever is called here.

        Pruning (Phase 6) is attempted when Symbol Table and/or Dependency
        Graph are available.  If they are absent the node gracefully falls back
        to plain code-block context (Phase 1.5-style prompt).
        """
        # ── 1. Retrieve if not pre-populated ─────────────────────────────
        if not self.last_chunks:
            if self.retriever is not None:
                self.last_chunks = self.retriever.retrieve(query)
            else:
                self.last_chunks = []

        # ── 2. Build pruned prompt (Phase 6) or fallback ─────────────────
        if self.last_chunks:
            pruning_available = (
                self.symbol_table is not None or self.dep_graph is not None
            )

            if pruning_available:
                prompt_str, system_instruction = build_pruned_prompt(
                    query=query,
                    retrieved_chunks=self.last_chunks,
                    symbol_table=self.symbol_table,
                    dep_graph=self.dep_graph,
                )
            else:
                # Fallback: plain context blocks (no pruning)
                prompt_str, system_instruction = self._plain_context_prompt(
                    query, self.last_chunks
                )
        else:
            # No chunks retrieved — answer directly
            prompt_str = f"User Query: {query}"
            system_instruction = PromptConstructor.SYSTEM_INSTRUCTION

        return self.llm_client.stream(prompt_str, system_instruction=system_instruction)

    # ------------------------------------------------------------------
    # Fallback prompt builder (no Phase 4 structures available)
    # ------------------------------------------------------------------

    @staticmethod
    def _plain_context_prompt(
        query: str,
        chunks: List[Dict[str, Any]],
    ):
        """
        Build a basic prompt from raw code chunks (no pruning).

        Used when the Symbol Table and Dependency Graph are unavailable so
        Phase 6 pruning cannot run.  Mirrors the Phase 1.5 approach.
        """
        context_parts: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            file_path = chunk.get("file_path", "unknown")
            start_line = chunk.get("start_line", 0)
            end_line = chunk.get("end_line", 0)
            node_type = chunk.get("node_type", "unknown")
            code_content = chunk.get("code_content", "")

            context_parts.append(
                f"[{idx}] File: {file_path} "
                f"(Lines {start_line}–{end_line}, Type: {node_type})\n"
                f"```python\n{code_content}\n```"
            )

        context_str = "\n\n".join(context_parts)
        system_instruction = PromptConstructor.SYSTEM_INSTRUCTION
        prompt_str = (
            f"Code Context:\n{context_str}\n\nUser Query: {query}"
            if context_str
            else f"User Query: {query}"
        )
        return prompt_str, system_instruction
