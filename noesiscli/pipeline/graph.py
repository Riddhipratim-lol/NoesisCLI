"""
Compiles and defines the workflow graph using LangGraph.

Hooks up:
  - LangGraph Conditional Router
  - Direct LLM Node (Phase 2.4)
  - Repository RAG Node (Phase 3.2 + Phase 6.1–6.3)
  - Phase 4 structures (SymbolTable, DependencyGraph) threaded through
    WorkflowState and into RAGNode for Phase 6 context pruning.
  - Phase 7.3 UI: both nodes use stream_response() for live Markdown
    rendering of streamed LLM tokens.
"""

import sys
from langgraph.graph import StateGraph, START, END
from noesiscli.pipeline.state import WorkflowState
from noesiscli.pipeline.direct import DirectResponder
from noesiscli.pipeline.rag import RAGNode
from noesiscli.models.client import GeminiClient
from noesiscli.utils.ui import stream_response


def check_route(state: WorkflowState) -> str:
    if state.get("route") == "repository_rag":
        return "rag_node"
    return "direct_llm_node"


class WorkflowGraph:
    """
    Assembles and compiles the LangGraph execution graph.

    Args:
        llm_client:   A :class:`~noesiscli.models.client.GeminiClient` instance.
        retriever:    Any object exposing a ``retrieve(query_str)`` method.
                      Typically a :class:`~noesiscli.retrieval.fusion.HybridRetriever`.
        symbol_table: Loaded SymbolTable (Phase 4.1), forwarded to the RAG
                      node for Phase 6 context pruning.
        dep_graph:    Loaded DependencyGraph (Phase 4.2), forwarded to the RAG
                      node for Phase 6 context pruning.
    """

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        symbol_table=None,
        dep_graph=None,
    ) -> None:
        self.llm_client = llm_client or GeminiClient()
        self.retriever = retriever
        # Phase 4 relational structures — consumed by Phase 6 inside RAGNode
        self.symbol_table = symbol_table
        self.dep_graph = dep_graph

        from noesiscli.config import GEMINI_3_1_FLASH_LITE
        direct_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)

        self.direct_responder = DirectResponder(llm_client=direct_client)

        # Pass symbol_table and dep_graph into RAGNode so Phase 6 can run
        self.rag_node = RAGNode(
            llm_client=self.llm_client,
            retriever=self.retriever,
            symbol_table=self.symbol_table,
            dep_graph=self.dep_graph,
        )

    # ------------------------------------------------------------------
    # Node Handlers
    # ------------------------------------------------------------------

    def direct_llm_node(self, state: WorkflowState) -> dict:
        """Execute the Direct LLM path and stream the response with Markdown rendering."""
        response_stream = self.direct_responder.execute(state["query"])
        full_response = stream_response(response_stream, title="Response")
        return {"response": full_response}

    def rag_node_node(self, state: WorkflowState) -> dict:
        """
        Execute the Repository RAG path.

        Steps:
        1. Retrieve chunks via the configured retriever (HybridRetriever).
        2. Print each retrieved chunk summary to stdout.
        3. Run Phase 6 pruning + prompt construction inside RAGNode.execute().
        4. Stream the reasoning response from the LLM.
        """
        chunks = []
        if self.retriever is not None:
            chunks = self.retriever.retrieve(state["query"])

        if chunks:
            print(f"\nRetrieved {len(chunks)} relevant chunk(s):")
            for idx, res in enumerate(chunks):
                rrf = res.get("rrf_score")
                score_str = f"  [RRF: {rrf:.4f}]" if rrf is not None else ""
                print(
                    f"\n[{idx + 1}] {res.get('file_path')} "
                    f"(Lines {res.get('start_line')}-{res.get('end_line')})"
                    f"{score_str}"
                )
                print("-" * 40)
                print(res.get("code_content"))
                print("-" * 40)

        # Log Phase 6 status
        phase6_active = self.symbol_table is not None or self.dep_graph is not None
        if phase6_active:
            st_info = (
                f"SymbolTable({len(self.symbol_table)} defs)"
                if self.symbol_table else "no SymbolTable"
            )
            dg_info = (
                f"DependencyGraph({self.dep_graph.node_count()} nodes)"
                if self.dep_graph else "no DependencyGraph"
            )
            print(f"\n[Phase 6] Context pruning active — {st_info}, {dg_info}")
        else:
            print("\n[Phase 6] Context pruning unavailable — using plain context.")

        print("\nReasoning over retrieved context...")

        # Pre-populate last_chunks so RAGNode doesn't re-retrieve
        self.rag_node.last_chunks = chunks
        response_stream = self.rag_node.execute(state["query"])
        full_response = stream_response(response_stream, title="Repository Analysis")
        return {
            "response": full_response,
            "context_chunks": chunks,
            # Propagate Phase 4 structures so they remain available in state
            "symbol_table": self.symbol_table,
            "dep_graph": self.dep_graph,
        }

    # ------------------------------------------------------------------
    # Graph Compilation
    # ------------------------------------------------------------------

    def compile(self):
        """Build and compile the LangGraph StateGraph."""
        builder = StateGraph(WorkflowState)

        # Register nodes
        builder.add_node("direct_llm_node", self.direct_llm_node)
        builder.add_node("rag_node", self.rag_node_node)

        # Conditional entry-point routing
        builder.add_conditional_edges(
            START,
            check_route,
            {
                "rag_node": "rag_node",
                "direct_llm_node": "direct_llm_node",
            },
        )

        # Terminal edges
        builder.add_edge("direct_llm_node", END)
        builder.add_edge("rag_node", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Phase 4 structure accessors (available for external inspection)
    # ------------------------------------------------------------------

    def get_symbol_table(self):
        """Return the loaded SymbolTable, or None if not available."""
        return self.symbol_table

    def get_dep_graph(self):
        """Return the loaded DependencyGraph, or None if not available."""
        return self.dep_graph
