"""
Compiles and defines the workflow graph using LangGraph.
Hooks up:
  - LangGraph Conditional Router
  - Direct LLM Node
  - Repository RAG Node (backed by HybridRetriever when available)
  - Phase 4 structures (SymbolTable, DependencyGraph) threaded through
    WorkflowState for consumption by Phase 6 (Context Pruner).
"""

import sys
from langgraph.graph import StateGraph, START, END
from noesiscli.pipeline.state import WorkflowState
from noesiscli.pipeline.direct import DirectResponder
from noesiscli.pipeline.rag import RAGNode
from noesiscli.models.client import GeminiClient


def check_route(state: WorkflowState) -> str:
    if state.get("route") == "repository_rag":
        return "rag_node"
    return "direct_llm_node"


class WorkflowGraph:
    """
    Assembles and compiles the LangGraph execution graph.

    Args:
        llm_client: A :class:`~noesiscli.models.client.GeminiClient` instance
                    used for reasoning in the RAG path.
        retriever:  Any object exposing a ``retrieve(query_str, top_k)`` method.
                    This should be a :class:`~noesiscli.retrieval.fusion.HybridRetriever`
                    (Phase 3) or the plain ``ChromaVectorStore`` (Phase 1 fallback).
    """

    def __init__(self, llm_client=None, retriever=None, symbol_table=None, dep_graph=None):
        self.llm_client = llm_client or GeminiClient()
        self.retriever = retriever
        # Phase 4 relational structures — passed through to state for Phase 6
        self.symbol_table = symbol_table
        self.dep_graph = dep_graph

        from noesiscli.config import GEMINI_3_1_FLASH_LITE
        direct_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)

        self.direct_responder = DirectResponder(llm_client=direct_client)
        self.rag_node = RAGNode(llm_client=self.llm_client, retriever=self.retriever)

    # ------------------------------------------------------------------
    # Node Handlers
    # ------------------------------------------------------------------

    def direct_llm_node(self, state: WorkflowState) -> dict:
        """Execute the Direct LLM path and stream the response to stdout."""
        print("\nResponse:")
        response_stream = self.direct_responder.execute(state["query"])
        full_response = []
        for token in response_stream:
            sys.stdout.write(token)
            sys.stdout.flush()
            full_response.append(token)
        print()
        return {"response": "".join(full_response)}

    def rag_node_node(self, state: WorkflowState) -> dict:
        """
        Execute the Repository RAG path.

        1. Retrieve chunks via the configured retriever (HybridRetriever or
           plain ChromaVectorStore — both expose ``retrieve()``).
        2. Print each retrieved chunk summary to stdout.
        3. Pass the chunks to RAGNode and stream the reasoning response.
        """
        chunks = []
        if self.retriever is not None:
            chunks = self.retriever.retrieve(state["query"])

        if chunks:
            print(f"\nRetrieved {len(chunks)} relevant chunks:")
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

            print("\nReasoning over retrieved context...")

        print("\nResponse:")

        # Populate rag_node.last_chunks so that RAGNode.execute doesn't re-retrieve
        self.rag_node.last_chunks = chunks
        response_stream = self.rag_node.execute(state["query"])
        full_response = []
        for token in response_stream:
            sys.stdout.write(token)
            sys.stdout.flush()
            full_response.append(token)
        print()
        return {
            "response": "".join(full_response),
            "context_chunks": chunks,
            # Propagate Phase 4 structures so they remain in state for Phase 6
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
    # Phase 4 structure accessors (consumed by Phase 6)
    # ------------------------------------------------------------------

    def get_symbol_table(self):
        """Return the loaded SymbolTable, or None if not available."""
        return self.symbol_table

    def get_dep_graph(self):
        """Return the loaded DependencyGraph, or None if not available."""
        return self.dep_graph
