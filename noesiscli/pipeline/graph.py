"""
Compiles and defines the workflow graph using LangGraph.
Hooks up:
  - LangGraph Conditional Router
  - Direct LLM Node
  - Repository RAG Node (backed by HybridRetriever when available)
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

    def __init__(self, llm_client=None, retriever=None):
        self.llm_client = llm_client or GeminiClient()
        self.retriever = retriever

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
        return {"response": "".join(full_response), "context_chunks": chunks}

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
