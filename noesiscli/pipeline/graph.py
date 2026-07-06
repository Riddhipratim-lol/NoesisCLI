"""
Compiles and defines the workflow graph using LangGraph.
Hooks up:
  - Query Validation Node
  - Intelligent Query Router Node
  - Direct LLM Node
  - Repository RAG Node
"""

import sys
from langgraph.graph import StateGraph, START, END
from noesiscli.pipeline.state import WorkflowState
from noesiscli.pipeline.validation import QueryValidator
from noesiscli.pipeline.router import QueryRouter
from noesiscli.pipeline.direct import DirectResponder
from noesiscli.pipeline.rag import RAGNode
from noesiscli.models.client import GeminiClient

def check_validation(state: WorkflowState) -> str:
    if state.get("is_valid"):
        return "route_node"
    return END

def check_route(state: WorkflowState) -> str:
    if state.get("route") == "repository_rag":
        return "rag_node"
    return "direct_llm_node"

class WorkflowGraph:
    def __init__(self, llm_client=None, retriever=None):
        self.llm_client = llm_client or GeminiClient()
        self.retriever = retriever
        
        from noesiscli.config import GEMINI_3_1_FLASH_LITE
        validator_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)
        router_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)
        direct_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)

        self.validator = QueryValidator(llm_client=validator_client)
        self.router = QueryRouter(llm_client=router_client)
        self.direct_responder = DirectResponder(llm_client=direct_client)
        self.rag_node = RAGNode(llm_client=self.llm_client, retriever=self.retriever)

    def validate_node(self, state: WorkflowState) -> dict:
        # Avoid logging/printing inside validation when testing
        is_valid, route = self.validator.validate_and_route(state["query"])
        if not is_valid:
            feedback = "Please ask a programming or repository-related question. I can only assist with coding or software development tasks."
            return {"is_valid": False, "feedback": feedback, "response": feedback, "route": None}
        return {"is_valid": True, "feedback": None, "route": route}

    def route_node(self, state: WorkflowState) -> dict:
        if state.get("route"):
            return {}
        route = self.router.route(state["query"])
        return {"route": route}

    def direct_llm_node(self, state: WorkflowState) -> dict:
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
        # Retrieve chunks first so we can print them before streaming the reasoning response
        chunks = []
        if self.retriever is not None:
            chunks = self.retriever.retrieve(state["query"])
        
        if chunks:
            print(f"\nRetrieved {len(chunks)} relevant chunks:")
            for idx, res in enumerate(chunks):
                print(f"\n[{idx + 1}] {res.get('file_path')} (Lines {res.get('start_line')}-{res.get('end_line')})")
                print("-" * 40)
                print(res.get("code_content"))
                print("-" * 40)
            
            print("\nReasoning over retrieved context...")
            
        print("\nResponse:")
        
        # Populate rag_node.last_chunks so that execute doesn't retrieve again
        self.rag_node.last_chunks = chunks
        response_stream = self.rag_node.execute(state["query"])
        full_response = []
        for token in response_stream:
            sys.stdout.write(token)
            sys.stdout.flush()
            full_response.append(token)
        print()
        return {"response": "".join(full_response), "context_chunks": chunks}

    def compile(self):
        builder = StateGraph(WorkflowState)
        
        # Add nodes
        builder.add_node("validate_node", self.validate_node)
        builder.add_node("route_node", self.route_node)
        builder.add_node("direct_llm_node", self.direct_llm_node)
        builder.add_node("rag_node", self.rag_node_node)
        
        # Set entry point
        builder.add_edge(START, "validate_node")
        
        # Add conditional edges
        builder.add_conditional_edges(
            "validate_node",
            check_validation,
            {
                "route_node": "route_node",
                END: END
            }
        )
        
        builder.add_conditional_edges(
            "route_node",
            check_route,
            {
                "rag_node": "rag_node",
                "direct_llm_node": "direct_llm_node"
            }
        )
        
        # Add normal transitions to END
        builder.add_edge("direct_llm_node", END)
        builder.add_edge("rag_node", END)
        
        return builder.compile()
