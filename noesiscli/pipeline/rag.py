"""
Repository RAG Node.
Retrieves context code chunks, constructs a prompt, and streams the reasoned response
using Gemini 3.5 Flash (with fallback to Gemini 3.1 Flash-Lite).
"""

from noesiscli.models.client import GeminiClient
from typing import Generator, List, Dict, Any

class RAGNode:
    def __init__(self, llm_client=None, retriever=None):
        self.llm_client = llm_client or GeminiClient()
        self.retriever = retriever
        self.last_chunks: List[Dict[str, Any]] = []

    def execute(self, query: str) -> Generator[str, None, None]:
        if not self.last_chunks:
            if self.retriever is not None:
                # By default retrieve top 3 chunks (or let retrieval define it)
                self.last_chunks = self.retriever.retrieve(query)
            else:
                self.last_chunks = []
            
        context_parts = []
        for idx, chunk in enumerate(self.last_chunks):
            file_path = chunk.get("file_path", "unknown")
            start_line = chunk.get("start_line", 0)
            end_line = chunk.get("end_line", 0)
            node_type = chunk.get("node_type", "unknown")
            code_content = chunk.get("code_content", "")
            
            context_part = (
                f"File: {file_path} (Lines {start_line}-{end_line}, Type: {node_type})\n"
                f"```python\n{code_content}\n```"
            )
            context_parts.append(context_part)
            
        context_str = "\n\n".join(context_parts)
        
        system_instruction = (
            "You are NoesisCLI, a professional AI coding assistant and codebase architect.\n"
            "Your task is to answer the user's programming questions or repository analysis queries using the provided code context.\n"
            "Base your answer on the provided code context. If the context does not contain the answer, "
            "provide the best answer possible while indicating the limitations of the provided context."
        )
        
        llm_prompt = f"Code Context:\n{context_str}\n\nUser Query: {query}" if context_str else f"User Query: {query}"
        
        return self.llm_client.stream(llm_prompt, system_instruction=system_instruction)
