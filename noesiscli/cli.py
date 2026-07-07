"""
CLI Entrypoint for NoesisCLI.
Defines argparse commands:
  - analyze: Ingest and index a local repository.
  - query: Run a single query against the codebase or general LLM.
  - chat: Enter an interactive session.
"""

import argparse
import sys
import os
from noesiscli.parser.scanner import RepositoryScanner
from noesiscli.parser.tree_sitter_parser import TreeSitterParser
from noesiscli.indexing.embedding import EmbeddingGenerator
from noesiscli.indexing.vector_store import ChromaVectorStore

def main():
    parser = argparse.ArgumentParser(description="NoesisCLI — Local Codebase Architect")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Ingest and index a local repository")
    analyze_parser.add_argument("repo_path", type=str, help="Path to the local repository")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a RAG query against the codebase context")
    query_parser.add_argument("prompt", type=str, help="The query/prompt to execute")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a general programming question directly to the LLM")
    ask_parser.add_argument("prompt", type=str, help="The query/prompt to execute")

    args = parser.parse_args()

    if args.command == "analyze":
        repo_path = args.repo_path
        if not os.path.isdir(repo_path):
            print(f"Error: Directory '{repo_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        
        scanner = RepositoryScanner()
        files = scanner.scan(repo_path)
        print(f"Found {len(files)} source files in {os.path.abspath(repo_path)}:")
        for f in files:
            print(f)
        
        # Filter python files for AST parsing
        python_files = [f for f in files if f.endswith(".py")]
        
        # Parse files using TreeSitterParser
        parser = TreeSitterParser(language="python")
        all_chunks = []
        for f in python_files:
            chunks = parser.parse_file(f)
            all_chunks.extend(chunks)
            
        print(f"\nParsed {len(python_files)} Python files into {len(all_chunks)} semantic chunks.")
        
        if all_chunks:
            # Generate embeddings
            print("Generating embeddings using Voyage AI...")
            embed_gen = EmbeddingGenerator()
            embeddings = embed_gen.embed_chunks(all_chunks)
            
            # Save to ChromaDB
            print("Indexing chunks into ChromaDB...")
            db_path = os.path.join(repo_path, ".noesis", "chroma")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            vector_store = ChromaVectorStore(persist_directory=db_path)
            vector_store.add_chunks(all_chunks, embeddings)
            print("Indexing completed successfully.")
            
        return all_chunks
        
    elif args.command in ("query", "ask"):
        prompt = args.prompt
        
        from noesiscli.models.client import GeminiClient
        from noesiscli.pipeline.graph import WorkflowGraph
        
        client = GeminiClient()
        
        if args.command == "query":
            # Search for .noesis in current dir or parents
            cwd = os.getcwd()
            db_path = os.path.join(cwd, ".noesis", "chroma")
            if not os.path.isdir(db_path):
                print("Error: Noesis index not found. Please run 'noesiscli analyze <path>' first in this directory.", file=sys.stderr)
                sys.exit(1)
                
            vector_store = ChromaVectorStore(persist_directory=db_path)
            wf_graph = WorkflowGraph(llm_client=client, retriever=vector_store)
            route = "repository_rag"
        else:
            # ask command
            wf_graph = WorkflowGraph(llm_client=client, retriever=None)
            route = "direct_llm"
            
        graph = wf_graph.compile()
        
        initial_state = {
            "query": prompt,
            "route": route,
            "context_chunks": [],
            "response": ""
        }
        
        final_state = graph.invoke(initial_state)
        return final_state.get("context_chunks", [])
        
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
