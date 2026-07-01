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

def main():
    parser = argparse.ArgumentParser(description="NoesisCLI — Local Codebase Architect")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Ingest and index a local repository")
    analyze_parser.add_argument("repo_path", type=str, help="Path to the local repository")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query against the codebase or general LLM")
    query_parser.add_argument("prompt", type=str, help="The query/prompt to execute")

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
        for chunk in all_chunks:
            print(f" - [{chunk['node_type'].upper()}] {chunk['file_path']} (Lines {chunk['start_line']}-{chunk['end_line']})")
            
        return all_chunks
        
    elif args.command == "query":
        print(f"Querying: '{args.prompt}' (RAG query execution placeholder)")
        return []
        
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
