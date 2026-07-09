"""
CLI Entrypoint for NoesisCLI.
Defines argparse commands:
  - analyze: Ingest and index a local repository
             (ChromaDB + BM25 + SymbolTable + DependencyGraph).
  - query:   Run a RAG query against the codebase using HybridRetriever,
             with Symbol Table and Dependency Graph loaded for Phase 6 pruning.
  - ask:     Ask a general programming question directly to the LLM.
"""

import argparse
import sys
import os

from noesiscli.parser.scanner import RepositoryScanner
from noesiscli.parser.tree_sitter_parser import TreeSitterParser
from noesiscli.parser.symbol_table import SymbolTable
from noesiscli.parser.dependency_graph import DependencyGraph
from noesiscli.indexing.embedding import EmbeddingGenerator
from noesiscli.indexing.vector_store import ChromaVectorStore
from noesiscli.indexing.bm25_store import BM25Store


def _noesis_dir(repo_path: str) -> str:
    """Return the absolute path of the .noesis/ directory for a repository."""
    return os.path.join(os.path.abspath(repo_path), ".noesis")


def _chroma_path(repo_path: str) -> str:
    return os.path.join(_noesis_dir(repo_path), "chroma")


def _bm25_path(repo_path: str) -> str:
    return os.path.join(_noesis_dir(repo_path), "bm25.pkl")


def _symbol_table_path(repo_path: str) -> str:
    return os.path.join(_noesis_dir(repo_path), "symbol_table.pkl")


def _dep_graph_path(repo_path: str) -> str:
    return os.path.join(_noesis_dir(repo_path), "dependency_graph.pkl")


def main():
    parser = argparse.ArgumentParser(description="NoesisCLI — Local Codebase Architect")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------ #
    # Subcommand: analyze                                                  #
    # ------------------------------------------------------------------ #
    analyze_parser = subparsers.add_parser(
        "analyze", help="Ingest and index a local repository"
    )
    analyze_parser.add_argument(
        "repo_path", type=str, help="Path to the local repository"
    )
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index the repository even if an existing index is found",
    )

    # ------------------------------------------------------------------ #
    # Subcommand: query                                                    #
    # ------------------------------------------------------------------ #
    query_parser = subparsers.add_parser(
        "query", help="Run a RAG query against the codebase context"
    )
    query_parser.add_argument("prompt", type=str, help="The query/prompt to execute")
    query_parser.add_argument(
        "--repo-path", "-p",
        type=str,
        default=None,
        help="Path to the indexed repository (defaults to current directory)",
    )

    # ------------------------------------------------------------------ #
    # Subcommand: ask                                                      #
    # ------------------------------------------------------------------ #
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a general programming question directly to the LLM"
    )
    ask_parser.add_argument("prompt", type=str, help="The query/prompt to execute")

    args = parser.parse_args()

    # ================================================================== #
    # ANALYZE                                                              #
    # ================================================================== #
    if args.command == "analyze":
        repo_path = os.path.abspath(args.repo_path)
        if not os.path.isdir(repo_path):
            print(
                f"Error: Directory '{repo_path}' does not exist.", file=sys.stderr
            )
            sys.exit(1)

        # Check for existing index (skip re-indexing unless --force)
        chroma_dir = _chroma_path(repo_path)
        bm25_file = _bm25_path(repo_path)
        sym_table_file = _symbol_table_path(repo_path)
        dep_graph_file = _dep_graph_path(repo_path)
        if (
            os.path.isdir(chroma_dir)
            and os.path.isfile(bm25_file)
            and os.path.isfile(sym_table_file)
            and os.path.isfile(dep_graph_file)
            and not args.force
        ):
            print(
                f"Index already exists at '{_noesis_dir(repo_path)}'.\n"
                "Use '--force' to re-index."
            )
            return

        # 1. Scan repository for source files
        scanner = RepositoryScanner()
        files = scanner.scan(repo_path)
        print(f"Found {len(files)} source files in {repo_path}:")
        for f in files:
            print(f"  {f}")

        # 2. Filter Python files and parse with Tree-sitter
        python_files = [f for f in files if f.endswith(".py")]
        tree_parser = TreeSitterParser(language="python")
        all_chunks = []
        for f in python_files:
            chunks = tree_parser.parse_file(f)
            all_chunks.extend(chunks)

        print(
            f"\nParsed {len(python_files)} Python file(s) into "
            f"{len(all_chunks)} semantic chunk(s)."
        )

        if not all_chunks:
            print("No chunks produced — nothing to index.")
            return all_chunks

        # 3. Generate embeddings (Voyage AI)
        print("\nGenerating embeddings using Voyage AI...")
        embed_gen = EmbeddingGenerator()
        embeddings = embed_gen.embed_chunks(all_chunks)

        # 4. Persist to ChromaDB
        print("Indexing chunks into ChromaDB...")
        os.makedirs(os.path.dirname(chroma_dir), exist_ok=True)
        vector_store = ChromaVectorStore(persist_directory=chroma_dir)
        vector_store.add_chunks(all_chunks, embeddings)
        print(f"  → ChromaDB index saved to '{chroma_dir}'")

        # 5. Build and persist the BM25 lexical index  (Phase 3.1)
        print("Building BM25 lexical index...")
        bm25_store = BM25Store()
        bm25_store.build(all_chunks)
        bm25_store.save(bm25_file)
        print(f"  → BM25 index saved to '{bm25_file}'")

        # 6. Build and persist the Global Symbol Table  (Phase 4.1)
        print("\nBuilding Global Symbol Table...")
        sym_table = SymbolTable()
        sym_table.build(all_chunks)
        sym_table.save(sym_table_file)
        print(
            f"  → Symbol Table saved to '{sym_table_file}' "
            f"({len(sym_table)} definitions across "
            f"{len(sym_table.all_names())} unique names)"
        )

        # 7. Build and persist the Codebase Dependency Graph  (Phase 4.2)
        print("Building Codebase Dependency Graph...")
        dep_graph = DependencyGraph()
        dep_graph.build(all_chunks, sym_table)
        dep_graph.save(dep_graph_file)
        print(
            f"  → Dependency Graph saved to '{dep_graph_file}' "
            f"({dep_graph.node_count()} nodes, {dep_graph.edge_count()} edges)"
        )

        print("\nIndexing completed successfully.")
        return all_chunks

    # ================================================================== #
    # QUERY / ASK                                                          #
    # ================================================================== #
    elif args.command in ("query", "ask"):
        prompt = args.prompt

        from noesiscli.models.client import GeminiClient
        from noesiscli.pipeline.graph import WorkflowGraph

        client = GeminiClient()

        if args.command == "query":
            # Resolve the repository path: use --repo-path if given, else cwd
            repo_root = os.path.abspath(args.repo_path) if args.repo_path else os.getcwd()
            chroma_dir = os.path.join(repo_root, ".noesis", "chroma")
            bm25_file = os.path.join(repo_root, ".noesis", "bm25.pkl")

            if not os.path.isdir(chroma_dir):
                print(
                    "Error: Noesis index not found. "
                    "Please run 'noesiscli analyze <path>' first.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Load dense vector store
            vector_store = ChromaVectorStore(persist_directory=chroma_dir)

            # Load BM25 store if available (Phase 3.1)
            bm25_store = None
            if os.path.isfile(bm25_file):
                try:
                    bm25_store = BM25Store.load(bm25_file)
                    print(
                        f"[HybridRetriever] Loaded BM25 index "
                        f"({len(bm25_store.chunks)} chunks)."
                    )
                except Exception as exc:
                    print(
                        f"[Warning] Could not load BM25 index: {exc}. "
                        "Falling back to dense-only retrieval.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "[Warning] BM25 index not found — using dense-only retrieval. "
                    "Re-run 'noesiscli analyze' to build the full hybrid index.",
                    file=sys.stderr,
                )

            # Build HybridRetriever (Phase 3.2)
            from noesiscli.retrieval.fusion import HybridRetriever

            retriever = HybridRetriever(
                vector_store=vector_store,
                bm25_store=bm25_store,
                top_k=5,
            )

            # Load Symbol Table (Phase 4.1)
            sym_table_file = os.path.join(repo_root, ".noesis", "symbol_table.pkl")
            symbol_table = None
            if os.path.isfile(sym_table_file):
                try:
                    symbol_table = SymbolTable.load(sym_table_file)
                    print(
                        f"[Phase 4] Loaded Symbol Table "
                        f"({len(symbol_table)} definitions, "
                        f"{len(symbol_table.all_names())} unique names)."
                    )
                except Exception as exc:
                    print(
                        f"[Warning] Could not load Symbol Table: {exc}. "
                        "Proceeding without symbol resolution.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "[Warning] Symbol Table not found — run 'noesiscli analyze' "
                    "to enable Phase 6 context pruning.",
                    file=sys.stderr,
                )

            # Load Dependency Graph (Phase 4.2)
            dep_graph_file = os.path.join(repo_root, ".noesis", "dependency_graph.pkl")
            dep_graph = None
            if os.path.isfile(dep_graph_file):
                try:
                    dep_graph = DependencyGraph.load(dep_graph_file)
                    print(
                        f"[Phase 4] Loaded Dependency Graph "
                        f"({dep_graph.node_count()} nodes, "
                        f"{dep_graph.edge_count()} edges)."
                    )
                except Exception as exc:
                    print(
                        f"[Warning] Could not load Dependency Graph: {exc}. "
                        "Proceeding without dependency resolution.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "[Warning] Dependency Graph not found — run 'noesiscli analyze' "
                    "to enable Phase 6 context pruning.",
                    file=sys.stderr,
                )

            wf_graph = WorkflowGraph(
                llm_client=client,
                retriever=retriever,
                symbol_table=symbol_table,
                dep_graph=dep_graph,
            )
            route = "repository_rag"

        else:
            # ask command — no retrieval needed
            wf_graph = WorkflowGraph(llm_client=client, retriever=None)
            route = "direct_llm"

        graph = wf_graph.compile()

        initial_state = {
            "query": prompt,
            "route": route,
            "context_chunks": [],
            "response": "",
            "symbol_table": None,
            "dep_graph": None,
        }

        final_state = graph.invoke(initial_state)
        return final_state.get("context_chunks", [])

    # ================================================================== #
    # Fallback — print help                                                #
    # ================================================================== #
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
