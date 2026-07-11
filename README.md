<div align="center">

# NoesisCLI

**A local-first, AI-powered codebase intelligence CLI.**
Index any Python repository, then interrogate it with natural language — backed by hybrid semantic + lexical retrieval, a structured dependency graph, context-aware pruning, and streaming Gemini reasoning.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow%20Orchestration-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green?style=flat-square)
![Voyage AI](https://img.shields.io/badge/Voyage%20AI-voyage--code--3-purple?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-3.1%20Flash--Lite%20%2B%203.5%20Flash-teal?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

</div>

---

## What is NoesisCLI?

NoesisCLI turns any local Python codebase into a queryable knowledge base without sending it to the cloud. It performs deep, AST-accurate code parsing, builds a global symbol registry and a dependency graph, and combines dense vector search with BM25 keyword search — all fused together before a context-pruned prompt is streamed through Gemini. The result is precise, grounded answers directly in your terminal.

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [License](#license)

---

## Tech Stack

Every component in the table below is fully implemented in this project.

| Component | Technology | Implementation |
|---|---|---|
| Programming Language | Python 3.12+ | `pyproject.toml` |
| Workflow Orchestration | LangGraph | `pipeline/graph.py` — `StateGraph` with conditional routing |
| LLM Framework | LangChain | `models/client.py` — `ChatGoogleGenerativeAI` via `langchain-google-genai` |
| Code Parsing | Tree-sitter | `parser/tree_sitter_parser.py` — full AST chunking with 6 bug-fix strategies |
| Parallel Processing | `concurrent.futures.ProcessPoolExecutor` | `parser/parallel.py` — N-worker multiprocess parsing |
| Symbol Table | Custom Registry | `parser/symbol_table.py` — `SymbolTable` + `SymbolDefinition` dataclass |
| Dependency Graph | NetworkX `DiGraph` | `parser/dependency_graph.py` — import / inherit / call edges |
| Embedding Model | Voyage AI `voyage-code-3` | `indexing/embedding.py` — batched via `voyageai` client |
| Inference Engine | Voyage AI HTTP Client | `indexing/embedding.py` — batch size 128, `input_type=query/document` |
| Vector Database | ChromaDB (SQLite-backed) | `indexing/vector_store.py` — `PersistentClient` collection |
| Lexical Search | BM25 (`rank-bm25`) | `indexing/bm25_store.py` — camelCase/snake_case tokenizer + `BM25Okapi` |
| Hybrid Retrieval | Custom Reciprocal Rank Fusion | `retrieval/fusion.py` — `HybridRetriever` + `reciprocal_rank_fusion()` (k=60) |
| CLI Framework | `argparse` (stdlib) | `cli.py` — `analyze`, `query`, `ask` subcommands |
| LLM | Gemini 3.1 Flash-Lite (primary) + Gemini 3.5 Flash (fallback) | `models/client.py` — lazy-init + auto-fallback on error |
| Streaming | LangChain streaming callbacks | `models/client.py` — `llm.stream()` → token generator |
| Terminal UI | Rich | `utils/ui.py` — live Markdown panel, progress bars, graceful fallback |
| Context Pruning | Custom (Tree-sitter-backed) | `retrieval/pruner.py` — `DependencyContextResolver`, `CodeStructurePruner`, `PromptConstructor` |
| Persistence | Pickle + ChromaDB on-disk | `.noesis/` directory inside target repo |

---

## System Architecture

NoesisCLI is built as a layered pipeline of seven tightly-specified phases, implemented across six decoupled Python packages. Each layer has a single responsibility and passes well-defined data structures to the next.

### Layer 1 — CLI & Ingestion (`cli.py`, `parser/scanner.py`)

The entry point is a standard `argparse`-based CLI (`cli.py`) exposing three subcommands: `analyze`, `query`, and `ask`. When `analyze` is invoked, `RepositoryScanner` performs a recursive `os.walk` over the target directory, collecting all `.py` file paths while filtering out `.git`, `.venv`, `__pycache__`, `node_modules`, `build`, and `dist` directories. The result is a sorted list of absolute paths that seeds every downstream phase.

### Layer 2 — Parallel AST Parsing (`parser/parallel.py`, `parser/tree_sitter_parser.py`)

Parsing is the most CPU-intensive step and is fully parallelized. `ParallelParserPipeline` distributes the file list across a `ProcessPoolExecutor` with up to N worker processes (default: all CPU cores, configurable via `--workers`). Each worker is a module-level function `_parse_file_worker` — a design requirement for picklability across process boundaries — that constructs its own `TreeSitterParser` instance and returns a list of Code Chunk dicts for its assigned file.

`TreeSitterParser` uses the `tree-sitter-python` grammar to walk the AST of each file and emit nine distinct chunk types: `module`, `imports`, `class`, `class_header`, `function`, `method`, `constant`, `type_alias`, and `global`. Every chunk carries a consistent schema: `code_content`, `file_path`, `node_type`, `start_line`, `end_line`, and a rich `metadata` dict containing `decorators`, `is_async`, `parent_class`, `is_dunder`, `special_type`, `docstring`, and `imports_in_file`. Four explicit parsing strategies (S1–S4) and six bug fixes are applied to handle decorated definitions, import isolation, async functions, nested functions, class-body traversal, and global node classification correctly.

A notable design point is the `class_header` chunk type: for every class, a second stub chunk is emitted containing only the class signature, docstring, and method signatures (no bodies). This pre-computed stub is consumed later by the Phase 6 pruner without any additional parsing.

### Layer 3 — Indexing (`indexing/`)

The aggregated chunk list flows into three independent index builders that run sequentially:

**Voyage AI Embeddings** (`embedding.py`): The `code_content` strings of all chunks are sent to the Voyage AI `voyage-code-3` model in batches of 128 via the `voyageai` HTTP client. The model is specialized for code retrieval and produces 1536-dimensional vectors. A `progress_callback` mechanism keeps the Rich progress bar in sync with batch completion. During query time, the same client is called with `input_type="query"` instead of `"document"` to produce asymmetrically-optimized query vectors.

**ChromaDB Vector Store** (`vector_store.py`): `ChromaVectorStore` wraps a `chromadb.PersistentClient` backed by SQLite on disk at `.noesis/chroma/`. Each chunk is stored with its embedding and a flattened metadata dict (ChromaDB only accepts primitive scalar values, so nested structures like `imports_in_file` are JSON-serialized). The collection is named `noesis_code` and supports cosine similarity queries.

**BM25 Lexical Index** (`bm25_store.py`): `BM25Store` builds a `rank_bm25.BM25Okapi` index over all chunk texts. Before indexing, a custom tokenizer splits on camelCase boundaries (`getUserId` → `["get", "user", "id"]`), lowercases, splits on non-alphanumeric runs, and filters tokens shorter than 2 characters. This maximizes recall for code-specific vocabulary. The index, tokenized corpus, and original chunks are pickled to `.noesis/bm25.pkl`.

### Layer 4 — Symbol Table & Dependency Graph (`parser/symbol_table.py`, `parser/dependency_graph.py`)

These two structures provide the relational layer that elevates NoesisCLI above a plain vector search tool.

**Global Symbol Table** (`symbol_table.py`): Iterates all chunks of type `class`, `method`, or `function` and registers each into a `dict[str, list[SymbolDefinition]]`. `SymbolDefinition` is a dataclass carrying `symbol_name`, `node_type`, `file_path`, `start_line`, `end_line`, `parent_class`, `signature`, `docstring`, `is_async`, `decorators`, and `base_classes`. The table supports exact case-sensitive lookup and case-insensitive fuzzy lookup. It is pickled to `.noesis/symbol_table.pkl`.

**Codebase Dependency Graph** (`dependency_graph.py`): Constructs a `networkx.DiGraph` with three categories of directed edges. Import edges (`relation="imports"`) connect each file node to the top-level module names extracted from its `imports` chunks using regex parsing of `import X` and `from X import Y` patterns (relative imports are skipped). Inheritance edges (`relation="inherits"`) connect class names to their base class names using the `base_classes` metadata from `class` chunks. Call edges (`relation="calls"`) are built by scanning the `code_content` of every function and method chunk with a regex for token patterns matching known symbol names in the Symbol Table — this is best-effort static analysis rather than full semantic resolution. The graph is pickled to `.noesis/dependency_graph.pkl`.

### Layer 5 — LangGraph Workflow Orchestration (`pipeline/`)

All query-time execution is orchestrated by a LangGraph `StateGraph`. The `WorkflowState` is a `TypedDict` carrying `query`, `route`, `context_chunks`, `response`, `symbol_table`, and `dep_graph`. Routing is deterministic: the `check_route()` function reads the `route` field set by the CLI and immediately directs execution to one of two nodes — `direct_llm_node` or `rag_node` — without any LLM-based classification step.

The `ask` subcommand sets `route="direct_llm"` and the graph invokes `DirectResponder`, which calls `gemini-3.1-flash-lite` directly with a concise system prompt. The `query` subcommand sets `route="repository_rag"` and the graph invokes `RAGNode`, which coordinates retrieval, pruning, and LLM streaming. The SymbolTable and DependencyGraph are threaded through `WorkflowState` so they are available inside the RAG node without global state.

### Layer 6 — Hybrid Retrieval (`retrieval/fusion.py`)

`HybridRetriever` runs dense and lexical search concurrently in a `ThreadPoolExecutor` with two workers. The dense branch generates a query embedding via Voyage AI (`input_type="query"`) and queries ChromaDB for the top-k nearest neighbors by cosine similarity. The lexical branch tokenizes the query with the same camelCase-aware tokenizer and runs `BM25Okapi.get_scores()`, returning the top-k chunks by BM25 score. Both branches are submitted as futures and resolved; failures in either branch are caught and the other branch's results are used alone.

Results from both branches are merged by `reciprocal_rank_fusion()`, which implements the standard RRF formula with `k=60`: for each document appearing in any ranked list, its RRF score is the sum of `1 / (60 + rank)` across all lists. Documents are deduplicated by a `file_path:start_line:end_line:node_type` key before scoring. The final merged list is sorted by descending RRF score, capped at `top_k`, and each chunk receives an injected `rrf_score` field for observability.

### Layer 7 — Context-Aware Pruning & Prompt Construction (`retrieval/pruner.py`)

This is the most architecturally distinctive layer of NoesisCLI. Instead of concatenating raw retrieved chunks into a prompt, Phase 6 reconstructs a token-efficient skeletal representation of the codebase.

**DependencyContextResolver** (Phase 6.1): Takes the fused retrieved chunks and walks the SymbolTable and DependencyGraph to classify all relevant symbols into two sets. Symbols directly referenced in the retrieved chunks become *target symbols* and will be included with their full implementation bodies. Symbols that are dependencies of targets — parent classes, called functions, inherited interfaces — become *reference symbols* and will appear only as signatures with `...` placeholders. This classification reduces prompt length while preserving the structural context the LLM needs.

**CodeStructurePruner** (Phase 6.2): For each source file touched by the retrieved chunks, it reconstructs a minimal file view. Target symbols appear in full. For reference symbols, the pruner preferentially reuses the pre-computed `class_header` chunks emitted during parsing (avoiding a second parse pass) and falls back to a `_sig_from_code()` helper that extracts just the decorator lines and `def`/`class` signature from the raw source, appending `...` as the body stub. The result is a list of `PrunedBlock` named tuples, each carrying the file path and its reconstructed skeletal content.

**PromptConstructor** (Phase 6.3): Assembles the final LLM prompt. It combines the pruned file blocks with dependency metadata (which symbols call which, which classes inherit from which), chunk locations, and the user's original query. A static system instruction establishes the LLM's role as a codebase reasoning assistant.

### Layer 8 — Fail-safe LLM Client & Terminal UI (`models/client.py`, `utils/ui.py`)

`GeminiClient` wraps LangChain's `ChatGoogleGenerativeAI` with a primary/fallback model pair. Both model instances are lazily initialized on first use to avoid startup latency. The `stream()` method yields tokens from the primary model (`gemini-3.1-flash-lite`); if the primary stream raises any exception (rate limit, quota, network error), it transparently falls back to `gemini-3.5-flash`. The API key is read from either `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variables.

`stream_response()` in `utils/ui.py` consumes the token generator and accumulates tokens into a `rich.live` panel that re-renders as live Markdown on each new token. Progress bars for the parsing and embedding phases use a unified `make_progress()` factory that configures a spinner, bar, M-of-N count, and elapsed time columns. All Rich calls are guarded by an `_RICH_AVAILABLE` flag so the CLI remains functional in minimal environments that lack the `rich` package.

---

## Features

### 🌳 AST-Accurate Semantic Chunking
Tree-sitter parses every Python file into structured **Code Chunks** with exact line ranges — no character-count splitting. Nine chunk types are emitted:

| Chunk Type | Description |
|---|---|
| `module` | File docstring + aggregated import list |
| `imports` | All import statements (first-class chunk for Phase 4) |
| `class` | Full class body including all methods |
| `class_header` | Class signature + docstring + method signatures only (used as stubs by Phase 6) |
| `function` | Module-level function with full body |
| `method` | Class method with full body |
| `constant` | Top-level ALL_CAPS assignments |
| `type_alias` | `TypeVar`, `Union`, `TypeAlias`, etc. |
| `global` | Any other top-level statement block |

Six bug-fix strategies are applied: decorated definition handling, import isolation, per-file import collection, nested function containment, async detection, and class-body traversal correctness.

Every chunk follows this schema:

```python
{
    "code_content": str,       # Raw source text of the construct
    "file_path":    str,       # Absolute path to the source file
    "node_type":    str,       # See chunk types table above
    "start_line":   int,       # 1-indexed
    "end_line":     int,       # 1-indexed
    "metadata": {
        "imports_in_file": list[str],  # All imports in the file (for Phase 4)
        "decorators":      list[str],  # e.g. ["@staticmethod"]
        "is_async":        bool,
        "parent_class":    str | None,
        "is_dunder":       bool,       # __init__, __repr__, etc.
        "special_type":    str | None, # "property" | "staticmethod" | "classmethod" | ...
        "docstring":       str | None,
        "func_name":       str | None, # function / method chunks
        "class_name":      str | None, # class chunks
        "base_classes":    list[str],  # class chunks
        "module_docstring":str | None,
    }
}
```

### ⚡ Multi-core Parallel Indexing
`ProcessPoolExecutor` distributes `TreeSitterParser` instances across all CPU cores. Each worker is a module-level function (fully picklable), constructs its own parser, and returns chunk lists. IPC overhead is amortised via batching. Configure with `--workers N`.

### 🔍 Hybrid Retrieval — Dense + Lexical + RRF
Both search strategies run **concurrently** in a `ThreadPoolExecutor`:

- **Dense**: Voyage AI `voyage-code-3` query embedding → ChromaDB cosine similarity
- **Lexical**: BM25Okapi over a camelCase/snake_case-aware tokenizer (`getUserId` → `["get", "user", "id"]`)

Results are merged via **Reciprocal Rank Fusion**:

```
RRF(d) = Σ_{m ∈ M}  1 / (k + rank_m(d))    k = 60
```

Chunks are deduplicated by `file_path:start_line:end_line:node_type` key. The final list is capped at `top_k` after fusion.

### 🗺️ Global Symbol Table
Indexes every `class`, `method`, and `function` declaration into `SymbolDefinition` records with: `symbol_name`, `node_type`, `file_path`, `start_line`, `end_line`, `parent_class`, `signature`, `docstring`, `is_async`, `decorators`, `base_classes`. Supports exact and case-insensitive fuzzy lookup. Persisted to `.noesis/symbol_table.pkl`.

### 🔗 Codebase Dependency Graph
A directed `networkx.DiGraph` with three edge relation types:

| Edge Type | Source → Target | Attribute |
|---|---|---|
| Import | `file` → `module` | `relation="imports"` |
| Inheritance | `class_name` → `base_class_name` | `relation="inherits"` |
| Call (best-effort) | `caller_name` → `callee_name` | `relation="calls"` |

Persisted to `.noesis/dependency_graph.pkl`.

### ✂️ Context-Aware Pruning (Phase 6)
Three-stage pipeline that produces a token-efficient skeletal prompt instead of raw file dumps:

1. **`DependencyContextResolver`** — walks retrieved chunks through the SymbolTable and DepGraph, classifying each symbol as *target* (keep full body) or *reference* (stub to signature + `...`)
2. **`CodeStructurePruner`** — reconstructs per-file views using pre-built `class_header` chunks as stubs; only target symbols retain their implementation bodies
3. **`PromptConstructor`** — assembles the final prompt with pruned file blocks, dependency metadata, file locations, and the user query

### 🤖 Fail-safe Gemini Client
Wraps all Gemini API calls via LangChain's `ChatGoogleGenerativeAI`:
- **Primary**: `gemini-3.1-flash-lite` (low latency, low cost)
- **Fallback**: `gemini-3.5-flash` (auto-switches on any exception: rate limit, quota, network)
- Lazy initialization — model objects are created only on the first API call
- Supports both `generate()` (blocking) and `stream()` (token generator) modes

### 🎨 Rich Terminal UI
- `stream_response()` — accumulates streamed LLM tokens into a `rich.live` panel with real-time Markdown rendering
- `make_progress()` — spinner + bar + M-of-N count + elapsed time for both parsing and embedding phases
- All UI functions degrade gracefully to plain `print()` when `rich` is not installed

---

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/your-username/NoesisCLI.git
cd NoesisCLI
uv sync
```

### Using pip

```bash
git clone https://github.com/your-username/NoesisCLI.git
cd NoesisCLI
pip install -e .
```

**Requirements:** Python ≥ 3.12, a Voyage AI API key, a Google Gemini API key.

---

## Configuration

```bash
cp .env.example .env
```

```env
# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# LLM Model overrides (optional — defaults shown)
GEMINI_PRIMARY_MODEL=gemini-3.1-flash-lite
GEMINI_FALLBACK_MODEL=gemini-3.5-flash

# Voyage AI
VOYAGE_API_KEY=your_voyage_api_key_here

# Internals (optional)
NOESIS_DIR_NAME=.noesis
LOG_LEVEL=INFO
```

> The index lives entirely inside `<repo>/.noesis/` — your code never leaves your machine.

---

## Usage

```bash
uv run -m noesiscli.cli <command> [options]
```

### `analyze` — Index a repository

```bash
uv run -m noesiscli.cli analyze <repo_path> [--force] [--workers N]
```

| Flag | Description |
|---|---|
| `repo_path` | Path to the local repository |
| `--force` | Re-index even if `.noesis/` already exists |
| `--workers N` | Parallel parser workers (default: all CPU cores) |

**Artifacts written to `.noesis/`:**

| Path | Contents |
|---|---|
| `chroma/` | SQLite-backed ChromaDB collection |
| `bm25.pkl` | Pickled `BM25Okapi` index + tokenized corpus |
| `symbol_table.pkl` | Pickled `SymbolTable` registry |
| `dependency_graph.pkl` | Pickled `networkx.DiGraph` |

---

### `query` — Ask a question about the codebase

```bash
cd /path/to/repo   # or use --repo-path
uv run -m noesiscli.cli query "How does authentication work?"
uv run -m noesiscli.cli query "Explain the data flow" --repo-path /path/to/repo
```

Loads all four `.noesis/` artifacts, runs hybrid retrieval, prunes context through Phases 6.1–6.3, and streams a Markdown-rendered answer.

---

### `ask` — General programming questions

```bash
uv run -m noesiscli.cli ask "What is the difference between a list and a generator?"
```

Skips retrieval entirely. Routes directly to `gemini-3.1-flash-lite` via `DirectResponder`.

---

## Project Structure

```
NoesisCLI/
├── noesiscli/
│   ├── cli.py                     # argparse entry point — analyze / query / ask
│   ├── config.py                  # Model names, .noesis/ path, language map
│   ├── parser/
│   │   ├── scanner.py             # RepositoryScanner (os.walk + ignore list)
│   │   ├── tree_sitter_parser.py  # TreeSitterParser — 9 chunk types, 6 bug-fix strategies
│   │   ├── parallel.py            # ParallelParserPipeline (ProcessPoolExecutor)
│   │   ├── symbol_table.py        # SymbolTable + SymbolDefinition dataclass
│   │   └── dependency_graph.py    # DependencyGraph — NetworkX DiGraph
│   ├── indexing/
│   │   ├── embedding.py           # EmbeddingGenerator — Voyage AI voyage-code-3
│   │   ├── vector_store.py        # ChromaVectorStore — PersistentClient
│   │   └── bm25_store.py          # BM25Store — BM25Okapi + camelCase tokenizer
│   ├── retrieval/
│   │   ├── fusion.py              # HybridRetriever + reciprocal_rank_fusion()
│   │   └── pruner.py              # DependencyContextResolver · CodeStructurePruner · PromptConstructor
│   ├── pipeline/
│   │   ├── state.py               # WorkflowState TypedDict
│   │   ├── graph.py               # WorkflowGraph — LangGraph StateGraph
│   │   ├── rag.py                 # RAGNode — Phase 6 integration + streaming
│   │   └── direct.py              # DirectResponder — general LLM path
│   ├── models/
│   │   └── client.py              # GeminiClient — primary/fallback, stream/generate
│   └── utils/
│       └── ui.py                  # stream_response · make_progress · embedding_progress
├── tests/
│   ├── conftest.py
│   ├── test_parser.py             # TreeSitterParser chunk extraction
│   ├── test_indexing.py           # EmbeddingGenerator · ChromaVectorStore · BM25Store
│   ├── test_retrieval.py          # HybridRetriever · RRF fusion
│   ├── test_pipeline.py           # WorkflowGraph · RAGNode · DirectResponder
│   └── test_models.py             # GeminiClient primary/fallback routing
├── docs/
│   ├── docs_pruner.md             # Phase 6 design reference
│   └── docs_tree_sitter.md        # Parser design reference
├── implementation.md              # Full phase-by-phase implementation plan
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## Testing

```bash
uv run pytest          # full suite
uv run pytest -v       # verbose
uv run pytest -x       # stop on first failure
```

API calls are bypassed in all tests via `PYTEST_CURRENT_TEST` environment detection — `EmbeddingGenerator` returns dummy 1536-dim vectors and `GeminiClient` is mocked.

---

## License

[MIT License](LICENSE)

---

<div align="center">
<sub>Tree-sitter &nbsp;·&nbsp; Voyage AI &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; BM25 &nbsp;·&nbsp; NetworkX &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; Gemini &nbsp;·&nbsp; Rich</sub>
</div>
