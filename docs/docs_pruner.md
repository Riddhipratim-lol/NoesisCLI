# Phase 6 — Context-Aware Pruning & Prompt Construction

**File:** [`noesiscli/retrieval/pruner.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/retrieval/pruner.py)  
**Phase:** 6 — Context-Aware Pruning & Prompt Construction  
**Sub-phases:** 6.1 `DependencyContextResolver` · 6.2 `CodeStructurePruner` · 6.3 `PromptConstructor`

---

## Overview

Phase 6 sits at the heart of NoesisCLI's token-efficiency strategy. After the Hybrid Retriever (Phase 3.2) returns a ranked list of relevant code chunks, Phase 6 decides **what to keep in full, what to stub, and how to format the final prompt** that the LLM actually sees.

The core problem it solves: blindly sending every retrieved file to the LLM produces bloated, noisy context. A 1 200-line file retrieved because one function is relevant still carries ~1 150 lines the model does not need. Phase 6 surgically reduces this to the critical skeleton while preserving enough architectural context for accurate reasoning.

**Three sequential components:**

```
HybridRetriever (Phase 3.2)
        │  ranked Code Chunk dicts
        ▼
DependencyContextResolver (6.1)
        │  target_symbols, reference_symbols, file_chunks_map
        ▼
CodeStructurePruner (6.2)
        │  PrunedBlock list (skeletal file views)
        ▼
PromptConstructor (6.3)
        │  formatted prompt string + system instruction
        ▼
GeminiClient.stream() (Phase 7.1)
```

---

## Quick Start

```python
from noesiscli.retrieval.pruner import build_pruned_prompt

# retrieved_chunks comes from HybridRetriever.retrieve()
prompt_str, system_instruction = build_pruned_prompt(
    query="How does authentication work?",
    retrieved_chunks=retrieved_chunks,
    symbol_table=symbol_table,   # from SymbolTable.load(...)
    dep_graph=dep_graph,         # from DependencyGraph.load(...)
    all_chunks=all_chunks,       # optional: all parsed chunks for richer file views
    max_call_depth=1,
    top_k_files=8,
)
```

`build_pruned_prompt()` is the single convenience entry-point that wires 6.1 → 6.2 → 6.3 end-to-end. The returned `(prompt_str, system_instruction)` pair is ready for direct consumption by `GeminiClient.stream()`.

---

## Data Contracts

### Input: Code Chunk dict

Every chunk produced by `TreeSitterParser` (Phase 1.2 / 5.1) and consumed by Phase 6 has this shape:

```python
{
    "node_type":    str,   # "module" | "imports" | "class" | "class_header" |
                           # "method" | "function" | "constant" | "type_alias" | "global"
    "file_path":    str,   # Absolute source file path
    "start_line":   int,   # 1-indexed start
    "end_line":     int,   # 1-indexed end (inclusive)
    "code_content": str,   # Raw source text of this construct
    "metadata": {
        "func_name":      str | None,   # Present on "function" / "method" chunks
        "class_name":     str | None,   # Present on "class" / "class_header" chunks
        "parent_class":   str | None,   # Enclosing class (for methods)
        "is_async":       bool,
        "decorators":     list[str],
        "is_dunder":      bool,
        "special_type":   str | None,   # e.g. "@property", "@staticmethod"
        "docstring":      str | None,
        "imports_in_file": list[str],   # All import lines in the file (Phase Fix 3)
        "base_classes":   list[str],    # For class chunks
    },
    # Optional — present on RRF-fused results from HybridRetriever:
    "rrf_score": float,
}
```

### Output: PrunedBlock

```python
@dataclass
class PrunedBlock:
    file_path:       str        # Absolute source file path
    pruned_content:  str        # Reconstructed skeletal source text
    kept_symbols:    list[str]  # Fully-preserved symbol names
    stubbed_symbols: list[str]  # Symbols replaced with stubs
```

---

## 6.1 — DependencyContextResolver

**Class:** [`DependencyContextResolver`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/retrieval/pruner.py#L130-L230)

### Responsibility

Given the ranked candidate chunks from the Hybrid Retriever, this component determines:

- **`target_symbols`** — symbols whose *full* implementation body must be preserved.
- **`reference_symbols`** — symbols that are architecturally related but whose body can be replaced by a signature stub.
- **`file_chunks_map`** — `{file_path: [chunk, ...]}` mapping every chunk in the touched files, giving the Pruner all the raw material it needs.

### Constructor

```python
DependencyContextResolver(
    symbol_table: SymbolTable | None = None,
    dep_graph: DependencyGraph | None = None,
    max_call_depth: int = 1,
)
```

| Parameter | Default | Description |
|---|---|---|
| `symbol_table` | `None` | Loaded Phase 4.1 registry. Used to resolve unqualified names to `ClassName.method_name` qualified names. |
| `dep_graph` | `None` | Loaded Phase 4.2 NetworkX DiGraph. Provides call-chain and inheritance edges. |
| `max_call_depth` | `1` | Maximum hops to follow in the call graph. `1` = direct callees of retrieved symbols. Set to `2` for deeper chains (higher token cost). |

### `resolve()` — the algorithm

```python
def resolve(
    retrieved_chunks: list[dict],
    all_chunks: list[dict] | None = None,
) -> tuple[set[str], set[str], dict[str, list[dict]]]
```

**Step 1 — Seed targets from retrieved chunks.**  
Every retrieved chunk's symbol name is immediately added to `target_symbols`. The file path is tracked in `retrieved_files`.

```
retrieved chunk: method "authenticate" in class "UserService"
  → target_symbols += {"UserService.authenticate"}
  → retrieved_files += {"/repo/auth.py"}
```

**Step 2 — Expand via Dependency Graph.**  
For each seeded target name, the call-chain and inheritance edges in the `DependencyGraph` are followed up to `max_call_depth` hops:

- **`calls` edges** — all functions/methods directly invoked by the target are added to `reference_symbols`.
- **`inherits` edges** — base classes of the target (if it is a class) are added to `reference_symbols`.

```
authenticate → calls → validate_token
  → reference_symbols += {"validate_token"}
```

**Step 3 — Resolve via Symbol Table.**  
Bare names in `reference_symbols` (e.g. `"validate_token"`) are looked up in the `SymbolTable` to resolve their qualified names and discover additional file paths:

```
"validate_token" → SymbolTable.lookup("validate_token")
  → SymbolDefinition(node_type="function", file_path="/repo/utils.py", ...)
  → reference_symbols = {"validate_token"}   (resolved qualified name)
  → retrieved_files += {"/repo/utils.py"}
```

**Step 4 — Build `file_chunks_map`.**  
All chunks (from `all_chunks` if provided, otherwise from `retrieved_chunks`) belonging to `retrieved_files` are grouped into a `{file_path: [chunk, ...]}` dict and returned.

**Returns:** `(target_symbols, reference_symbols, file_chunks_map)`

### Graceful degradation

If `symbol_table` or `dep_graph` is `None`, the resolver simply skips the corresponding step. The retrieved chunks still become `target_symbols`; no references are added. The system continues without error.

---

## 6.2 — CodeStructurePruner

**Class:** [`CodeStructurePruner`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/retrieval/pruner.py#L234-L360)

### Responsibility

Takes the `file_chunks_map` from the Resolver and reconstructs a minimal skeletal view of every touched file, producing one `PrunedBlock` per file.

### Constructor

```python
CodeStructurePruner(
    symbol_table: SymbolTable | None = None,
)
```

### `prune()` — the algorithm

```python
def prune(
    file_chunks_map: dict[str, list[dict]],
    target_symbols: set[str],
    reference_symbols: set[str],
) -> list[PrunedBlock]
```

For each file in `file_chunks_map`, `_prune_file()` is called internally:

**Step 1 — Index `class_header` chunks.**  
`class_header` chunks (pre-computed skeletal views from [S2] in the Tree-sitter parser) are extracted and indexed by class name. They are the preferred stub source for non-target classes because they are already structurally correct and require no text rewriting.

**Step 2 — Sort remaining chunks by `start_line`.**  
All non-`class_header` chunks are sorted by their position in the file so the reconstructed content reads naturally.

**Step 3 — Decide per chunk: keep / stub / header.**  
The decision table:

| `node_type` | Condition | Action |
|---|---|---|
| `module`, `imports`, `constant`, `type_alias`, `global` | Always | Include verbatim (structural metadata) |
| `class` | `qname` in `target_symbols` | Keep full class body → `kept_symbols` |
| `class` | `qname` NOT in targets | Use pre-built `class_header` chunk if available, else generate stub → `stubbed_symbols` |
| `function` / `method` | `qname` in `target_symbols` | Keep full body → `kept_symbols` |
| `function` / `method` | `qname` in `reference_symbols` | Generate signature stub → `stubbed_symbols` |
| `function` / `method` | neither | Generate signature stub → `stubbed_symbols` |

**Step 4 — Assemble pruned content.**  
All parts are joined with blank lines between them into a coherent source listing. The result is stored in `PrunedBlock.pruned_content`.

### Stub generation — `_sig_from_code()`

The helper `_sig_from_code(code_content, node_type)` extracts the first meaningful line(s) of a function or class definition and appends `...` as the body:

```python
# Input (method):
def authenticate(self, user_id: int) -> bool:
    """Authenticate a user."""
    return validate_token(user_id)

# Output stub:
def authenticate(self, user_id: int) -> bool:
    ...
```

Decorators preceding the `def`/`class` keyword are preserved in the stub:

```python
@staticmethod
def from_dict(data: dict) -> "UserService":
    ...
```

### Example output (`PrunedBlock`)

Given a retrieved `authenticate` method in `auth.py`:

```python
# auth.py — pruned view

"""Auth module."""
import hashlib

class UserService:
    ...                   # ← class_header stub (non-target class)

    def authenticate(self, user_id: int) -> bool:
        """Authenticate a user."""
        return validate_token(user_id)   # ← full body (target)

    def logout(self, user_id: int) -> None:
        ...               # ← stub (non-target method)
```

```python
# utils.py — pruned view

def validate_token(user_id: int) -> bool:
    ...                   # ← stub (reference symbol)
```

---

## 6.3 — PromptConstructor

**Class:** [`PromptConstructor`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/retrieval/pruner.py#L363-L490)

### Responsibility

Assembles the final structured prompt string from all upstream outputs. The prompt is split into clearly labelled sections so the LLM receives maximum signal with minimum noise.

### Constructor

```python
PromptConstructor(
    symbol_table: SymbolTable | None = None,
    dep_graph: DependencyGraph | None = None,
    max_file_blocks: int = 8,
)
```

| Parameter | Default | Description |
|---|---|---|
| `symbol_table` | `None` | Used to render symbol signatures in the Key Symbol Definitions section. |
| `dep_graph` | `None` | Used to render call and inheritance summaries in the Dependency Relationships section. |
| `max_file_blocks` | `8` | Maximum number of `PrunedBlock` file listings to include. Prevents runaway context on very large repositories. |

### `build()` — prompt structure

```python
def build(
    query: str,
    pruned_blocks: list[PrunedBlock],
    target_symbols: set[str],
    reference_symbols: set[str],
    retrieved_chunks: list[dict] | None = None,
) -> str
```

The returned prompt string is composed of up to five sections separated by `---` dividers:

#### Section 1 — Retrieved Code Context

Each `PrunedBlock` is rendered as a fenced Python code block under a heading that names the source file. Inline notes document which symbols were kept vs. stubbed:

```markdown
## Retrieved Code Context

### `/repo/auth.py`
> **Full implementations**: `UserService.authenticate`
> **Stubbed to signatures**: `UserService`, `UserService.logout`

```python
"""Auth module."""
import hashlib

class UserService:
    ...

    def authenticate(self, user_id: int) -> bool:
        """Authenticate a user."""
        return validate_token(user_id)

    def logout(self, user_id: int) -> None:
        ...
```
```

#### Section 2 — Dependency Relationships

Summarises call-chain and inheritance edges for every `target_symbol`, derived from the `DependencyGraph`:

```markdown
## Dependency Relationships

- `UserService.authenticate` **calls**: `validate_token`
- `AdminService` **inherits from**: `UserService`
```

#### Section 3 — Key Symbol Definitions

Lists the resolved `SymbolDefinition` records for each `target_symbol`, including file location and extracted signature:

```markdown
## Key Symbol Definitions

- **`UserService.authenticate`** (method) at `/repo/auth.py` L8–20
  ```python
  def authenticate(self, user_id: int) -> bool
  ```
```

#### Section 4 — Retrieved Chunk Locations

A compact numbered list of every chunk's file path, line range, type, and RRF score:

```markdown
## Retrieved Chunk Locations

1. `/repo/auth.py` L8–20 (method) [RRF: 0.0159]
```

#### Section 5 — User Query

The original, unmodified user query string:

```markdown
## User Query

How does authentication work?
```

### System Instruction

The `PromptConstructor.SYSTEM_INSTRUCTION` class attribute holds the system-level prompt that accompanies every RAG query. Key clauses:

- Role-sets the model as **NoesisCLI**, a professional Python codebase architect.
- Explains that `...` bodies are **intentional stubs**, not missing code.
- Instructs the model to reference **specific file paths and line numbers**.
- Instructs the model to **acknowledge context gaps** rather than hallucinate.

---

## `build_pruned_prompt()` — Convenience Entry-Point

```python
def build_pruned_prompt(
    query: str,
    retrieved_chunks: list[dict],
    symbol_table: SymbolTable | None = None,
    dep_graph: DependencyGraph | None = None,
    all_chunks: list[dict] | None = None,
    max_call_depth: int = 1,
    top_k_files: int = 8,
) -> tuple[str, str]
```

Wires the full Phase 6 pipeline in three lines:

```python
# 1. Resolve dependencies (Phase 6.1)
target_symbols, reference_symbols, file_chunks_map = resolver.resolve(retrieved_chunks, all_chunks)

# 2. Prune file structures (Phase 6.2)
pruned_blocks = pruner.prune(file_chunks_map, target_symbols, reference_symbols)

# 3. Build prompt (Phase 6.3)
prompt_str = constructor.build(query, pruned_blocks, target_symbols, reference_symbols, retrieved_chunks)
```

**Returns:** `(prompt_str, system_instruction)` — both strings are ready for `GeminiClient.stream(prompt_str, system_instruction=system_instruction)`.

---

## Integration with the RAG Pipeline

### `RAGNode` (Phase 3.2 + Phase 6)

**File:** [`noesiscli/pipeline/rag.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/pipeline/rag.py)

`RAGNode` is the LangGraph node that runs Phase 6. It accepts `symbol_table` and `dep_graph` in its constructor (threaded in by `WorkflowGraph`):

```python
class RAGNode:
    def __init__(
        self,
        llm_client: GeminiClient | None = None,
        retriever=None,
        symbol_table: SymbolTable | None = None,
        dep_graph: DependencyGraph | None = None,
    ): ...
```

Inside `execute()`:

1. If `self.last_chunks` is pre-populated (by `WorkflowGraph.rag_node_node()`), those chunks are used directly (no double-retrieval).
2. If `symbol_table` **or** `dep_graph` is present, `build_pruned_prompt()` is called → Phase 6 pruning runs.
3. If **neither** is present, `_plain_context_prompt()` builds a raw code-block prompt (Phase 1.5-style fallback). No errors are raised; the system degrades gracefully.

### `WorkflowGraph`

**File:** [`noesiscli/pipeline/graph.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/pipeline/graph.py)

`WorkflowGraph` receives `symbol_table` and `dep_graph` from the CLI and passes them into `RAGNode`:

```python
self.rag_node = RAGNode(
    llm_client=self.llm_client,
    retriever=self.retriever,
    symbol_table=self.symbol_table,   # Phase 4.1 → Phase 6
    dep_graph=self.dep_graph,         # Phase 4.2 → Phase 6
)
```

The `rag_node_node()` handler logs Phase 6 activation status to stdout before streaming begins:

```
[Phase 6] Context pruning active — SymbolTable(42 defs), DependencyGraph(87 nodes)
```

or, when the index is incomplete:

```
[Phase 6] Context pruning unavailable — using plain context.
```

### CLI (`cli.py`)

**File:** [`noesiscli/cli.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/cli.py)

The `query` command loads `symbol_table` and `dep_graph` from `.noesis/` and passes them into both `WorkflowGraph` and the LangGraph `initial_state`:

```python
initial_state = {
    "query": prompt,
    "route": "repository_rag",
    "context_chunks": [],
    "response": "",
    "symbol_table": symbol_table,  # Phase 4.1 — consumed by Phase 6
    "dep_graph": dep_graph,        # Phase 4.2 — consumed by Phase 6
}
```

---

## Data Flow Diagram

```
noesiscli query "How does authentication work?"
        │
        ▼
CLI loads from .noesis/:
  ├─ ChromaVectorStore  (chroma/)
  ├─ BM25Store          (bm25.pkl)
  ├─ SymbolTable        (symbol_table.pkl)    ──────────────┐
  └─ DependencyGraph    (dependency_graph.pkl) ─────────────┤
        │                                                   │
        ▼                                                   │
HybridRetriever.retrieve(query)                             │
  ├─ ChromaDB dense search  ─┐                              │
  └─ BM25 lexical search    ─┴─ RRF fusion                 │
        │ ranked Code Chunk dicts                           │
        ▼                                                   │
DependencyContextResolver.resolve()   ◄────────────────────┤
  ├─ target_symbols   = {"UserService.authenticate"}        │
  ├─ reference_symbols = {"validate_token"}                 │
  └─ file_chunks_map  = {"/repo/auth.py": [...],           │
                          "/repo/utils.py": [...]}          │
        │                                                   │
        ▼                                                   │
CodeStructurePruner.prune()                                 │
  ├─ /repo/auth.py  → authenticate: FULL BODY              │
  │                   UserService:  class_header stub       │
  │                   logout:       signature stub          │
  └─ /repo/utils.py → validate_token: signature stub       │
        │ list[PrunedBlock]                                 │
        ▼                                                   │
PromptConstructor.build()     ◄────────────────────────────┘
  ├─ Section 1: Retrieved Code Context (pruned blocks)
  ├─ Section 2: Dependency Relationships (calls/inherits)
  ├─ Section 3: Key Symbol Definitions (signatures + locs)
  ├─ Section 4: Retrieved Chunk Locations (metadata)
  └─ Section 5: User Query
        │ (prompt_str, system_instruction)
        ▼
GeminiClient.stream(prompt_str, system_instruction)
        │ streamed tokens
        ▼
Terminal output
```

---

## Design Decisions

### Why `class_header` chunks are preferred for stubs

The Tree-sitter parser emits pre-built `class_header` chunks (Strategy [S2]) that already contain the class signature, docstring, and all method signatures with `...` bodies. Using these for non-target classes is:

- **Faster** — no text rewriting required.
- **Accurate** — produced directly from the AST, so indentation and multi-line signatures are always correct.
- **Consistent** — the same format as the rest of the parser output.

`_sig_from_code()` is only invoked as a fallback when no `class_header` is available.

### Why `max_call_depth=1` is the default

Following call chains deeper (depth 2 or 3) can rapidly expand the reference set to dozens of symbols, each requiring at least a signature stub in the prompt. The default depth of 1 (only direct callees of the retrieved symbols) provides high-value context — the functions a retrieved method *directly calls* — without risking context explosion.

Users who need deeper cross-file tracing can pass `max_call_depth=2` to `build_pruned_prompt()`.

### Why structural chunks (`module`, `imports`, `constant`) are always kept

These chunk types carry file-level metadata that gives the LLM grounding context: which modules are imported, what constants are defined, and what the file-level docstring says. They are typically very short (a few lines) so including them verbatim does not meaningfully increase token cost, while their absence can cause the LLM to misidentify dependencies or infer wrong type information.

### Graceful degradation when Phase 4 is absent

All three Phase 6 components accept `None` for `symbol_table` and `dep_graph`. If the user runs `noesiscli query` against an index built before Phase 4 was added, the system falls back silently:

- `DependencyContextResolver` — seeds targets from retrieved chunks only; skips Steps 2 and 3.
- `CodeStructurePruner` — stubs everything except target symbols (no call-chain expansion to guide reference selection).
- `RAGNode` — falls all the way back to plain context blocks (Phase 1.5 style) if both Phase 4 structures are `None`.

---

## `PrunedBlock` Reference

```python
@dataclass
class PrunedBlock:
    file_path:       str        # "/abs/path/to/source.py"
    pruned_content:  str        # reconstructed skeletal source text
    kept_symbols:    list[str]  # e.g. ["UserService.authenticate"]
    stubbed_symbols: list[str]  # e.g. ["UserService", "UserService.logout"]
```

`kept_symbols` and `stubbed_symbols` are surfaced in the prompt as inline notes above each code block, giving the LLM explicit awareness of which bodies were intentionally omitted.

---

## Private Helpers

### `_sig_from_code(code_content, node_type) -> str`

Extracts the signature (decorator lines + `def`/`class` header) from raw source and appends an indented `...` stub body. Handles multi-line signatures (long argument lists spanning multiple lines ending in `:`).

### `_extract_base_classes(code_content) -> list[str]`

Parses `class Foo(Bar, Baz):` using a regex to return `["Bar", "Baz"]`. Used internally during dependency expansion if the `DependencyGraph` is unavailable.

### `_infer_name_from_code(code_content, node_type) -> str | None`

Best-effort symbol name inference from raw source when metadata `func_name`/`class_name` keys are absent. Mirrors the equivalent helper in `symbol_table.py` to keep the pruner self-contained.

---

## Module Exports

```python
# noesiscli/retrieval/__init__.py
from noesiscli.retrieval.pruner import (
    DependencyContextResolver,
    CodeStructurePruner,
    PromptConstructor,
    PrunedBlock,
    build_pruned_prompt,
)
```

All five names are available from the `noesiscli.retrieval` package directly.

---

## Files Modified in This Session

| File | Change |
|---|---|
| [`noesiscli/retrieval/pruner.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/retrieval/pruner.py) | Written from scratch — full Phase 6 implementation (~450 lines) |
| [`noesiscli/retrieval/__init__.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/retrieval/__init__.py) | Added Phase 6 exports alongside existing `HybridRetriever` |
| [`noesiscli/pipeline/rag.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/pipeline/rag.py) | Rewritten to call `build_pruned_prompt()` with graceful fallback |
| [`noesiscli/pipeline/graph.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/pipeline/graph.py) | Threads `symbol_table` and `dep_graph` into `RAGNode`; adds Phase 6 status logging |
| [`noesiscli/cli.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/cli.py) | Passes Phase 4 structures into `initial_state` with Phase 6 commentary |
| [`implementation.md`](file:///Users/riddhipratim/Projects/NoesisCLI/implementation.md) | Phase 6.1, 6.2, 6.3 checkboxes marked `[x]` |
