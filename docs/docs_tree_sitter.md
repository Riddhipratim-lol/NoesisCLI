# Tree-Sitter Parser — Developer Reference

**File:** [`noesiscli/parser/tree_sitter_parser.py`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py)  
**Class:** [`TreeSitterParser`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L61-L738)  
**Phase:** 1.2 — Tree-Sitter Parser & Base Semantic Chunker

---

## Overview

`TreeSitterParser` converts raw Python source files into a list of **structured semantic chunk dicts** using Tree-sitter's AST engine. Instead of character-count splitting, it walks the AST and emits one chunk per logical programming construct — preserving complete semantic boundaries that downstream phases (embedding, BM25 indexing, Symbol Table, Dependency Graph) can directly consume.

Every construct in a file is captured. Nothing is dropped or silently skipped.

---

## Quick Start

```python
from noesiscli.parser.tree_sitter_parser import TreeSitterParser

parser = TreeSitterParser()

# Parse a file from disk
chunks = parser.parse_file("/path/to/module.py")

# Or parse a string directly
chunks = parser.parse_code(source_code_string, "/path/to/module.py")

for chunk in chunks:
    print(chunk["node_type"], chunk["start_line"], chunk["code_content"][:60])
```

---

## Public API

### [`__init__(language="python")`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L69-L81)

Initialises the Tree-sitter `Language` and `Parser` objects.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | `str` | `"python"` | Language to parse. Only `"python"` / `"py"` supported in Phase 1.2. |

Raises `ValueError` if an unsupported language is requested or if the Tree-sitter grammar fails to load.

---

### [`parse_file(file_path)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L87-L94)

Reads a file from disk and delegates to `parse_code`. Returns `[]` on any read error (graceful failure — never raises).

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Absolute path to the `.py` file. |

**Returns:** `list[dict]` — list of chunk dicts (see [Chunk Schema](#chunk-schema)).

---

### [`parse_code(code, file_path)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L96-L176)

Core entry point. Parses a source string and returns all semantic chunks in document order.

| Parameter | Type | Description |
|---|---|---|
| `code` | `str` | Raw Python source code. |
| `file_path` | `str` | Used as the `file_path` value on every emitted chunk. |

**Returns:** `list[dict]` — ordered list of chunk dicts.

Returns `[]` for empty/whitespace-only input.

---

## Chunk Schema

Every chunk dict has this guaranteed shape:

```python
{
    "code_content": str,       # Raw source text of the construct
    "file_path":    str,       # Absolute source file path
    "node_type":    str,       # One of the types in the table below
    "start_line":   int,       # 1-indexed line where the construct begins
    "end_line":     int,       # 1-indexed line where it ends (inclusive)
    "metadata":     dict,      # Rich structural metadata (see below)
}
```

### `node_type` Values

| `node_type` | Description | Example construct |
|---|---|---|
| `module` | File-level overview: docstring + all imports | Entire file header |
| `imports` | All import statements aggregated into one chunk | `import os`, `from x import y` |
| `class` | Full class body including all method implementations | `class Foo(Bar): ...` |
| `class_header` | Skeletal class: signature + docstring + method signatures only | Ready for Phase 6 context pruner |
| `method` | A single method extracted from a class body | `def authenticate(self, ...)` |
| `function` | A top-level (module-level) function | `def standalone(x): ...` |
| `constant` | Top-level ALL_CAPS assignment | `MAX_RETRIES = 5` |
| `type_alias` | Top-level type alias assignment | `UserId = Union[int, str]` |
| `global` | Any other top-level statement | `if __name__ == "__main__": ...` |

### Chunk Emission Order Per File

For any given `.py` file, chunks are emitted in this fixed order:

```
1.  module          ← always first (file overview)
2.  imports         ← always second, only if the file has any imports
3+. class / class_header / method / function / constant / type_alias / global
    ← in document (top-to-bottom) order
```

> [!NOTE]
> For each class, three chunk groups are emitted in sequence: `class` (full body) → `class_header` (skeletal) → `method` × N (one per method). This means every class produces at least 2 chunks plus one per method.

---

## Metadata Schema

Every chunk carries a `metadata` dict. Not all keys are populated for every `node_type` — see the table for which keys are meaningful per type.

```python
metadata = {
    # Present on ALL chunk types
    "imports_in_file":  list[str],   # All import strings in the file
    "decorators":       list[str],   # Decorator strings including @, e.g. ["@property"]
    "is_async":         bool,        # True for async def functions/methods
    "parent_class":     str | None,  # Class name if this is a method; None otherwise
    "is_dunder":        bool,        # True if name is __xxx__ (e.g. __init__, __repr__)
    "special_type":     str | None,  # "property" | "staticmethod" | "classmethod" |
                                     # "abstractmethod" | "cached_property" | "overload"
                                     # None if no recognised decorator
    "docstring":        str | None,  # Extracted docstring text, or None

    # module chunk only
    "module_docstring": str | None,  # Same as docstring for module chunks
    "total_lines":      int,         # Total line count of the file

    # imports chunk only
    "imports_parsed":   list[str],   # Alias for imports_in_file (same list)

    # class / class_header chunks only
    "class_name":       str,         # Unqualified class name
    "base_classes":     list[str],   # Parent class name strings

    # function / method chunks only
    "func_name":        str,         # Unqualified function name
}
```

### Metadata Availability by `node_type`

| Key | `module` | `imports` | `class` | `class_header` | `method` | `function` | `constant` | `type_alias` | `global` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `imports_in_file` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `decorators` | — | — | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `is_async` | — | — | — | — | ✅ | ✅ | — | — | — |
| `parent_class` | — | — | — | — | ✅ | — | — | — | — |
| `is_dunder` | — | — | — | — | ✅ | — | — | — | — |
| `special_type` | — | — | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `docstring` | ✅ | — | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `total_lines` | ✅ | — | — | — | — | — | — | — | — |
| `class_name` | — | — | ✅ | ✅ | — | — | — | — | — |
| `base_classes` | — | — | ✅ | ✅ | — | — | — | — | — |
| `func_name` | — | — | — | — | ✅ | ✅ | — | — | — |
| `imports_parsed` | — | ✅ | — | — | — | — | — | — | — |

(✅ = populated, — = present but `None` / `[]` / not meaningful)

---

## Internal Call Graph

```
parse_file(file_path)
└── parse_code(code, file_path)
    ├── _collect_import_nodes(root)           → list[Node]
    ├── _make_module_chunk(...)               → dict          [module chunk]
    ├── _make_imports_chunk(...)              → dict          [imports chunk]
    ├── _find_module_docstring_node(root)     → Node | None
    │
    └── [top-level walk loop]
        ├── _make_global_chunk(nodes, ...)    → dict | None   [constant/type_alias/global]
        │   └── _classify_global_nodes(...)  → str
        │
        └── _extract_top_level(node, ...)    → list[dict]
            ├── _extract_decorated(...)       → list[dict]    [decorated_definition]
            │   ├── _extract_class(...)       → list[dict]
            │   └── _extract_function(...)    → dict
            │
            ├── _extract_class(node, ...)    → list[dict]    [class_definition]
            │   ├── _build_class_header(...)  → str
            │   │   └── _extract_signature_line(member, ...) → str | None
            │   ├── _extract_body_docstring(...)
            │   ├── _get_base_classes(...)
            │   └── [class body walk]
            │       ├── _extract_decorated(member, ...)      [decorated method]
            │       ├── _extract_function(member, ...)       [plain method]
            │       └── _extract_class(member, ...)          [nested class]
            │
            └── _extract_function(node, ...) → dict          [function_definition]
```

---

## Module-Level Constants

```python
_FUNC_TYPES = {"function_definition", "async_function_definition"}
# NOTE: tree-sitter-python represents both sync and async functions as
# 'function_definition'. Async is detected by the presence of an 'async'
# child token — NOT by a separate node type.

_CLASS_TYPE     = "class_definition"
_DECORATED_TYPE = "decorated_definition"
_IMPORT_TYPES   = {"import_statement", "import_from_statement"}

_SPECIAL_DECORATORS = {
    "property", "staticmethod", "classmethod",
    "abstractmethod", "cached_property", "overload"
}
```

---

## Method Reference

### Private Methods

#### [`_collect_import_nodes(root)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L182-L184)
Returns the ordered list of `import_statement` / `import_from_statement` AST nodes from the module root. Used to build `imports_in_file` and to construct the `imports` chunk.

---

#### [`_make_imports_chunk(import_nodes, code_bytes, file_path, imports_in_file)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L186-L225)
Builds the dedicated `"imports"` chunk. `code_content` is all import lines joined by `\n` exactly as written. `metadata.imports_parsed` holds the same list for easy iteration by Phase 4.

---

#### [`_make_module_chunk(root, code_bytes, file_path, imports_in_file)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L231-L271)
Builds the `"module"` chunk. `code_content` = module docstring + `\n\n` + import lines. `start_line` is always 1, `end_line` is the last line of the file.

---

#### [`_find_module_docstring_node(root)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L273-L282)
Returns the `expression_statement` AST node containing the module docstring (if any), or `None`. Used to exclude the docstring from the global accumulator so it doesn't appear duplicated in a `global`/`constant` chunk.

---

#### [`_extract_module_docstring(root, code_bytes)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L284-L289)
Thin wrapper over `_find_module_docstring_node` that returns the decoded text string instead of the node.

---

#### [`_make_global_chunk(nodes, code_bytes, file_path, imports_in_file)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L295-L333)
Flushes the `pending_global` accumulator into a single chunk spanning from the first to the last node in the group. Delegates classification to `_classify_global_nodes`. Returns `None` if the extracted text is empty.

---

#### [`_classify_global_nodes(nodes, code_bytes)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L335-L371)
Heuristic classifier for groups of top-level statement nodes.

Classification priority (checked in order on the first substantive node):

1. Node type is `type_alias_statement` → `"type_alias"`
2. Node is an `assignment` whose text contains `TypeVar`, `Union`, `Optional`, `Literal`, `Annotated`, `Final`, or `TypeAlias` → `"type_alias"`
3. Node is an `assignment` whose LHS identifier is ALL_CAPS and `len > 1` → `"constant"`
4. Fallback → `"global"`

> [!NOTE]
> Top-level assignments in the Python grammar are wrapped inside `expression_statement` nodes. The classifier unwraps this layer before type-checking.

---

#### [`_extract_top_level(node, code_bytes, file_path, imports_in_file)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L377-L400)
Dispatcher for the main top-level walk. Routes to `_extract_decorated`, `_extract_class`, or `_extract_function` based on `node.type`.

---

#### [`_extract_decorated(node, ...)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L406-L449)
Handles `decorated_definition` nodes. Extracts all `decorator` children as strings, then finds the inner `function_definition` or `class_definition` and delegates — passing `decorator_start_byte = node.start_byte` so the emitted `code_content` begins at the `@decorator` line, not the bare `def`/`class` line.

---

#### [`_extract_class(node, ...)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L455-L549)
Emits **three groups** of chunks for a class:

1. **`class` chunk** — full source text from the first decorator (or `class` keyword) to the closing line.
2. **`class_header` chunk** — skeletal view built by `_build_class_header`: decorators + class signature line + docstring + one `def foo(...)\n    ...` signature per member. No method bodies.
3. **`method` chunks** — one per member in the `block` body, produced by recursing into `_extract_decorated` or `_extract_function`.

`override_start_byte` is set when the class is decorated, so the class chunk's `start_line` and `code_content` include the `@decorator` lines.

---

#### [`_build_class_header(node, code_bytes, class_name, docstring, decorator_names)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L551-L593)
Constructs the skeletal class header string. Walks the `block` body to collect method signatures via `_extract_signature_line`, appending `...` after each.

---

#### [`_extract_signature_line(node, code_bytes)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L595-L615)
Extracts the `def foo(args) -> ret` signature line from a `function_definition` node (everything up to but not including the body `block`). Also handles `decorated_definition` by prepending decorator lines. Returns `None` for unrecognised node types.

---

#### [`_extract_function(node, ...)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L621-L684)
Builds a single `function` or `method` chunk. Key behaviours:

- **Async detection:** checks for a child token `c.type == "async"` — tree-sitter-python does not use a separate node type for async functions.
- **Decorator inclusion:** when `decorator_start_byte` is set, the byte range — and therefore `code_content` — starts at the `@decorator` line.
- **No nested recursion:** the function body is captured as-is. Nested function definitions remain embedded in the parent's `code_content` and are NOT extracted separately. This prevents double-indexing.
- **Dunder detection:** `is_dunder = name.startswith("__") and name.endswith("__")`.

---

#### [`_get_name(node, code_bytes)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L690-L695)
Returns the first `identifier` child of a `class_definition` or `function_definition` node. Returns `"<anonymous>"` if none found.

---

#### [`_get_base_classes(node, code_bytes)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L697-L707)
Reads the `argument_list` child of a `class_definition` to extract base class name strings. Filters out punctuation tokens (`,`, `(`, `)`) and `keyword_argument` nodes (e.g. `metaclass=ABCMeta`).

---

#### [`_extract_body_docstring(node, code_bytes)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L709-L726)
Finds the `block` child of a function or class, then returns the text of the first `expression_statement` if its value is a `string` or `concatenated_string` literal. Stops looking as soon as it hits any other substantive statement.

---

#### [`_special_decorator_type(decorator_names)`](file:///Users/riddhipratim/Projects/NoesisCLI/noesiscli/parser/tree_sitter_parser.py#L728-L737)
Scans decorator strings for a match in `_SPECIAL_DECORATORS`. Strips the leading `@` and any arguments (e.g. `@lru_cache(maxsize=128)` → `lru_cache`). Returns the first match or `None`.

---

## Parsing Pipeline — Step by Step

Given a source file, `parse_code` executes in this order:

```
Step 1  Encode source to UTF-8 bytes and run Tree-sitter parser → CST root
Step 2  Walk root.children → collect all import AST nodes (no side effects on other nodes)
Step 3  Decode import nodes to strings → imports_in_file list
Step 4  Emit module chunk  (docstring + import text)
Step 5  Emit imports chunk (if file has any imports)
Step 6  Find module docstring node (to exclude from global accumulator)
Step 7  Walk root.children top-to-bottom:
          import nodes       → skip (already collected)
          docstring node     → skip (already in module chunk)
          class / function   → flush pending_global; extract chunks
          everything else    → append to pending_global accumulator
Step 8  Final flush of pending_global
Step 9  Return full chunks list
```

---

## Concrete Example

Given this source:

```python
"""Auth module."""

import os
from typing import Optional

MAX_RETRIES = 5
T = Optional[str]

class UserService:
    """Manages users."""

    def __init__(self, db):
        self.db = db

    @property
    def connected(self) -> bool:
        return self.db is not None

    @classmethod
    def from_env(cls):
        return cls(None)

async def verify(token: str) -> bool:
    return bool(token)

if __name__ == "__main__":
    verify("x")
```

The parser emits these chunks in order:

| # | `node_type` | Lines | Notable metadata |
|---|---|---|---|
| 1 | `module` | 1–27 | `total_lines=27`, `module_docstring="\"\"\"Auth module.\"\"\""` |
| 2 | `imports` | 3–4 | `imports_parsed=["import os", "from typing import Optional"]` |
| 3 | `constant` | 6–6 | `MAX_RETRIES = 5` — ALL_CAPS LHS |
| 4 | `type_alias` | 7–7 | Contains `Optional` keyword |
| 5 | `class` | 9–21 | Full `UserService` body, `class_name="UserService"` |
| 6 | `class_header` | 9–21 | Skeletal view with `...` bodies, `docstring` present |
| 7 | `method` | 12–13 | `func_name="__init__"`, `is_dunder=True` |
| 8 | `method` | 15–17 | `func_name="connected"`, `special_type="property"`, `decorators=["@property"]` |
| 9 | `method` | 19–21 | `func_name="from_env"`, `special_type="classmethod"` |
| 10 | `function` | 23–24 | `func_name="verify"`, `is_async=True` |
| 11 | `global` | 26–27 | `if __name__ == "__main__":` block |

---

## Design Decisions & Rationale

### Why emit both `class` and `class_header`?

The `class` chunk feeds the **embedding + BM25 index** with full implementation detail. The `class_header` chunk is a pre-built skeletal view for Phase 6's **Context Pruner** (Code Structure Pruner), which needs to replace non-retrieved method bodies with `...` placeholders. Computing the header at parse time avoids re-parsing at retrieval time.

### Why are nested functions NOT extracted separately?

A nested function (closure) is implementation detail of its parent. Extracting it separately would:
1. Duplicate content in the index (parent chunk already contains the full text).
2. Create a chunk with no meaningful standalone semantic context.

The parent function's `code_content` already includes the nested function's full body, which is correct for LLM reasoning.

### Why is `imports` a first-class chunk and not just metadata?

Phase 4 (Dependency Graph) needs to iterate over import statements to build module-level edges. Having a dedicated `node_type == "imports"` chunk means Phase 4 can filter the chunk list with a simple equality check instead of inspecting `metadata["imports_in_file"]` on every chunk.

### Why does `is_async` check child tokens instead of node type?

In the version of `tree-sitter-python` used by this project, both `def foo()` and `async def foo()` produce a `function_definition` node. The difference is that the async version contains an `async` keyword token as a child. There is no separate `async_function_definition` node type in this grammar version.

### Why does the global classifier unwrap `expression_statement`?

The Python Tree-sitter grammar wraps almost all statements in an `expression_statement` parent. A raw `assignment` node is therefore never a direct child of the module root — it is always nested one level inside `expression_statement`. Failing to unwrap this layer means all `assignment` type-checks silently fail, causing every constant to fall through to `"global"`.

---

## Integration Points

| Downstream Phase | What it consumes |
|---|---|
| **Phase 1.3** Voyage AI Embedding Generator | `code_content`, `node_type`, `file_path` |
| **Phase 1.4** ChromaDB Storage | All fields as document + metadata payload |
| **Phase 3.1** BM25 Indexer | `code_content` tokenised for keyword index |
| **Phase 4.1** Symbol Table Builder | `node_type`, `func_name`, `class_name`, `parent_class`, `start_line`, `end_line` |
| **Phase 4.2** Dependency Graph | `node_type == "imports"`, `imports_parsed` |
| **Phase 6.2** Code Structure Pruner | `class_header` chunks used as pruning templates |

---

## Known Limitations (Phase 1.2)

> [!WARNING]
> These limitations are intentional deferrals, not bugs. They will be resolved in the phases noted.

- **Python only.** Multi-language support (JavaScript, TypeScript, Go, Java, C++) is deferred to **Phase 5.1**.
- **No nested function extraction.** Nested functions remain embedded in their parent's `code_content`. Phase 5 may add a `nested_function` chunk type with a `parent_function` reference field.
- **`constant` classification is heuristic.** Annotated assignments (`x: int = 5`) and lowercase module-level assignments are always classified as `"global"`, not `"constant"`.
- **`type_alias` relies on keyword matching.** A custom type alias that doesn't use standard `typing` module names will be classified as `"global"`.
- **Decorator argument stripping for `special_type`.** `@lru_cache(maxsize=128)` → checks `lru_cache` which is not in `_SPECIAL_DECORATORS`, so `special_type` will be `None`. Only the base decorator name is matched.
