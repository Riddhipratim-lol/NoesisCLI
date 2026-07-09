"""
Codebase Dependency Graph Constructor — Phase 4.2

Builds a directed NetworkX graph from the structured Code Chunks and the
Global Symbol Table.  Three categories of edges are emitted:

  1. Import edges  (file → module)
     Parsed from ``imports`` and ``module`` chunk types.
     Edge attribute: ``relation="imports"``.

  2. Inheritance edges  (class_name → base_class_name)
     Derived from the ``base_classes`` metadata on every ``class`` chunk.
     Edge attribute: ``relation="inherits"``.

  3. Call edges  (caller_name → callee_name)  [best-effort]
     For every ``function`` / ``method`` chunk, the code_content is scanned
     with a regex for call patterns matching known symbol names in the
     SymbolTable.  This is best-effort — it matches surface-level token
     occurrences and is not a full static analysis.
     Edge attribute: ``relation="calls"``.

Node types stored as node attributes:
  ``node_type`` ∈ { "file", "class", "function", "method", "module" }
  ``file_path``  — absolute source path (where applicable)

Data flow:
  Input:  Code Chunk dicts (Phase 1.2 / 5.2) + SymbolTable (Phase 4.1).
  Output: DependencyGraph instance, serialized to `.noesis/dependency_graph.pkl`.
          Consumed by DependencyContextResolver (Phase 6.1).
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Dict, List, Optional, Set

import networkx as nx

from noesiscli.parser.symbol_table import SymbolTable


# ---------------------------------------------------------------------------
# Import-string parsing helpers
# ---------------------------------------------------------------------------

def _parse_module_name(import_line: str) -> Optional[str]:
    """
    Extract the top-level module name from a Python import statement string.

    Examples:
        ``"import os.path"``            → ``"os"``
        ``"from typing import List"``   → ``"typing"``
        ``"from . import utils"``       → ``None``  (relative — skipped)
        ``"from ..models import User"`` → ``None``  (relative — skipped)
    """
    import_line = import_line.strip()

    # Relative imports (from . or from ..) — skip
    if re.match(r"^from\s+\.+", import_line):
        return None

    # from X import Y  /  from X.Y import Z
    m = re.match(r"^from\s+([\w.]+)\s+import", import_line)
    if m:
        return m.group(1).split(".")[0]

    # import X  /  import X.Y  /  import X as Y
    m = re.match(r"^import\s+([\w.]+)", import_line)
    if m:
        return m.group(1).split(".")[0]

    return None


def _collect_imports_from_chunks(chunks: List[dict]) -> Dict[str, List[str]]:
    """
    Build a mapping of ``{file_path: [module_name, ...]}`` from all
    ``imports`` and ``module`` chunk types.

    Returns:
        Dict keyed by file_path, values are de-duplicated module name lists.
    """
    file_imports: Dict[str, Set[str]] = {}

    for chunk in chunks:
        node_type = chunk.get("node_type", "")
        if node_type not in ("imports", "module"):
            continue

        file_path = chunk.get("file_path", "")
        metadata = chunk.get("metadata", {})

        # imports_parsed is a list of raw import statement strings (Fix 3)
        raw_imports: List[str] = metadata.get("imports_parsed") or \
                                  metadata.get("imports_in_file") or []

        for stmt in raw_imports:
            mod = _parse_module_name(stmt)
            if mod:
                file_imports.setdefault(file_path, set()).add(mod)

    return {fp: list(mods) for fp, mods in file_imports.items()}


# ---------------------------------------------------------------------------
# Call-pattern detection helper
# ---------------------------------------------------------------------------

def _find_calls_in_code(code_content: str, known_names: Set[str]) -> List[str]:
    """
    Scan *code_content* for function/method call tokens that match a set of
    known symbol names.

    Strategy: find all ``name(`` patterns in the source and return those that
    appear in ``known_names``.  This is a best-effort heuristic — it may
    produce false positives for names that are passed as values rather than
    called, and it may miss calls made through aliases or dynamic dispatch.

    Args:
        code_content: Raw source text of a function or method chunk.
        known_names:  Set of symbol names from the SymbolTable to match against.

    Returns:
        De-duplicated list of matching callee symbol names.
    """
    if not code_content or not known_names:
        return []

    # Pattern: word boundary + identifier + opening paren (call site)
    call_pattern = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
    found: Set[str] = set()

    for m in call_pattern.finditer(code_content):
        candidate = m.group(1)
        if candidate in known_names:
            found.add(candidate)

    return list(found)


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------

class DependencyGraph:
    """
    Directed codebase dependency graph.

    Wraps a ``networkx.DiGraph`` and provides domain-specific helpers for
    building, querying, and persisting the repository dependency graph.

    Node IDs are strings — either:
      - file paths (for file nodes), or
      - simple symbol names (for class / function / method nodes).

    Graph attributes on each node:
      ``node_type``  – ``"file"`` | ``"class"`` | ``"function"`` | ``"method"``
      ``file_path``  – source file (for symbol nodes)

    Edge attributes:
      ``relation``   – ``"imports"`` | ``"inherits"`` | ``"calls"``
      ``source_file`` – file where the edge originates
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: List[dict],
        symbol_table: SymbolTable,
    ) -> "DependencyGraph":
        """
        Construct the dependency graph from Code Chunks and a SymbolTable.

        Populates:
          1. File and symbol nodes.
          2. Import edges  (file → module).
          3. Inheritance edges  (class → base_class).
          4. Call edges  (caller → callee) for known symbols.

        Args:
            chunks:       List of Code Chunk dicts from the parser.
            symbol_table: Populated :class:`~noesiscli.parser.symbol_table.SymbolTable`.

        Returns:
            ``self`` to allow chaining.
        """
        self.graph.clear()

        # ── Step 1: add all known symbol nodes ──────────────────────────
        for defn in symbol_table.all_definitions():
            node_id = defn.symbol_name
            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    node_type=defn.node_type,
                    file_path=defn.file_path,
                    qualified_name=defn.qualified_name,
                )
            # Also add file nodes
            file_node = defn.file_path
            if not self.graph.has_node(file_node):
                self.graph.add_node(file_node, node_type="file", file_path=file_node)

        # ── Step 2: import edges (file → module) ─────────────────────────
        file_imports = _collect_imports_from_chunks(chunks)
        for file_path, modules in file_imports.items():
            if not self.graph.has_node(file_path):
                self.graph.add_node(file_path, node_type="file", file_path=file_path)
            for mod in modules:
                # Module node may or may not resolve to a local file — add it
                # as a generic node regardless so the graph stays complete.
                if not self.graph.has_node(mod):
                    self.graph.add_node(mod, node_type="module", file_path="")
                self.graph.add_edge(
                    file_path,
                    mod,
                    relation="imports",
                    source_file=file_path,
                )

        # ── Step 3: inheritance edges (class → base_class) ───────────────
        for chunk in chunks:
            if chunk.get("node_type") != "class":
                continue
            metadata = chunk.get("metadata", {})
            class_name = metadata.get("class_name") or ""
            base_classes: List[str] = metadata.get("base_classes") or []
            if not class_name:
                continue
            for base in base_classes:
                # Strip subscript generics, e.g. "Generic[T]" → "Generic"
                base_clean = re.split(r"[\[\(,\s]", base)[0].strip()
                if not base_clean:
                    continue
                if not self.graph.has_node(base_clean):
                    self.graph.add_node(base_clean, node_type="class", file_path="")
                self.graph.add_edge(
                    class_name,
                    base_clean,
                    relation="inherits",
                    source_file=chunk.get("file_path", ""),
                )

        # ── Step 4: call edges (caller → callee) ─────────────────────────
        known_names: Set[str] = set(symbol_table.all_names())
        for chunk in chunks:
            node_type = chunk.get("node_type", "")
            if node_type not in ("function", "method"):
                continue

            metadata = chunk.get("metadata", {})
            caller_name = (
                metadata.get("func_name")
                or _infer_caller_name(chunk.get("code_content", ""))
            )
            if not caller_name:
                continue

            callees = _find_calls_in_code(chunk.get("code_content", ""), known_names)
            for callee in callees:
                # Avoid self-loops (recursive functions)
                if callee == caller_name:
                    continue
                if not self.graph.has_node(caller_name):
                    self.graph.add_node(
                        caller_name,
                        node_type=node_type,
                        file_path=chunk.get("file_path", ""),
                    )
                self.graph.add_edge(
                    caller_name,
                    callee,
                    relation="calls",
                    source_file=chunk.get("file_path", ""),
                )

        return self

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_dependencies(self, node_id: str, relation: Optional[str] = None) -> List[str]:
        """
        Return the direct successors of *node_id* (outgoing edges).

        Args:
            node_id:  Node identifier (symbol name or file path).
            relation: If given, filter edges by ``relation`` attribute
                      (``"imports"`` | ``"inherits"`` | ``"calls"``).

        Returns:
            List of successor node IDs.
        """
        if not self.graph.has_node(node_id):
            return []
        successors = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                successors.append(target)
        return successors

    def get_callers(self, node_id: str) -> List[str]:
        """
        Return all symbols that call *node_id* (incoming ``calls`` edges).

        Args:
            node_id: Symbol name to look up callers for.

        Returns:
            List of caller node IDs.
        """
        if not self.graph.has_node(node_id):
            return []
        return [
            src
            for src, _, data in self.graph.in_edges(node_id, data=True)
            if data.get("relation") == "calls"
        ]

    def get_inheritors(self, class_name: str) -> List[str]:
        """
        Return all classes that directly inherit from *class_name*.

        Args:
            class_name: Name of the base class to query.

        Returns:
            List of child class names.
        """
        if not self.graph.has_node(class_name):
            return []
        return [
            src
            for src, _, data in self.graph.in_edges(class_name, data=True)
            if data.get("relation") == "inherits"
        ]

    def get_file_imports(self, file_path: str) -> List[str]:
        """
        Return all module names imported by *file_path*.

        Args:
            file_path: Absolute source file path used as a node ID.

        Returns:
            List of imported module name strings.
        """
        return self.get_dependencies(file_path, relation="imports")

    def node_count(self) -> int:
        """Return total number of nodes in the graph."""
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        """Return total number of edges in the graph."""
        return self.graph.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"DependencyGraph("
            f"{self.node_count()} nodes, "
            f"{self.edge_count()} edges)"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the DependencyGraph to a pickle file.

        We use Python's native ``pickle`` rather than NetworkX's
        ``write_gpickle`` (deprecated in NetworkX ≥ 3.x) for maximum
        compatibility and version stability.

        Args:
            path: Destination file path (e.g. ``.noesis/dependency_graph.pkl``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "DependencyGraph":
        """
        Deserialize a DependencyGraph from a pickle file.

        Args:
            path: Path to the ``.noesis/dependency_graph.pkl`` file.

        Returns:
            A :class:`DependencyGraph` instance with the loaded graph.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dependency graph not found at '{path}'. "
                "Run 'noesiscli analyze <path>' first."
            )
        with open(path, "rb") as fh:
            graph = pickle.load(fh)

        instance = cls()
        instance.graph = graph
        return instance


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _infer_caller_name(code_content: str) -> Optional[str]:
    """
    Infer the name of a function/method from its raw source code when the
    metadata ``func_name`` key is absent.
    """
    if not code_content:
        return None
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*[\(:]", code_content)
    return m.group(1) if m else None
