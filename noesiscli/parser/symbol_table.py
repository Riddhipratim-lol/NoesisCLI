"""
Global Symbol Table Builder — Phase 4.1

Extracts declarations of all classes, methods, and functions from the
structured Code Chunks produced by the Tree-sitter parser, mapping each
symbol name to one or more SymbolDefinition records.

The registry supports:
  - Exact (case-sensitive) lookup by symbol name.
  - Case-insensitive fuzzy lookup for user-facing queries.
  - Pickle-based persistence to `.noesis/symbol_table.pkl`.

Data flow:
  Input:  List of Code Chunk dicts (Phase 1.2 / Phase 5.1 output).
  Output: SymbolTable instance (in-memory dict + serializable to disk).
          Consumed by DependencyGraph (Phase 4.2) and Context Pruner (Phase 6.1).
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Symbol node types we index
# ---------------------------------------------------------------------------
_INDEXABLE_TYPES = {"class", "method", "function"}


# ---------------------------------------------------------------------------
# SymbolDefinition — a single resolved symbol record
# ---------------------------------------------------------------------------

@dataclass
class SymbolDefinition:
    """
    Represents one declaration of a named code symbol in the repository.

    Attributes:
        symbol_name:  The simple identifier name (e.g. ``"authenticate"``).
        node_type:    One of ``"class"``, ``"method"``, or ``"function"``.
        file_path:    Absolute path of the source file.
        start_line:   1-indexed line where the symbol starts.
        end_line:     1-indexed line where the symbol ends.
        parent_class: Enclosing class name, or ``None`` for module-level symbols.
        signature:    The ``def``/``class`` signature line (best-effort, no body).
        docstring:    The symbol's docstring if present.
        is_async:     Whether the function/method is ``async def``.
        decorators:   List of decorator strings (e.g. ``["@staticmethod"]``).
        base_classes: For class symbols — list of base class name strings.
    """

    symbol_name: str
    node_type: str          # "class" | "method" | "function"
    file_path: str
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def qualified_name(self) -> str:
        """Return ``ClassName.method_name`` for methods, bare name otherwise."""
        if self.parent_class:
            return f"{self.parent_class}.{self.symbol_name}"
        return self.symbol_name

    def __repr__(self) -> str:
        return (
            f"SymbolDefinition({self.node_type} '{self.qualified_name}' "
            f"@ {os.path.basename(self.file_path)}:{self.start_line})"
        )


# ---------------------------------------------------------------------------
# Signature extraction helper
# ---------------------------------------------------------------------------

def _extract_signature(code_content: str, node_type: str) -> Optional[str]:
    """
    Extract the first (header) line of a function or class definition.

    For functions/methods this is the ``def foo(...)`` line (possibly with
    a return annotation).  For classes it is the ``class Foo(Bar):`` line.
    We strip everything after the body-opening colon so the signature remains
    a single tidy line.
    """
    if not code_content:
        return None

    lines = code_content.splitlines()
    # Skip leading decorator lines
    for i, line in enumerate(lines):
        stripped = line.strip()
        if node_type == "class" and stripped.startswith("class "):
            return stripped.rstrip(":")
        if node_type in ("function", "method") and (
            stripped.startswith("def ") or stripped.startswith("async def ")
        ):
            # May span multiple lines (e.g. long arg list) — collect until ':'
            sig_lines = [stripped]
            for cont in lines[i + 1:]:
                stripped_cont = cont.strip()
                sig_lines.append(stripped_cont)
                if stripped_cont.endswith(":"):
                    break
            return " ".join(sig_lines).rstrip(":")
    # Fallback: first non-empty line
    for line in lines:
        if line.strip():
            return line.strip().rstrip(":")
    return None


# ---------------------------------------------------------------------------
# SymbolTable — registry
# ---------------------------------------------------------------------------

class SymbolTable:
    """
    Global Symbol Table Registry.

    Maps symbol names (classes, methods, functions) to a list of their
    :class:`SymbolDefinition` records.  One name may resolve to multiple
    definitions when the same symbol is declared in different files (e.g.
    overloads, re-exports, or genuinely distinct classes with the same name).

    Attributes:
        _registry:         ``{symbol_name: [SymbolDefinition, ...]}``.
        _qualified_index:  ``{qualified_name: [SymbolDefinition, ...]}``.
                           Enables ``ClassName.method`` lookups.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, List[SymbolDefinition]] = {}
        self._qualified_index: Dict[str, List[SymbolDefinition]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[dict]) -> "SymbolTable":
        """
        Populate the registry from a list of Code Chunk dicts.

        Only chunks with ``node_type`` in ``{"class", "method", "function"}``
        are indexed.  All other chunk types (module, imports, global, etc.)
        are silently skipped.

        Args:
            chunks: Flat list of Code Chunk dicts produced by
                    :class:`~noesiscli.parser.tree_sitter_parser.TreeSitterParser`.

        Returns:
            ``self`` to allow chaining (``SymbolTable().build(chunks)``).
        """
        self._registry.clear()
        self._qualified_index.clear()

        for chunk in chunks:
            node_type = chunk.get("node_type", "")
            if node_type not in _INDEXABLE_TYPES:
                continue

            metadata = chunk.get("metadata", {})
            code_content = chunk.get("code_content", "")

            # Determine symbol name -----------------------------------------
            # For functions/methods the parser stores 'func_name' in metadata.
            # For classes it stores 'class_name'.  Fall back to regex on
            # code_content if the metadata key is absent.
            if node_type in ("function", "method"):
                symbol_name = metadata.get("func_name") or _infer_name_from_code(
                    code_content, node_type
                )
            else:
                symbol_name = metadata.get("class_name") or _infer_name_from_code(
                    code_content, node_type
                )

            if not symbol_name:
                continue

            defn = SymbolDefinition(
                symbol_name=symbol_name,
                node_type=node_type,
                file_path=chunk.get("file_path", ""),
                start_line=chunk.get("start_line", 0),
                end_line=chunk.get("end_line", 0),
                parent_class=metadata.get("parent_class"),
                signature=_extract_signature(code_content, node_type),
                docstring=metadata.get("docstring"),
                is_async=metadata.get("is_async", False),
                decorators=list(metadata.get("decorators", [])),
                base_classes=list(metadata.get("base_classes", [])),
            )

            # Index by simple name
            self._registry.setdefault(symbol_name, []).append(defn)

            # Index by qualified name (ClassName.method_name)
            qname = defn.qualified_name
            if qname != symbol_name:
                self._qualified_index.setdefault(qname, []).append(defn)

        return self

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        name: str,
        case_sensitive: bool = True,
    ) -> List[SymbolDefinition]:
        """
        Look up a symbol by name.

        Searches both the simple-name registry and the qualified-name index
        (e.g. ``"UserService.authenticate"``).

        Args:
            name:           The symbol name to look up (simple or qualified).
            case_sensitive: When ``False``, all comparisons are lowercased.
                            Useful for user-facing queries that may not match
                            the exact casing of the source code.

        Returns:
            A list of matching :class:`SymbolDefinition` records (may be
            empty if the symbol is not found).
        """
        if case_sensitive:
            results = self._registry.get(name, [])
            results = results + self._qualified_index.get(name, [])
            # De-duplicate (same object may appear in both indexes)
            seen = set()
            deduped = []
            for r in results:
                key = (r.file_path, r.start_line, r.node_type)
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            return deduped
        else:
            target = name.lower()
            results = []
            seen: set = set()
            for key, defs in self._registry.items():
                if key.lower() == target:
                    for d in defs:
                        k = (d.file_path, d.start_line, d.node_type)
                        if k not in seen:
                            seen.add(k)
                            results.append(d)
            for key, defs in self._qualified_index.items():
                if key.lower() == target:
                    for d in defs:
                        k = (d.file_path, d.start_line, d.node_type)
                        if k not in seen:
                            seen.add(k)
                            results.append(d)
            return results

    def all_names(self) -> List[str]:
        """Return all registered simple symbol names."""
        return list(self._registry.keys())

    def all_definitions(self) -> List[SymbolDefinition]:
        """Return a flat list of every SymbolDefinition in the registry."""
        result = []
        for defs in self._registry.values():
            result.extend(defs)
        return result

    def __len__(self) -> int:
        return sum(len(v) for v in self._registry.values())

    def __repr__(self) -> str:
        return f"SymbolTable({len(self._registry)} unique names, {len(self)} definitions)"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the SymbolTable to a pickle file.

        Args:
            path: Destination file path (e.g. ``.noesis/symbol_table.pkl``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "registry": self._registry,
            "qualified_index": self._qualified_index,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "SymbolTable":
        """
        Deserialize a SymbolTable from a pickle file.

        Args:
            path: Path to the ``.noesis/symbol_table.pkl`` file.

        Returns:
            A fully populated :class:`SymbolTable` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Symbol table not found at '{path}'. "
                "Run 'noesiscli analyze <path>' first."
            )
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        table = cls()
        table._registry = payload.get("registry", {})
        table._qualified_index = payload.get("qualified_index", {})
        return table


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _infer_name_from_code(code_content: str, node_type: str) -> Optional[str]:
    """
    Best-effort symbol name inference from raw code_content when the parser
    metadata does not supply a ``func_name`` / ``class_name`` key.

    Handles:
      - ``def foo(...)`` / ``async def foo(...)``
      - ``class Foo(...)``
    """
    if not code_content:
        return None
    if node_type in ("function", "method"):
        m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*[\(:]", code_content)
        if m:
            return m.group(1)
    elif node_type == "class":
        m = re.search(r"\bclass\s+([A-Za-z_]\w*)\s*[:\(]", code_content)
        if m:
            return m.group(1)
    return None
