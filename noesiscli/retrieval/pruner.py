"""
Context-Aware Pruning & Prompt Construction — Phase 6

Three components:

  6.1  DependencyContextResolver
       Inspects retrieved chunks, queries the Symbol Table and Dependency
       Graph, and produces two sets of symbol names:
         • target_symbols  — must appear with full implementation bodies.
         • reference_symbols — appear as signatures/stubs only.

  6.2  CodeStructurePruner
       Reconstructs a minimal skeletal representation of every source file
       that is touched by the retrieved chunks.  Full bodies are kept only
       for target_symbols; everything else in the same file is stubbed to
       its signature + "..." placeholder.  Pre-built class_header chunks are
       leveraged when available so Tree-sitter is only invoked as a fallback.

  6.3  PromptConstructor
       Assembles the final context-optimised prompt that is sent to the LLM.
       Combines pruned file blocks, dependency metadata, file locations, basic
       chunk metadata, and the user's original query.

Data flow:
  HybridRetriever (Phase 3.2)
        ↓  ranked Code Chunk dicts
  DependencyContextResolver (6.1)
        ↓  target_symbols, reference_symbols, file_chunks_map
  CodeStructurePruner (6.2)
        ↓  pruned_blocks  (list of PrunedBlock namedtuples)
  PromptConstructor (6.3)
        ↓  populated prompt string
  LLM Reasoner (Phase 7.1)
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from noesiscli.parser.symbol_table import SymbolTable, SymbolDefinition
from noesiscli.parser.dependency_graph import DependencyGraph


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sig_from_code(code_content: str, node_type: str) -> str:
    """
    Extract only the first meaningful line of a function or class definition
    from raw source code and return a stub with ``...`` as the body.

    Examples:
        ``def authenticate(self, user_id: int) -> bool:``
          → ``def authenticate(self, user_id: int) -> bool:\n    ...``
        ``class UserService(BaseService):``
          → ``class UserService(BaseService):\n    ...``
    """
    if not code_content:
        return "..."

    lines = code_content.splitlines()
    # Collect decorator lines that precede the def/class keyword
    decorator_lines: List[str] = []
    sig_lines: List[str] = []
    collecting_sig = False

    for line in lines:
        stripped = line.strip()
        if not collecting_sig:
            if stripped.startswith("@"):
                decorator_lines.append(line.rstrip())
            elif stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                collecting_sig = True
                sig_lines.append(line.rstrip())
                # Single-line signature ends with ":"
                if stripped.endswith(":"):
                    break
            # Skip everything else until we hit the def/class
        else:
            sig_lines.append(line.rstrip())
            if stripped.endswith(":"):
                break

    if not sig_lines:
        return "..."

    # Determine indentation from the first sig line
    first_line = sig_lines[0]
    indent = len(first_line) - len(first_line.lstrip())
    stub_indent = " " * (indent + 4)

    parts = decorator_lines + sig_lines + [stub_indent + "..."]
    return "\n".join(parts)


def _extract_base_classes(code_content: str) -> List[str]:
    """
    Extract base class names from a class definition line.

    Examples:
        ``class Foo(Bar, Baz):`` → ``["Bar", "Baz"]``
        ``class Foo:``            → ``[]``
    """
    m = re.search(r"class\s+\w+\s*\(([^)]+)\)\s*:", code_content)
    if not m:
        return []
    raw = m.group(1)
    bases = [b.strip().split("[")[0].strip() for b in raw.split(",")]
    return [b for b in bases if b]


# ===========================================================================
# 6.1 — Dependency Context Resolver
# ===========================================================================

class DependencyContextResolver:
    """
    Phase 6.1 — Dependency Context Resolver.

    Given the ranked candidate chunks from the Hybrid Retriever, the Global
    Symbol Table (Phase 4.1), and the Codebase Dependency Graph (Phase 4.2),
    this component identifies:

    * **target_symbols**    — symbols whose *full* implementation bodies must
                              be preserved in the pruned context.
    * **reference_symbols** — symbols that are related (called, inherited from,
                              or imported in the same files) but whose bodies
                              can be replaced with signatures/stubs to save
                              tokens while retaining architectural insight.

    The resolver also returns a *file_chunks_map* that groups every chunk in
    the same source files as the retrieved set, providing the Pruner with all
    the raw material it needs to reconstruct skeletal file views.

    Args:
        symbol_table:  Populated :class:`~noesiscli.parser.symbol_table.SymbolTable`.
        dep_graph:     Populated :class:`~noesiscli.parser.dependency_graph.DependencyGraph`.
        max_call_depth: Maximum depth for recursive call-chain expansion.
                        Depth 1 means only direct callees of the retrieved
                        chunks are added; depth 2 also adds their callees etc.
                        Keep low (1–2) to avoid context explosion.
    """

    def __init__(
        self,
        symbol_table: Optional[SymbolTable] = None,
        dep_graph: Optional[DependencyGraph] = None,
        max_call_depth: int = 1,
    ) -> None:
        self.symbol_table = symbol_table
        self.dep_graph = dep_graph
        self.max_call_depth = max_call_depth

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        all_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Set[str], Set[str], Dict[str, List[Dict[str, Any]]]]:
        """
        Resolve dependency context for the retrieved chunks.

        Args:
            retrieved_chunks: Ranked list of Code Chunk dicts from the Hybrid
                              Retriever.
            all_chunks:       (Optional) Full flat list of *all* parsed chunks
                              for the repository.  When provided, the resolver
                              builds a ``file_chunks_map`` so the Pruner can
                              reconstruct complete skeletal file views.

        Returns:
            A 3-tuple ``(target_symbols, reference_symbols, file_chunks_map)``:

            * ``target_symbols``  – ``set[str]`` of qualified symbol names
              (``"ClassName.method_name"`` or bare ``"func_name"``) whose
              full bodies must be kept.
            * ``reference_symbols`` – ``set[str]`` of qualified names to
              include as stubs only.
            * ``file_chunks_map`` – ``{file_path: [chunk, ...]}`` grouping
              every available chunk by file.  Only files touched by the
              retrieved set are included unless *all_chunks* contains more.
        """
        target_symbols: Set[str] = set()
        reference_symbols: Set[str] = set()

        # ── Step 1: Every retrieved chunk is a target ────────────────────
        retrieved_files: Set[str] = set()
        retrieved_names: Set[str] = set()

        for chunk in retrieved_chunks:
            node_type = chunk.get("node_type", "")
            meta = chunk.get("metadata", {})
            file_path = chunk.get("file_path", "")
            retrieved_files.add(file_path)

            if node_type in ("function", "method"):
                func_name = meta.get("func_name") or _infer_name_from_code(
                    chunk.get("code_content", ""), node_type
                )
                if func_name:
                    parent = meta.get("parent_class")
                    qname = f"{parent}.{func_name}" if parent else func_name
                    target_symbols.add(qname)
                    retrieved_names.add(func_name)

            elif node_type == "class":
                class_name = meta.get("class_name") or _infer_name_from_code(
                    chunk.get("code_content", ""), node_type
                )
                if class_name:
                    target_symbols.add(class_name)
                    retrieved_names.add(class_name)

            elif node_type in ("module", "imports", "constant", "type_alias", "global"):
                # Not a callable symbol — mark file as touched but no symbol name
                pass

        # ── Step 2: Expand via Dependency Graph ──────────────────────────
        if self.dep_graph is not None:
            frontier = set(retrieved_names)
            for _depth in range(self.max_call_depth):
                next_frontier: Set[str] = set()
                for name in frontier:
                    callees = self.dep_graph.get_dependencies(name, relation="calls")
                    for callee in callees:
                        if callee not in target_symbols and callee not in reference_symbols:
                            reference_symbols.add(callee)
                            next_frontier.add(callee)

                    # Inheritance: parent classes are reference symbols
                    bases = self.dep_graph.get_dependencies(name, relation="inherits")
                    for base in bases:
                        if base not in target_symbols:
                            reference_symbols.add(base)

                frontier = next_frontier
                if not frontier:
                    break

        # ── Step 3: Resolve reference symbols via Symbol Table ───────────
        if self.symbol_table is not None:
            expanded_refs: Set[str] = set()
            for sym in list(reference_symbols):
                # Look up bare name and qualified name
                bare_name = sym.split(".")[-1] if "." in sym else sym
                defs = self.symbol_table.lookup(bare_name)
                for defn in defs:
                    qname = defn.qualified_name
                    if qname not in target_symbols:
                        expanded_refs.add(qname)
                        retrieved_files.add(defn.file_path)

            # Replace raw reference set with resolved qualified names
            reference_symbols = expanded_refs

        # Remove any reference that was already promoted to target
        reference_symbols -= target_symbols

        # ── Step 4: Build file_chunks_map ────────────────────────────────
        file_chunks_map: Dict[str, List[Dict[str, Any]]] = {}
        source_chunks = all_chunks if all_chunks else retrieved_chunks
        for chunk in source_chunks:
            fp = chunk.get("file_path", "")
            if fp in retrieved_files:
                file_chunks_map.setdefault(fp, []).append(chunk)

        return target_symbols, reference_symbols, file_chunks_map


# ===========================================================================
# 6.2 — Code Structure Pruner
# ===========================================================================

@dataclass
class PrunedBlock:
    """
    A pruned representation of a source file (or logical file section).

    Attributes:
        file_path:       Absolute path of the source file.
        pruned_content:  The reconstructed, partially-stubbed source text.
        kept_symbols:    Names of symbols whose full body was preserved.
        stubbed_symbols: Names of symbols that were replaced with stubs.
    """
    file_path: str
    pruned_content: str
    kept_symbols: List[str] = field(default_factory=list)
    stubbed_symbols: List[str] = field(default_factory=list)


class CodeStructurePruner:
    """
    Phase 6.2 — Code Structure Pruner.

    Reconstructs a minimal skeletal view of every file in ``file_chunks_map``:

    * **target_symbols** keep their full ``code_content``.
    * **reference_symbols** are rendered as ``<signature>\n    ...`` stubs.
    * Everything else in the file is stubbed automatically.

    The pruner groups chunks by file, sorts them by ``start_line``, and
    assembles a coherent code listing.  Pre-computed ``class_header`` chunks
    (produced by [S2] in the parser) are preferred for classes that are not
    in the target set — they are already skeletal and require no rewriting.

    Args:
        symbol_table:  Populated SymbolTable used to resolve stub signatures.
    """

    def __init__(self, symbol_table: Optional[SymbolTable] = None) -> None:
        self.symbol_table = symbol_table

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prune(
        self,
        file_chunks_map: Dict[str, List[Dict[str, Any]]],
        target_symbols: Set[str],
        reference_symbols: Set[str],
    ) -> List[PrunedBlock]:
        """
        Produce a :class:`PrunedBlock` for every file in *file_chunks_map*.

        Args:
            file_chunks_map:   ``{file_path: [chunk, ...]}`` from the Resolver.
            target_symbols:    Qualified names to keep in full.
            reference_symbols: Qualified names to render as stubs.

        Returns:
            List of :class:`PrunedBlock` objects, one per file.
        """
        blocks: List[PrunedBlock] = []

        for file_path, chunks in file_chunks_map.items():
            block = self._prune_file(
                file_path, chunks, target_symbols, reference_symbols
            )
            blocks.append(block)

        return blocks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_file(
        self,
        file_path: str,
        chunks: List[Dict[str, Any]],
        target_symbols: Set[str],
        reference_symbols: Set[str],
    ) -> PrunedBlock:
        """
        Build a pruned view for a single file.

        Strategy:
        1. Separate chunks by type.  ``class_header`` chunks are used to
           represent non-target classes.  Full ``class`` chunks are used
           only when the class is a target.
        2. Sort remaining chunks by ``start_line``.
        3. For each chunk decide: keep full, use pre-built header, or stub.
        4. Emit a comment divider ``# --- <file_path> ---`` followed by the
           assembled content.
        """
        kept: List[str] = []
        stubbed: List[str] = []
        parts: List[str] = []

        # Index class_header chunks by class name for fast lookup
        header_chunks: Dict[str, Dict[str, Any]] = {}
        for ch in chunks:
            if ch.get("node_type") == "class_header":
                meta = ch.get("metadata", {})
                cname = meta.get("class_name") or _infer_name_from_code(
                    ch.get("code_content", ""), "class"
                )
                if cname:
                    header_chunks[cname] = ch

        # Filter out class_header and sort remaining chunks by start_line
        working_chunks = [
            ch for ch in chunks if ch.get("node_type") != "class_header"
        ]
        working_chunks.sort(key=lambda c: c.get("start_line", 0))

        for chunk in working_chunks:
            node_type = chunk.get("node_type", "")
            meta = chunk.get("metadata", {})
            code_content = chunk.get("code_content", "")

            # ── Determine the chunk's qualified symbol name ────────────
            qname = self._chunk_qname(chunk)
            bare_name = qname.split(".")[-1] if qname else ""

            # ── Decide: keep / stub / use pre-built header ─────────────
            if node_type in ("module", "imports", "constant", "type_alias", "global"):
                # Always include structural / metadata chunks verbatim
                if code_content.strip():
                    parts.append(code_content.rstrip())

            elif node_type == "class":
                class_name = meta.get("class_name") or bare_name
                if qname in target_symbols or class_name in target_symbols:
                    # Keep full class body
                    parts.append(code_content.rstrip())
                    kept.append(class_name or qname)
                else:
                    # Use pre-built class_header if available (Phase [S2])
                    if class_name and class_name in header_chunks:
                        header_content = header_chunks[class_name].get("code_content", "").rstrip()
                        parts.append(header_content)
                    else:
                        stub = _sig_from_code(code_content, "class")
                        parts.append(stub)
                    stubbed.append(class_name or qname)

            elif node_type in ("function", "method"):
                func_name = meta.get("func_name") or bare_name
                if qname in target_symbols or func_name in target_symbols:
                    parts.append(code_content.rstrip())
                    kept.append(qname or func_name)
                elif qname in reference_symbols or func_name in reference_symbols:
                    # Include as a signature stub (reference context)
                    stub = _sig_from_code(code_content, "function")
                    parts.append(stub)
                    stubbed.append(qname or func_name)
                else:
                    # Non-target, non-reference → minimal stub
                    stub = _sig_from_code(code_content, "function")
                    parts.append(stub)
                    stubbed.append(qname or func_name)

        pruned_content = "\n\n".join(p for p in parts if p)
        return PrunedBlock(
            file_path=file_path,
            pruned_content=pruned_content,
            kept_symbols=kept,
            stubbed_symbols=stubbed,
        )

    @staticmethod
    def _chunk_qname(chunk: Dict[str, Any]) -> str:
        """
        Derive the qualified name for a chunk.

        Returns ``"ClassName.method_name"`` for methods, bare name otherwise.
        """
        node_type = chunk.get("node_type", "")
        meta = chunk.get("metadata", {})
        code = chunk.get("code_content", "")

        if node_type in ("function", "method"):
            func_name = meta.get("func_name") or _infer_name_from_code(code, node_type)
            parent = meta.get("parent_class")
            return f"{parent}.{func_name}" if parent and func_name else (func_name or "")
        elif node_type == "class":
            return meta.get("class_name") or _infer_name_from_code(code, "class") or ""
        return ""


# ===========================================================================
# 6.3 — Prompt Constructor
# ===========================================================================

class PromptConstructor:
    """
    Phase 6.3 — Prompt Constructor.

    Assembles the context-optimised prompt for the LLM from:
    * Pruned file blocks  (from CodeStructurePruner).
    * Dependency relationship summary  (from DependencyContextResolver).
    * Symbol definitions  (from SymbolTable).
    * File locations and basic chunk metadata.
    * The user's original query.

    The resulting prompt is concise — only the most relevant code and its
    direct relationships are included — while still giving the LLM enough
    architectural context to reason accurately.

    Args:
        symbol_table: Optional SymbolTable for resolving symbol signatures.
        dep_graph:    Optional DependencyGraph for relationship summaries.
        max_file_blocks: Maximum number of file blocks to include in the
                         prompt to prevent runaway context sizes.
    """

    SYSTEM_INSTRUCTION = (
        "You are NoesisCLI, a professional AI coding assistant and codebase "
        "architect specialised in analysing Python repositories.\n"
        "Answer the user's question using ONLY the provided code context. "
        "When the implementation of a function is shown as '...' it means the "
        "body was intentionally omitted to save tokens — do not assume it is "
        "empty or unimplemented.\n"
        "Be precise, reference specific file paths and symbol names, and "
        "cite line numbers where relevant.\n"
        "If the context does not contain enough information to fully answer "
        "the question, say so explicitly rather than guessing."
    )

    def __init__(
        self,
        symbol_table: Optional[SymbolTable] = None,
        dep_graph: Optional[DependencyGraph] = None,
        max_file_blocks: int = 8,
    ) -> None:
        self.symbol_table = symbol_table
        self.dep_graph = dep_graph
        self.max_file_blocks = max_file_blocks

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        query: str,
        pruned_blocks: List[PrunedBlock],
        target_symbols: Set[str],
        reference_symbols: Set[str],
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Assemble the final prompt string.

        Args:
            query:            The user's raw question string.
            pruned_blocks:    Output of :class:`CodeStructurePruner`.
            target_symbols:   Symbols kept with full bodies.
            reference_symbols: Symbols kept as stubs.
            retrieved_chunks: (Optional) Original retrieved chunks for metadata.

        Returns:
            A formatted prompt string ready for LLM consumption.
        """
        sections: List[str] = []

        # ── 1. Pruned code context ────────────────────────────────────────
        code_sections = self._build_code_sections(pruned_blocks)
        if code_sections:
            sections.append("## Retrieved Code Context\n\n" + code_sections)

        # ── 2. Dependency relationships summary ──────────────────────────
        dep_section = self._build_dependency_section(target_symbols, reference_symbols)
        if dep_section:
            sections.append("## Dependency Relationships\n\n" + dep_section)

        # ── 3. Symbol definitions ─────────────────────────────────────────
        sym_section = self._build_symbol_section(target_symbols)
        if sym_section:
            sections.append("## Key Symbol Definitions\n\n" + sym_section)

        # ── 4. Chunk metadata summary ─────────────────────────────────────
        if retrieved_chunks:
            meta_section = self._build_metadata_section(retrieved_chunks)
            if meta_section:
                sections.append("## Retrieved Chunk Locations\n\n" + meta_section)

        # ── 5. User query ─────────────────────────────────────────────────
        sections.append(f"## User Query\n\n{query}")

        return "\n\n---\n\n".join(sections)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_code_sections(self, pruned_blocks: List[PrunedBlock]) -> str:
        """Format each PrunedBlock as a fenced Python code block."""
        if not pruned_blocks:
            return ""

        # Limit number of file blocks to avoid runaway context
        blocks = pruned_blocks[: self.max_file_blocks]
        parts: List[str] = []

        for block in blocks:
            if not block.pruned_content.strip():
                continue
            header = f"### `{block.file_path}`"
            kept_note = ""
            if block.kept_symbols:
                kept_note = (
                    f"\n> **Full implementations**: "
                    f"{', '.join(f'`{s}`' for s in block.kept_symbols)}"
                )
            if block.stubbed_symbols:
                kept_note += (
                    f"\n> **Stubbed to signatures**: "
                    f"{', '.join(f'`{s}`' for s in block.stubbed_symbols[:8])}"
                    + (" ..." if len(block.stubbed_symbols) > 8 else "")
                )
            code_block = f"```python\n{block.pruned_content}\n```"
            parts.append(f"{header}{kept_note}\n\n{code_block}")

        return "\n\n".join(parts)

    def _build_dependency_section(
        self, target_symbols: Set[str], reference_symbols: Set[str]
    ) -> str:
        """Summarise call-chain and inheritance relationships."""
        if self.dep_graph is None or not target_symbols:
            return ""

        lines: List[str] = []
        for sym in sorted(target_symbols):
            bare = sym.split(".")[-1]
            callees = self.dep_graph.get_dependencies(bare, relation="calls")
            bases = self.dep_graph.get_dependencies(bare, relation="inherits")

            if callees:
                lines.append(
                    f"- `{sym}` **calls**: "
                    + ", ".join(f"`{c}`" for c in callees[:6])
                    + (" ..." if len(callees) > 6 else "")
                )
            if bases:
                lines.append(
                    f"- `{sym}` **inherits from**: "
                    + ", ".join(f"`{b}`" for b in bases)
                )

        return "\n".join(lines) if lines else ""

    def _build_symbol_section(self, target_symbols: Set[str]) -> str:
        """List resolved SymbolDefinition records for target symbols."""
        if self.symbol_table is None or not target_symbols:
            return ""

        lines: List[str] = []
        seen: Set[Tuple[str, int]] = set()

        for sym in sorted(target_symbols):
            bare = sym.split(".")[-1]
            defs = self.symbol_table.lookup(bare)
            for defn in defs:
                key = (defn.file_path, defn.start_line)
                if key in seen:
                    continue
                seen.add(key)
                sig = defn.signature or f"{defn.node_type} {defn.qualified_name}"
                loc = f"`{defn.file_path}` L{defn.start_line}–{defn.end_line}"
                lines.append(f"- **`{defn.qualified_name}`** ({defn.node_type}) at {loc}\n  ```python\n  {sig}\n  ```")

        return "\n".join(lines) if lines else ""

    def _build_metadata_section(
        self, retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """List file locations and chunk types for the retrieved set."""
        lines: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            fp = chunk.get("file_path", "?")
            sl = chunk.get("start_line", "?")
            el = chunk.get("end_line", "?")
            nt = chunk.get("node_type", "?")
            rrf = chunk.get("rrf_score")
            score_str = f" [RRF: {rrf:.4f}]" if rrf is not None else ""
            lines.append(f"{idx}. `{fp}` L{sl}–{el} ({nt}){score_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry-point used by the RAG pipeline node
# ---------------------------------------------------------------------------

def build_pruned_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    symbol_table: Optional[SymbolTable] = None,
    dep_graph: Optional[DependencyGraph] = None,
    all_chunks: Optional[List[Dict[str, Any]]] = None,
    max_call_depth: int = 1,
    top_k_files: int = 8,
) -> Tuple[str, str]:
    """
    Convenience function that wires Phase 6.1 → 6.2 → 6.3 end-to-end.

    Args:
        query:            User query string.
        retrieved_chunks: Ranked Code Chunk dicts from HybridRetriever.
        symbol_table:     Optional loaded SymbolTable.
        dep_graph:        Optional loaded DependencyGraph.
        all_chunks:       Optional full list of all parsed chunks (used to
                          reconstruct complete file views).
        max_call_depth:   How many call hops to follow (default 1).
        top_k_files:      Maximum file blocks in the prompt.

    Returns:
        A 2-tuple ``(prompt_str, system_instruction)`` ready for LLM inference.
    """
    # 6.1 — Resolve dependencies
    resolver = DependencyContextResolver(
        symbol_table=symbol_table,
        dep_graph=dep_graph,
        max_call_depth=max_call_depth,
    )
    target_symbols, reference_symbols, file_chunks_map = resolver.resolve(
        retrieved_chunks, all_chunks=all_chunks
    )

    # 6.2 — Prune file structures
    pruner = CodeStructurePruner(symbol_table=symbol_table)
    pruned_blocks = pruner.prune(file_chunks_map, target_symbols, reference_symbols)

    # 6.3 — Build prompt
    constructor = PromptConstructor(
        symbol_table=symbol_table,
        dep_graph=dep_graph,
        max_file_blocks=top_k_files,
    )
    prompt_str = constructor.build(
        query=query,
        pruned_blocks=pruned_blocks,
        target_symbols=target_symbols,
        reference_symbols=reference_symbols,
        retrieved_chunks=retrieved_chunks,
    )

    return prompt_str, PromptConstructor.SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# Private helper (mirrors symbol_table._infer_name_from_code for local use)
# ---------------------------------------------------------------------------

def _infer_name_from_code(code_content: str, node_type: str) -> Optional[str]:
    """Best-effort name inference from raw code."""
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
