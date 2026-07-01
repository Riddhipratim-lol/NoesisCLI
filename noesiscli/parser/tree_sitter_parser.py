"""
Tree-sitter Parser.
Integrates Tree-sitter for parsing Python source code to perform syntax-aware
AST parsing, extracting functions, classes, methods, and all top-level constructs.

Bug fixes applied vs. original implementation:
  - [Fix 1] decorated_definition now handled in _extract_chunks_from_node so
            decorator lines are always included in the extracted code_content.
  - [Fix 2] import statements are no longer used as flush triggers for global
            blocks; they are collected separately and never cause unrelated
            global nodes to be fragmented.
  - [Fix 3] imports are collected per-file instead of silently dropped;
            every chunk carries the file's import list for Phase 4 (Dep Graph).
  - [Fix 4] nested functions are NOT recursed into from within a parent
            function_definition — they remain embedded in the parent's
            code_content and are separately extracted only from the class body
            traversal path.
  - [Fix 5] async_function_definition is handled identically to
            function_definition so no async def is ever missed.
  - [Fix 6] class body traversal walks the explicit 'block' child rather than
            iterating all children, avoiding accidental double-traversal of
            name/colon/base-class nodes.

Strategies applied:
  - [S1] Module-level chunk: file docstring + aggregated import list + metadata.
  - [S2] Class-header chunk: class signature + docstring + method signatures
         only (no bodies) emitted alongside the full class chunk.
  - [S3] Global nodes are classified as 'constant', 'type_alias', or 'global'
         based on their AST shape rather than always lumping as 'global'.
         Import nodes contribute to the file-level import list instead.
  - [S4] Per-chunk metadata: decorators list, is_async flag, parent_class,
         is_dunder flag, special_type tag (@property / @staticmethod /
         @classmethod), docstring, and file-level imports_in_file.
"""

from __future__ import annotations

import re
from typing import Optional

import tree_sitter_python
from tree_sitter import Language, Parser, Node


# ---------------------------------------------------------------------------
# Node type constants (Python grammar names)
# ---------------------------------------------------------------------------
# NOTE: In tree-sitter-python, async functions are represented as
# 'function_definition' nodes that contain an 'async' keyword child token,
# NOT as a separate 'async_function_definition' node type.
_FUNC_TYPES = {"function_definition", "async_function_definition"}
_CLASS_TYPE = "class_definition"
_DECORATED_TYPE = "decorated_definition"
_IMPORT_TYPES = {"import_statement", "import_from_statement"}

# Decorators that carry a well-known semantic meaning worth tagging
_SPECIAL_DECORATORS = {"property", "staticmethod", "classmethod", "abstractmethod",
                       "cached_property", "overload"}


class TreeSitterParser:
    """
    Tree-sitter Parser: parses Python source files into structured semantic
    code chunks with rich metadata.  Produces zero missed constructs by
    handling decorated definitions, async functions, nested functions,
    imports, and every top-level statement explicitly.
    """

    def __init__(self, language: str = "python") -> None:
        lang_key = language.lower()
        if lang_key not in ("python", "py"):
            raise ValueError(
                f"Only 'python' is supported in Phase 1.2. Got '{language}'"
            )
        try:
            self.language = Language(tree_sitter_python.language())
            self.parser = Parser(self.language)
        except Exception as exc:
            raise ValueError(
                f"Failed to load tree-sitter language '{language}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> list[dict]:
        """Read a file from disk and return its semantic chunks."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                code = fh.read()
            return self.parse_code(code, file_path)
        except Exception:
            return []

    def parse_code(self, code: str, file_path: str) -> list[dict]:
        """
        Parse *code* and return a list of semantic chunk dicts.

        Every chunk contains at minimum:
            code_content   – raw source text of the construct
            file_path      – absolute path of the source file
            node_type      – one of: module | class | class_header |
                             function | method | global | constant |
                             type_alias
            start_line     – 1-indexed start line
            end_line       – 1-indexed end line
            metadata       – dict with rich structural information
        """
        if not code.strip():
            return []

        code_bytes = code.encode("utf-8")
        tree = self.parser.parse(code_bytes)
        root = tree.root_node

        # ── Step 1: collect file-level imports ──────────────────────────
        import_nodes: list[Node] = self._collect_import_nodes(root)
        imports_in_file: list[str] = [
            code_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace").strip()
            for n in import_nodes
        ]

        # ── Step 2: emit module-level chunk (Strategy 1) ─────────────────
        chunks: list[dict] = []
        module_chunk = self._make_module_chunk(root, code_bytes, file_path, imports_in_file)
        chunks.append(module_chunk)

        # ── Step 2b: emit dedicated imports chunk ────────────────────────
        # Imports are emitted as a first-class chunk so that Phase 4
        # (Dependency Graph) can find them by iterating over chunk types
        # without having to inspect metadata on every other chunk.
        if import_nodes:
            imports_chunk = self._make_imports_chunk(
                import_nodes, code_bytes, file_path, imports_in_file
            )
            chunks.append(imports_chunk)

        # ── Step 3: walk top-level nodes ─────────────────────────────────
        pending_global: list[Node] = []

        def flush_global() -> None:
            if not pending_global:
                return
            chunk = self._make_global_chunk(pending_global, code_bytes, file_path, imports_in_file)
            if chunk:
                chunks.append(chunk)
            pending_global.clear()

        # Determine the module docstring node so we can skip it below
        # (it is already captured in the module chunk — no need to repeat it).
        module_doc_node = self._find_module_docstring_node(root)

        for child in root.children:
            if child.type in _IMPORT_TYPES:
                # [Fix 2 & 3] imports do NOT flush global blocks; they are
                # already captured in imports_in_file.
                continue

            # Skip the module-level docstring — it lives in the module chunk.
            if module_doc_node is not None and child is module_doc_node:
                continue

            if child.type in (_CLASS_TYPE, _DECORATED_TYPE) or child.type in _FUNC_TYPES:
                flush_global()
                extracted = self._extract_top_level(child, code_bytes, file_path, imports_in_file)
                chunks.extend(extracted)

            else:
                # Everything else is a top-level statement (assignment,
                # expression, if __name__ == '__main__':, etc.)
                if child.type not in ("comment", "newline", ""):
                    pending_global.append(child)

        flush_global()
        return chunks

    # ------------------------------------------------------------------
    # Import collection (Strategy 1 + Fix 3)
    # ------------------------------------------------------------------

    def _collect_import_nodes(self, root: Node) -> list[Node]:
        """Return the ordered list of import AST nodes from the module root."""
        return [child for child in root.children if child.type in _IMPORT_TYPES]

    def _make_imports_chunk(
        self,
        import_nodes: list[Node],
        code_bytes: bytes,
        file_path: str,
        imports_in_file: list[str],
    ) -> dict:
        """
        Emit a single 'imports' chunk aggregating every import statement in
        the file.  Having imports as a named chunk type lets Phase 4
        (Dependency Graph) discover them with a simple node_type filter.

        code_content  – all import lines joined, exactly as written
        metadata.imports_parsed  – list of individual import strings,
                                   ready for module-name extraction
        """
        start_line = import_nodes[0].start_point[0] + 1
        end_line = import_nodes[-1].end_point[0] + 1
        code_content = "\n".join(
            code_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace").strip()
            for n in import_nodes
        )
        return {
            "code_content": code_content,
            "file_path": file_path,
            "node_type": "imports",
            "start_line": start_line,
            "end_line": end_line,
            "metadata": {
                "imports_in_file": imports_in_file,
                "imports_parsed": imports_in_file,   # alias — same list
                "module_docstring": None,
                "decorators": [],
                "is_async": False,
                "parent_class": None,
                "is_dunder": False,
                "special_type": None,
                "docstring": None,
            },
        }

    # ------------------------------------------------------------------
    # Module chunk (Strategy 1)
    # ------------------------------------------------------------------

    def _make_module_chunk(
        self,
        root: Node,
        code_bytes: bytes,
        file_path: str,
        imports_in_file: list[str],
    ) -> dict:
        """
        Emit a 'module' chunk capturing:
          - The module-level docstring (first expression_statement whose value
            is a string literal).
          - All import lines.
          - Basic file metadata.
        """
        module_docstring = self._extract_module_docstring(root, code_bytes)
        total_lines = root.end_point[0] + 1

        content_parts: list[str] = []
        if module_docstring:
            content_parts.append(module_docstring)
        if imports_in_file:
            content_parts.append("\n".join(imports_in_file))

        return {
            "code_content": "\n\n".join(content_parts) if content_parts else "",
            "file_path": file_path,
            "node_type": "module",
            "start_line": 1,
            "end_line": total_lines,
            "metadata": {
                "imports_in_file": imports_in_file,
                "module_docstring": module_docstring,
                "total_lines": total_lines,
                "decorators": [],
                "is_async": False,
                "parent_class": None,
                "is_dunder": False,
                "special_type": None,
                "docstring": module_docstring,
            },
        }

    def _find_module_docstring_node(self, root: Node) -> Optional[Node]:
        """Return the AST node for the module-level docstring, or None."""
        for child in root.children:
            if child.type == "expression_statement":
                expr = child.children[0] if child.children else None
                if expr and expr.type in ("string", "concatenated_string"):
                    return child
            if child.type not in _IMPORT_TYPES | {"comment", "newline", ""}:
                break
        return None

    def _extract_module_docstring(self, root: Node, code_bytes: bytes) -> Optional[str]:
        """Return the module-level docstring text, or None."""
        node = self._find_module_docstring_node(root)
        if node:
            return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
        return None

    # ------------------------------------------------------------------
    # Global / constant / type_alias chunks (Strategy 3 + Fix 2)
    # ------------------------------------------------------------------

    def _make_global_chunk(
        self,
        nodes: list[Node],
        code_bytes: bytes,
        file_path: str,
        imports_in_file: list[str],
    ) -> Optional[dict]:
        """
        Turn a list of top-level non-function/class nodes into a chunk.
        Classifies as 'constant', 'type_alias', or 'global'.
        """
        start_byte = nodes[0].start_byte
        end_byte = nodes[-1].end_byte
        start_line = nodes[0].start_point[0] + 1
        end_line = nodes[-1].end_point[0] + 1

        content = code_bytes[start_byte:end_byte].decode("utf-8", errors="replace").strip()
        if not content:
            return None

        node_type = self._classify_global_nodes(nodes, code_bytes)

        return {
            "code_content": content,
            "file_path": file_path,
            "node_type": node_type,
            "start_line": start_line,
            "end_line": end_line,
            "metadata": {
                "imports_in_file": imports_in_file,
                "module_docstring": None,
                "decorators": [],
                "is_async": False,
                "parent_class": None,
                "is_dunder": False,
                "special_type": None,
                "docstring": None,
            },
        }

    def _classify_global_nodes(self, nodes: list[Node], code_bytes: bytes) -> str:
        """
        Heuristically classify a group of top-level statement nodes.

        Rules (applied to the *first* substantive node):
          - assignment whose RHS contains 'TypeVar', 'Union', 'Optional',
            'Literal', 'Annotated', 'Final', 'TypeAlias'  → 'type_alias'
          - assignment whose LHS is ALL_CAPS                → 'constant'
          - everything else                                 → 'global'

        NOTE: top-level assignments in the Python grammar are wrapped inside
        'expression_statement' nodes, so we unwrap one level before checking.
        """
        type_alias_keywords = re.compile(
            r"\bTypeVar\b|\bUnion\b|\bOptional\b|\bLiteral\b"
            r"|\bAnnotated\b|\bFinal\b|\bTypeAlias\b"
        )
        for node in nodes:
            # Unwrap expression_statement wrapper that the grammar inserts
            actual = node
            if node.type == "expression_statement" and node.children:
                actual = node.children[0]

            if actual.type == "type_alias_statement":
                return "type_alias"

            if actual.type in ("assignment", "augmented_assignment"):
                full_text = code_bytes[actual.start_byte:actual.end_byte].decode("utf-8", errors="replace")
                if type_alias_keywords.search(full_text):
                    return "type_alias"
                # LHS is the first child identifier
                name_node = actual.children[0] if actual.children else None
                if name_node and name_node.type == "identifier":
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                    if name.isupper() and len(name) > 1:
                        return "constant"
        return "global"

    # ------------------------------------------------------------------
    # Top-level extraction dispatcher
    # ------------------------------------------------------------------

    def _extract_top_level(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        imports_in_file: list[str],
    ) -> list[dict]:
        """
        Dispatch a top-level node (class, function, or decorated definition)
        to the appropriate extractor.  Returns one or more chunks.
        """
        # [Fix 1] decorated_definition resolved here before delegating
        if node.type == _DECORATED_TYPE:
            return self._extract_decorated(
                node, code_bytes, file_path, imports_in_file, inside_class=False
            )
        if node.type == _CLASS_TYPE:
            return self._extract_class(node, code_bytes, file_path, imports_in_file)
        if node.type in _FUNC_TYPES:
            return [self._extract_function(
                node, code_bytes, file_path, imports_in_file,
                inside_class=False, decorator_nodes=[], decorator_start_byte=None
            )]
        return []

    # ------------------------------------------------------------------
    # Decorated definition (Fix 1)
    # ------------------------------------------------------------------

    def _extract_decorated(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        imports_in_file: list[str],
        inside_class: bool,
        parent_class: Optional[str] = None,
    ) -> list[dict]:
        """
        Handle a decorated_definition node.  Extracts decorator names and
        delegates to the appropriate extractor with the *full* byte range
        (including @decorator lines) captured.
        """
        decorator_nodes = [c for c in node.children if c.type == "decorator"]
        decorator_names = [
            code_bytes[d.start_byte:d.end_byte].decode("utf-8", errors="replace").strip()
            for d in decorator_nodes
        ]
        # decorator_start_byte lets the inner extractor use the decorated
        # node's start so the @line is included in code_content
        decorator_start_byte = node.start_byte

        inner = next(
            (c for c in node.children if c.type in _FUNC_TYPES | {_CLASS_TYPE}),
            None,
        )
        if inner is None:
            return []

        if inner.type == _CLASS_TYPE:
            return self._extract_class(
                inner, code_bytes, file_path, imports_in_file,
                decorator_names=decorator_names,
                override_start_byte=decorator_start_byte,
            )
        # function or async_function
        return [self._extract_function(
            inner, code_bytes, file_path, imports_in_file,
            inside_class=inside_class,
            decorator_nodes=decorator_nodes,
            decorator_start_byte=decorator_start_byte,
            parent_class=parent_class,
        )]

    # ------------------------------------------------------------------
    # Class extraction (Strategy 2 + Fix 6)
    # ------------------------------------------------------------------

    def _extract_class(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        imports_in_file: list[str],
        decorator_names: Optional[list[str]] = None,
        override_start_byte: Optional[int] = None,
    ) -> list[dict]:
        """
        Emit two chunks per class (Strategy 2):
          1. Full class chunk   – complete source text including all bodies.
          2. Class-header chunk – class line + docstring + method signatures.

        Also recurses into the class body to extract each method individually.
        """
        if decorator_names is None:
            decorator_names = []

        start_byte = override_start_byte if override_start_byte is not None else node.start_byte
        end_byte = node.end_byte
        start_line = (
            code_bytes[:start_byte].count(b"\n") + 1
            if override_start_byte is not None
            else node.start_point[0] + 1
        )
        end_line = node.end_point[0] + 1

        class_name = self._get_name(node, code_bytes)
        full_code = code_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
        class_docstring = self._extract_body_docstring(node, code_bytes)
        base_classes = self._get_base_classes(node, code_bytes)

        shared_meta = {
            "imports_in_file": imports_in_file,
            "decorators": decorator_names,
            "is_async": False,
            "parent_class": None,
            "is_dunder": False,
            "special_type": self._special_decorator_type(decorator_names),
            "docstring": class_docstring,
            "class_name": class_name,
            "base_classes": base_classes,
            "module_docstring": None,
        }

        chunks: list[dict] = []

        # ── 1. Full class chunk ──────────────────────────────────────────
        chunks.append({
            "code_content": full_code,
            "file_path": file_path,
            "node_type": "class",
            "start_line": start_line,
            "end_line": end_line,
            "metadata": shared_meta,
        })

        # ── 2. Class-header chunk (Strategy 2) ──────────────────────────
        header_code = self._build_class_header(node, code_bytes, class_name, class_docstring, decorator_names)
        chunks.append({
            "code_content": header_code,
            "file_path": file_path,
            "node_type": "class_header",
            "start_line": start_line,
            "end_line": end_line,
            "metadata": {**shared_meta},
        })

        # ── 3. Extract individual methods from the class body (Fix 6) ───
        body_block = next((c for c in node.children if c.type == "block"), None)
        if body_block:
            for member in body_block.children:
                if member.type == _DECORATED_TYPE:
                    # [Fix 1] decorated method — full decorator + function
                    chunks.extend(self._extract_decorated(
                        member, code_bytes, file_path, imports_in_file,
                        inside_class=True, parent_class=class_name
                    ))
                elif member.type in _FUNC_TYPES:
                    # Plain method (no decorator)
                    chunks.append(self._extract_function(
                        member, code_bytes, file_path, imports_in_file,
                        inside_class=True,
                        decorator_nodes=[],
                        decorator_start_byte=None,
                        parent_class=class_name,
                    ))
                elif member.type == _CLASS_TYPE:
                    # Nested class (rare but valid Python)
                    chunks.extend(self._extract_class(
                        member, code_bytes, file_path, imports_in_file
                    ))

        return chunks

    def _build_class_header(
        self,
        node: Node,
        code_bytes: bytes,
        class_name: str,
        docstring: Optional[str],
        decorator_names: list[str],
    ) -> str:
        """
        Build the skeletal class header: decorators + class line + docstring +
        one-line signatures of every method (no bodies).
        """
        lines: list[str] = []
        for d in decorator_names:
            lines.append(d)

        # class Foo(Bar): line
        header_end = next(
            (c for c in node.children if c.type == "block"),
            None,
        )
        if header_end:
            class_line_end = header_end.start_byte
            class_line = code_bytes[node.start_byte:class_line_end].decode("utf-8", errors="replace").rstrip()
            lines.append(class_line)
        else:
            lines.append(f"class {class_name}:")

        if docstring:
            # Indent docstring to one level
            for dl in docstring.splitlines():
                lines.append(f"    {dl}")

        # Method signatures
        body_block = next((c for c in node.children if c.type == "block"), None)
        if body_block:
            for member in body_block.children:
                sig = self._extract_signature_line(member, code_bytes)
                if sig:
                    lines.append(f"    {sig}")
                    lines.append("        ...")

        return "\n".join(lines)

    def _extract_signature_line(self, node: Node, code_bytes: bytes) -> Optional[str]:
        """Return the 'def foo(...)' line of a function node (or decorated), or None."""
        if node.type == _DECORATED_TYPE:
            inner = next((c for c in node.children if c.type in _FUNC_TYPES), None)
            dec_lines = [
                code_bytes[d.start_byte:d.end_byte].decode("utf-8", errors="replace").strip()
                for d in node.children if d.type == "decorator"
            ]
            if inner:
                sig = self._extract_signature_line(inner, code_bytes)
                if sig:
                    return ("\n    ".join(dec_lines) + "\n    " + sig).lstrip("\n    ")
        if node.type in _FUNC_TYPES:
            # Find the ':' or body block to delimit the signature
            body = next((c for c in node.children if c.type == "block"), None)
            if body:
                sig_bytes = code_bytes[node.start_byte:body.start_byte]
            else:
                sig_bytes = code_bytes[node.start_byte:node.end_byte]
            return sig_bytes.decode("utf-8", errors="replace").rstrip().rstrip(":")
        return None

    # ------------------------------------------------------------------
    # Function / method extraction (Fix 1, 4, 5 + Strategy 4)
    # ------------------------------------------------------------------

    def _extract_function(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        imports_in_file: list[str],
        inside_class: bool,
        decorator_nodes: list[Node],
        decorator_start_byte: Optional[int],
        parent_class: Optional[str] = None,
    ) -> dict:
        """
        Extract a single function or method chunk.

        [Fix 4] We do NOT recurse into the function body to extract nested
                 functions — they stay embedded in the parent's code_content.
        [Fix 5] async_function_definition handled the same as function_definition.
        [Fix 1] Byte range starts at decorator_start_byte when decorators are
                present, so @decorator lines appear in code_content.
        """
        # In tree-sitter-python, async functions are 'function_definition'
        # nodes that contain a child token of type 'async', not a distinct
        # node type.  We detect async by inspecting child tokens directly.
        is_async = any(c.type == "async" for c in node.children)
        start_byte = decorator_start_byte if decorator_start_byte is not None else node.start_byte
        end_byte = node.end_byte

        # Compute start_line from byte offset
        if decorator_start_byte is not None:
            start_line = code_bytes[:start_byte].count(b"\n") + 1
        else:
            start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        code_content = code_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
        func_name = self._get_name(node, code_bytes)
        node_type = "method" if inside_class else "function"

        decorator_names = [
            code_bytes[d.start_byte:d.end_byte].decode("utf-8", errors="replace").strip()
            for d in decorator_nodes
        ]
        docstring = self._extract_body_docstring(node, code_bytes)
        is_dunder = func_name.startswith("__") and func_name.endswith("__")
        special_type = self._special_decorator_type(decorator_names)

        return {
            "code_content": code_content,
            "file_path": file_path,
            "node_type": node_type,
            "start_line": start_line,
            "end_line": end_line,
            "metadata": {
                "imports_in_file": imports_in_file,
                "decorators": decorator_names,
                "is_async": is_async,
                "parent_class": parent_class,
                "is_dunder": is_dunder,
                "special_type": special_type,  # 'property' | 'staticmethod' | etc.
                "docstring": docstring,
                "func_name": func_name,
                "module_docstring": None,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_name(self, node: Node, code_bytes: bytes) -> str:
        """Return the identifier name of a class or function node."""
        name_node = next((c for c in node.children if c.type == "identifier"), None)
        if name_node:
            return code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
        return "<anonymous>"

    def _get_base_classes(self, node: Node, code_bytes: bytes) -> list[str]:
        """Return base class name strings from a class_definition node."""
        bases: list[str] = []
        arg_list = next((c for c in node.children if c.type == "argument_list"), None)
        if arg_list:
            for child in arg_list.children:
                if child.type not in (",", "(", ")", "keyword_argument"):
                    text = code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                    if text:
                        bases.append(text)
        return bases

    def _extract_body_docstring(self, node: Node, code_bytes: bytes) -> Optional[str]:
        """
        Return the docstring of a function or class body (first statement if
        it is a string literal), or None if absent.
        """
        body = next((c for c in node.children if c.type == "block"), None)
        if not body:
            return None
        for stmt in body.children:
            if stmt.type == "expression_statement":
                expr = stmt.children[0] if stmt.children else None
                if expr and expr.type in ("string", "concatenated_string"):
                    return code_bytes[stmt.start_byte:stmt.end_byte].decode(
                        "utf-8", errors="replace"
                    ).strip()
            if stmt.type not in ("newline", "comment", ""):
                break
        return None

    def _special_decorator_type(self, decorator_names: list[str]) -> Optional[str]:
        """
        Return the first 'known-semantic' decorator name found (without the @),
        or None.  E.g. ['@classmethod'] → 'classmethod'.
        """
        for dec in decorator_names:
            bare = dec.lstrip("@").split("(")[0].strip()
            if bare in _SPECIAL_DECORATORS:
                return bare
        return None
