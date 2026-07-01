"""
Tree-sitter Parser.
Integrates Tree-sitter for parsing Python source code
to perform syntax-aware AST parsing, extracting functions, classes, and methods.
"""

import tree_sitter_python
from tree_sitter import Language, Parser

class TreeSitterParser:
    """
    Tree-sitter Parser to parse source code into AST chunks.
    Currently optimized for Python.
    """
    def __init__(self, language: str = "python"):
        lang_key = language.lower()
        if lang_key not in ("python", "py"):
            raise ValueError(f"Only 'python' is supported in the current Phase 1.2 implementation. Got '{language}'")
            
        try:
            self.language = Language(tree_sitter_python.language())
            self.parser = Parser(self.language)
        except Exception as e:
            raise ValueError(f"Failed to load tree-sitter language '{language}': {e}")

    def parse_code(self, code: str, file_path: str) -> list[dict]:
        """
        Parses source code and returns a list of semantic code chunks (classes, functions, methods, global).
        """
        if not code:
            return []
            
        # Parse code as bytes to get proper offsets
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        
        code_bytes = code.encode("utf8")
        
        chunks = []
        pending_global_nodes = []

        def process_pending_global():
            if not pending_global_nodes:
                return
            start_byte = pending_global_nodes[0].start_byte
            end_byte = pending_global_nodes[-1].end_byte
            start_line = pending_global_nodes[0].start_point[0] + 1
            end_line = pending_global_nodes[-1].end_point[0] + 1
            
            content = code_bytes[start_byte:end_byte].decode("utf8", errors="replace").strip()
            if content:
                chunks.append({
                    "code_content": content,
                    "file_path": file_path,
                    "node_type": "global",
                    "start_line": start_line,
                    "end_line": end_line
                })
            pending_global_nodes.clear()

        # Iterate top-level children under module root
        for child in root_node.children:
            if child.type in ("class_definition", "function_definition", "decorated_definition"):
                process_pending_global()
                chunks.extend(self._extract_chunks_from_node(child, code_bytes, file_path, inside_class=False))
            elif child.type in ("import_statement", "import_from_statement"):
                process_pending_global()
                # Skip import statements
            else:
                pending_global_nodes.append(child)

        process_pending_global()
        return chunks

    def parse_file(self, file_path: str) -> list[dict]:
        """
        Reads a file from disk and parses its content into chunks.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
            return self.parse_code(code, file_path)
        except Exception as e:
            # Gracefully return empty list if file cannot be read
            return []

    def _extract_chunks_from_node(
        self, node, code_bytes: bytes, file_path: str, inside_class: bool = False
    ) -> list[dict]:
        chunks = []
        
        # Check node type
        if node.type == "class_definition":
            # Extract class chunk
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            # Extract raw code slice
            code_content = code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="replace")
            
            chunks.append({
                "code_content": code_content,
                "file_path": file_path,
                "node_type": "class",
                "start_line": start_line,
                "end_line": end_line
            })
            
            # Recurse into children as part of class body
            for child in node.children:
                chunks.extend(self._extract_chunks_from_node(child, code_bytes, file_path, inside_class=True))
                
        elif node.type == "function_definition":
            # Extract function/method chunk
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            code_content = code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="replace")
            node_type = "method" if inside_class else "function"
            
            chunks.append({
                "code_content": code_content,
                "file_path": file_path,
                "node_type": node_type,
                "start_line": start_line,
                "end_line": end_line
            })
            
            # Recurse into children (for potential nested definitions)
            for child in node.children:
                chunks.extend(self._extract_chunks_from_node(child, code_bytes, file_path, inside_class=inside_class))
                
        else:
            # Recurse into all other nodes
            for child in node.children:
                chunks.extend(self._extract_chunks_from_node(child, code_bytes, file_path, inside_class=inside_class))
                
        return chunks
