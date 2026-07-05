import pytest
import os
from unittest.mock import MagicMock, patch

# Try to import, otherwise skip tests or mock imports
try:
    from noesiscli.parser.scanner import RepositoryScanner
except ImportError:
    RepositoryScanner = None

try:
    from noesiscli.parser.tree_sitter_parser import TreeSitterParser
except ImportError:
    TreeSitterParser = None

try:
    from noesiscli.parser.symbol_table import SymbolTableBuilder
except ImportError:
    SymbolTableBuilder = None

try:
    from noesiscli.parser.dependency_graph import DependencyGraphBuilder
except ImportError:
    DependencyGraphBuilder = None

try:
    from noesiscli.parser.parallel import ParallelParser
except ImportError:
    ParallelParser = None


@pytest.mark.skipif(RepositoryScanner is None, reason="RepositoryScanner not implemented")
def test_repository_scanner(temp_repo):
    """Test that the scanner finds all supported files while ignoring build/venv folders."""
    scanner = RepositoryScanner(ignore_dirs=[".git", ".venv", "tests"])
    files = scanner.scan(temp_repo)
    
    # Check that we found the python file and JS file in src
    basenames = [os.path.basename(f) for f in files]
    assert "user_service.py" in basenames
    assert "math.js" in basenames
    assert "ignored.py" not in basenames


@pytest.mark.skipif(TreeSitterParser is None, reason="TreeSitterParser not implemented")
def test_tree_sitter_parser_python():
    """Test that the parser extracts classes and functions from a Python code block."""
    parser = TreeSitterParser(language="python")
    code = """
class Target:
    def method(self):
        pass

def top_func():
    pass
"""
    chunks = parser.parse_code(code, file_path="/mock/path.py")
    
    # Assert return structure
    assert isinstance(chunks, list)
    assert len(chunks) >= 2
    
    node_types = [c.get("node_type") for c in chunks]
    assert "class" in node_types
    assert "function" in node_types or "method" in node_types
    
    for chunk in chunks:
        assert "code_content" in chunk
        assert "start_line" in chunk
        assert "end_line" in chunk


@pytest.mark.skipif(SymbolTableBuilder is None, reason="SymbolTableBuilder not implemented")
def test_symbol_table_builder(mock_code_chunks):
    """Test that symbol table builder registers and indexes code symbols correctly."""
    builder = SymbolTableBuilder()
    
    # Build table
    symbol_table = builder.build(mock_code_chunks)
    
    assert isinstance(symbol_table, dict)
    assert "UserService" in symbol_table
    assert "find_user" in symbol_table
    
    # Check definition content
    user_service_def = symbol_table["UserService"]
    assert isinstance(user_service_def, list)
    assert len(user_service_def) > 0
    assert user_service_def[0]["file_path"] == "/mock/project/src/user_service.py"


@pytest.mark.skipif(DependencyGraphBuilder is None, reason="DependencyGraphBuilder not implemented")
def test_dependency_graph_builder(mock_code_chunks):
    """Test that dependency graph registers imports and relationship edges."""
    builder = DependencyGraphBuilder()
    
    # In networkx, we build a directed graph
    graph = builder.build(mock_code_chunks)
    
    assert graph is not None
    # Nodes should be the symbols or file paths
    # If using NetworkX graph:
    assert hasattr(graph, "add_edge")
    assert hasattr(graph, "nodes")
    
    # Check if files or symbols exist as nodes
    nodes = list(graph.nodes)
    assert len(nodes) > 0


@pytest.mark.skipif(ParallelParser is None, reason="ParallelParser not implemented")
@patch("multiprocessing.Pool")
def test_parallel_parser(mock_pool):
    """Test that ParallelParser distributes file parsing across multiprocessing pool."""
    # Setup mock pool return
    mock_instance = mock_pool.return_value
    mock_instance.map.return_value = [[{"node_type": "function", "code_content": "def parallel_func(): pass"}]]
    
    parser = ParallelParser(num_cores=2)
    files = ["/mock/file1.py", "/mock/file2.py"]
    
    results = parser.parse_files(files)
    
    assert isinstance(results, list)
    assert len(results) > 0
    mock_instance.map.assert_called_once()


def test_cli_analyze(temp_repo):
    """Test that CLI analyze command successfully scans and parses Python files."""
    from noesiscli.cli import main
    with patch("sys.argv", ["noesiscli", "analyze", temp_repo]):
        chunks = main()
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        node_types = [c.get("node_type") for c in chunks]
        assert "class" in node_types
        assert "function" in node_types


@pytest.mark.skipif(TreeSitterParser is None, reason="TreeSitterParser not implemented")
def test_tree_sitter_parser_module_and_imports():
    """Test that the parser extracts a module chunk and a dedicated imports chunk."""
    parser = TreeSitterParser(language="python")
    code = """
\"\"\"
Module docstring.
\"\"\"
import os
import sys
"""
    chunks = parser.parse_code(code, file_path="/mock/path.py")
    
    module_chunks = [c for c in chunks if c.get("node_type") == "module"]
    imports_chunks = [c for c in chunks if c.get("node_type") == "imports"]
    
    assert len(module_chunks) == 1
    assert len(imports_chunks) == 1
    
    # Verify module chunk
    assert 'Module docstring.' in module_chunks[0]["code_content"]
    assert 'import os' in module_chunks[0]["code_content"]
    assert module_chunks[0]["metadata"]["module_docstring"] == '"""\nModule docstring.\n"""'
    assert module_chunks[0]["metadata"]["imports_in_file"] == ["import os", "import sys"]
    
    # Verify imports chunk
    assert "import os\nimport sys" == imports_chunks[0]["code_content"]
    assert imports_chunks[0]["metadata"]["imports_in_file"] == ["import os", "import sys"]
    assert imports_chunks[0]["metadata"]["imports_parsed"] == ["import os", "import sys"]


@pytest.mark.skipif(TreeSitterParser is None, reason="TreeSitterParser not implemented")
def test_tree_sitter_parser_global_classification():
    """Test that global statements are classified into 'constant', 'type_alias', or 'global'."""
    parser = TreeSitterParser(language="python")
    
    # 1. Constant
    chunks_const = parser.parse_code("API_VERSION = 'v1'\nDEBUG = True", file_path="/mock/path.py")
    constant_chunks = [c for c in chunks_const if c.get("node_type") == "constant"]
    assert len(constant_chunks) == 1
    assert "API_VERSION = 'v1'\nDEBUG = True" in constant_chunks[0]["code_content"]
    
    # 2. Type Alias
    chunks_alias = parser.parse_code("MyType = Union[int, str]\nX: TypeAlias = int", file_path="/mock/path.py")
    type_alias_chunks = [c for c in chunks_alias if c.get("node_type") == "type_alias"]
    assert len(type_alias_chunks) == 1
    assert "MyType = Union[int, str]\nX: TypeAlias = int" in type_alias_chunks[0]["code_content"]
    
    # 3. Global expression/call
    chunks_global = parser.parse_code("configure_logging()", file_path="/mock/path.py")
    global_chunks = [c for c in chunks_global if c.get("node_type") == "global"]
    assert len(global_chunks) == 1
    assert "configure_logging()" in global_chunks[0]["code_content"]



@pytest.mark.skipif(TreeSitterParser is None, reason="TreeSitterParser not implemented")
def test_tree_sitter_parser_decorators():
    """Test that decorated functions and classes include decorators in code content and metadata."""
    parser = TreeSitterParser(language="python")
    code = """
@deco
def my_func():
    pass

@class_deco
class MyClass:
    @property
    def my_prop(self):
        return 42
"""
    chunks = parser.parse_code(code, file_path="/mock/path.py")
    
    func_chunk = [c for c in chunks if c.get("node_type") == "function"][0]
    assert func_chunk["code_content"].startswith("@deco")
    assert func_chunk["metadata"]["decorators"] == ["@deco"]
    
    class_chunk = [c for c in chunks if c.get("node_type") == "class"][0]
    assert class_chunk["code_content"].startswith("@class_deco")
    assert class_chunk["metadata"]["decorators"] == ["@class_deco"]
    
    method_chunk = [c for c in chunks if c.get("node_type") == "method"][0]
    assert method_chunk["code_content"].startswith("@property")
    assert method_chunk["metadata"]["decorators"] == ["@property"]
    assert method_chunk["metadata"]["special_type"] == "property"


@pytest.mark.skipif(TreeSitterParser is None, reason="TreeSitterParser not implemented")
def test_tree_sitter_parser_async_functions():
    """Test that async functions are correctly parsed with is_async metadata."""
    parser = TreeSitterParser(language="python")
    code = """
async def async_func(x):
    return x
"""
    chunks = parser.parse_code(code, file_path="/mock/path.py")
    func_chunk = [c for c in chunks if c.get("node_type") == "function"][0]
    assert func_chunk["metadata"]["is_async"] is True


@pytest.mark.skipif(TreeSitterParser is None, reason="TreeSitterParser not implemented")
def test_tree_sitter_parser_nested_functions():
    """Test that nested functions are NOT extracted as separate chunks."""
    parser = TreeSitterParser(language="python")
    code = """
def outer():
    def inner():
        pass
    return inner
"""
    chunks = parser.parse_code(code, file_path="/mock/path.py")
    
    # We should have a module chunk and the outer function chunk.
    # No chunk should exist for the inner function alone.
    func_chunks = [c for c in chunks if c.get("node_type") == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0]["metadata"]["func_name"] == "outer"
    assert "def inner():" in func_chunks[0]["code_content"]


@pytest.mark.skipif(TreeSitterParser is None, reason="TreeSitterParser not implemented")
def test_tree_sitter_parser_class_headers_and_methods():
    """Test class header skeletal signatures and recursive method extraction."""
    parser = TreeSitterParser(language="python")
    code = """
class MyClass(Base):
    \"\"\"Docstring.\"\"\"
    def method_one(self):
        pass
    
    @classmethod
    def method_two(cls):
        pass
"""
    chunks = parser.parse_code(code, file_path="/mock/path.py")
    
    # Extract class_header chunk
    header_chunk = [c for c in chunks if c.get("node_type") == "class_header"][0]
    expected_header = (
        "class MyClass(Base):\n"
        "    \"\"\"Docstring.\"\"\"\n"
        "    def method_one(self)\n"
        "        ...\n"
        "    @classmethod\n"
        "    def method_two(cls)\n"
        "        ..."
    )
    assert header_chunk["code_content"] == expected_header
    assert header_chunk["metadata"]["class_name"] == "MyClass"
    assert header_chunk["metadata"]["base_classes"] == ["Base"]
    
    # Verify methods are extracted separately
    methods = [c for c in chunks if c.get("node_type") == "method"]
    assert len(methods) == 2
    
    method_names = [m["metadata"]["func_name"] for m in methods]
    assert "method_one" in method_names
    assert "method_two" in method_names
    
    for m in methods:
        assert m["metadata"]["parent_class"] == "MyClass"


@patch("noesiscli.cli.ChromaVectorStore")
@patch("noesiscli.models.client.GeminiClient")
@patch("os.path.isdir", return_value=True)
def test_cli_query(mock_isdir, mock_gemini_client_class, mock_chroma_store_class):
    """Test that CLI query command retrieves context and calls Gemini stream."""
    from noesiscli.cli import main
    
    # Mock vector store
    mock_store = mock_chroma_store_class.return_value
    mock_store.query.return_value = [
        {
            "code_content": "def test_func(): pass",
            "file_path": "/mock/test.py",
            "start_line": 1,
            "end_line": 2,
            "node_type": "function"
        }
    ]
    
    # Mock GeminiClient
    mock_gemini = mock_gemini_client_class.return_value
    mock_gemini.stream.return_value = ["Test ", "Stream ", "Response"]
    
    # Run CLI query
    with patch("sys.argv", ["noesiscli", "query", "How does test_func work?"]):
        results = main()
        
        # Assertions
        assert results is not None
        assert len(results) == 1
        assert results[0]["code_content"] == "def test_func(): pass"
        
        mock_store.query.assert_called_once_with("How does test_func work?", top_k=3)
        mock_gemini.stream.assert_called_once()



