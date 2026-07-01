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

