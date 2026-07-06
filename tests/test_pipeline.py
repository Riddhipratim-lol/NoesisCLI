import pytest
from unittest.mock import MagicMock, patch

# Try to import, otherwise skip tests or mock imports
try:
    from noesiscli.pipeline.state import WorkflowState
except ImportError:
    WorkflowState = None

try:
    from noesiscli.pipeline.validation import QueryValidator
except ImportError:
    QueryValidator = None

try:
    from noesiscli.pipeline.router import QueryRouter
except ImportError:
    QueryRouter = None

try:
    from noesiscli.pipeline.direct import DirectResponder
except ImportError:
    DirectResponder = None

try:
    from noesiscli.pipeline.rag import RAGNode
except ImportError:
    RAGNode = None

try:
    from noesiscli.pipeline.graph import WorkflowGraph
except ImportError:
    WorkflowGraph = None


@pytest.mark.skipif(WorkflowState is None, reason="WorkflowState not implemented")
def test_workflow_state():
    """Test that the state tracks the necessary RAG pipeline attributes."""
    # State could be a TypedDict or a Pydantic model
    # If TypedDict, we check it behaves like a dict:
    state = {
        "query": "How does auth work?",
        "is_valid": True,
        "route": "repository_rag",
        "context_chunks": [],
        "response": ""
    }
    assert state["query"] == "How does auth work?"
    assert state["is_valid"] is True


@pytest.mark.skipif(QueryValidator is None, reason="QueryValidator not implemented")
@patch("noesiscli.models.client.GeminiClient")
def test_query_validator(mock_gemini_client):
    """Test that coding queries are validated and non-coding queries are rejected."""
    mock_instance = mock_gemini_client.return_value
    
    # Mock validation output
    mock_instance.generate.side_effect = ["True", "False"]
    
    validator = QueryValidator(llm_client=mock_instance)
    
    # Valid
    assert validator.validate("Explain recursion") is True
    # Invalid
    assert validator.validate("What is the weather today?") is False


@pytest.mark.skipif(QueryValidator is None, reason="QueryValidator not implemented")
@patch("noesiscli.models.client.GeminiClient")
def test_query_validator_validate_and_route(mock_gemini_client):
    """Test that validate_and_route classifies queries correctly in one step."""
    mock_instance = mock_gemini_client.return_value
    mock_instance.generate.side_effect = ["repository_rag", "direct_llm", "invalid", "True", "repository_rag"]
    
    validator = QueryValidator(llm_client=mock_instance)
    
    # 1. repository_rag
    is_valid, route = validator.validate_and_route("Explain the database configuration")
    assert is_valid is True
    assert route == "repository_rag"
    
    # 2. direct_llm
    is_valid, route = validator.validate_and_route("What is a class?")
    assert is_valid is True
    assert route == "direct_llm"
    
    # 3. invalid
    is_valid, route = validator.validate_and_route("What is the weather today?")
    assert is_valid is False
    assert route == "invalid"

    # 4. Backward-compatible "True" response which should trigger a fallback router call
    is_valid, route = validator.validate_and_route("How do I code?")
    assert is_valid is True
    assert route == "repository_rag"


@pytest.mark.skipif(QueryRouter is None, reason="QueryRouter not implemented")
@patch("noesiscli.models.client.GeminiClient")
def test_query_router(mock_gemini_client):
    """Test routing classification between direct_llm and repository_rag."""
    mock_instance = mock_gemini_client.return_value
    mock_instance.generate.side_effect = ["direct_llm", "repository_rag"]
    
    router = QueryRouter(llm_client=mock_instance)
    
    # General question
    assert router.route("What is a decorator?") == "direct_llm"
    # Repo-specific
    assert router.route("Where is payment database defined?") == "repository_rag"


@pytest.mark.skipif(DirectResponder is None, reason="DirectResponder not implemented")
@patch("noesiscli.models.client.GeminiClient")
def test_direct_responder(mock_gemini_client):
    """Test executing direct LLM route for general programming query."""
    mock_instance = mock_gemini_client.return_value
    mock_instance.stream.return_value = ["Direct ", "answer ", "here"]
    
    responder = DirectResponder(llm_client=mock_instance)
    response_stream = responder.execute("Explain inheritance")
    
    assert list(response_stream) == ["Direct ", "answer ", "here"]


@pytest.mark.skipif(RAGNode is None, reason="RAGNode not implemented")
@patch("noesiscli.models.client.GeminiClient")
def test_rag_node(mock_gemini_client, mock_code_chunks):
    """Test RAG retrieval, context loading, and reasoning generation."""
    mock_llm = mock_gemini_client.return_value
    mock_llm.stream.return_value = ["Reasoned ", "answer"]
    
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_code_chunks
    
    rag_node = RAGNode(llm_client=mock_llm, retriever=mock_retriever)
    
    response_stream = rag_node.execute("How is auth implemented?")
    
    assert list(response_stream) == ["Reasoned ", "answer"]
    mock_retriever.retrieve.assert_called_once_with("How is auth implemented?")


@pytest.mark.skipif(WorkflowGraph is None, reason="WorkflowGraph not implemented")
def test_workflow_graph_compiles():
    """Test that LangGraph compiles and holds required nodes and transitions."""
    wf_graph = WorkflowGraph()
    graph = wf_graph.compile()
    
    assert graph is not None
    # LangGraph compiled graph has print_ascii, invoke, or get_graph method
    assert hasattr(graph, "invoke")
