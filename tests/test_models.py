import pytest
from unittest.mock import MagicMock, patch
from noesiscli.models.client import GeminiClient

@patch("noesiscli.models.client.ChatGoogleGenerativeAI")
def test_gemini_client_generate_success(mock_chat_class):
    # Mock primary LLM
    mock_primary = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Primary Response"
    mock_primary.invoke.return_value = mock_response
    
    # Configure mock class to return our mocks
    mock_chat_class.side_effect = [mock_primary, MagicMock()]
    
    client = GeminiClient()
    response = client.generate("Hello", system_instruction="system")
    
    assert response == "Primary Response"
    mock_primary.invoke.assert_called_once()


@patch("noesiscli.models.client.ChatGoogleGenerativeAI")
def test_gemini_client_generate_fallback(mock_chat_class):
    # Mock primary LLM to fail
    mock_primary = MagicMock()
    mock_primary.invoke.side_effect = Exception("Primary error")
    
    # Mock fallback LLM to succeed
    mock_fallback = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Fallback Response"
    mock_fallback.invoke.return_value = mock_response
    
    # Configure mock class to return primary, then fallback
    mock_chat_class.side_effect = [mock_primary, mock_fallback]
    
    client = GeminiClient()
    response = client.generate("Hello", system_instruction="system")
    
    assert response == "Fallback Response"
    mock_primary.invoke.assert_called_once()
    mock_fallback.invoke.assert_called_once()


@patch("noesiscli.models.client.ChatGoogleGenerativeAI")
def test_gemini_client_stream_success(mock_chat_class):
    mock_primary = MagicMock()
    
    # Mock chunks returned by stream
    mock_chunk1 = MagicMock()
    mock_chunk1.content = "Chunk1 "
    mock_chunk2 = MagicMock()
    mock_chunk2.content = "Chunk2"
    mock_primary.stream.return_value = [mock_chunk1, mock_chunk2]
    
    mock_chat_class.side_effect = [mock_primary, MagicMock()]
    
    client = GeminiClient()
    stream_generator = client.stream("Hello", system_instruction="system")
    response_list = list(stream_generator)
    
    assert response_list == ["Chunk1 ", "Chunk2"]
    mock_primary.stream.assert_called_once()


@patch("noesiscli.models.client.ChatGoogleGenerativeAI")
def test_gemini_client_stream_fallback(mock_chat_class):
    mock_primary = MagicMock()
    # Mock stream method to fail immediately when iterator is read
    def failing_stream(*args, **kwargs):
        raise Exception("Stream failed")
    mock_primary.stream.side_effect = failing_stream
    
    mock_fallback = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.content = "Fallback1 "
    mock_chunk2 = MagicMock()
    mock_chunk2.content = "Fallback2"
    mock_fallback.stream.return_value = [mock_chunk1, mock_chunk2]
    
    mock_chat_class.side_effect = [mock_primary, mock_fallback]
    
    client = GeminiClient()
    stream_generator = client.stream("Hello", system_instruction="system")
    response_list = list(stream_generator)
    
    assert response_list == ["Fallback1 ", "Fallback2"]
    mock_primary.stream.assert_called_once()
    mock_fallback.stream.assert_called_once()
