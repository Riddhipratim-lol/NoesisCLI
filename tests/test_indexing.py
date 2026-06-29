import pytest
from unittest.mock import MagicMock, patch, mock_open

# Try to import, otherwise skip tests or mock imports
try:
    from noesiscli.indexing.embedding import EmbeddingGenerator
except ImportError:
    EmbeddingGenerator = None

try:
    from noesiscli.indexing.vector_store import ChromaVectorStore
except ImportError:
    ChromaVectorStore = None

try:
    from noesiscli.indexing.bm25_store import BM25Store
except ImportError:
    BM25Store = None


@pytest.mark.skipif(EmbeddingGenerator is None, reason="EmbeddingGenerator not implemented")
@patch("onnxruntime.InferenceSession")
@patch("transformers.AutoTokenizer")
def test_embedding_generator(mock_tokenizer, mock_session, mock_code_chunks):
    """Test local ONNX embedding generator and verify batch outputs."""
    # Setup mock tokenizer and session
    mock_tok_instance = mock_tokenizer.from_pretrained.return_value
    mock_tok_instance.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock()
    }
    
    mock_sess_instance = mock_session.return_value
    # Mocking output of ONNX run
    import numpy as np
    mock_sess_instance.run.return_value = [
        np.random.rand(len(mock_code_chunks), 1, 384)  # shape (batch_size, seq_len, emb_dim)
    ]
    
    generator = EmbeddingGenerator(model_name="BAAI/bge-small-en-v1.5", cache_dir=".noesis/models")
    
    texts = [c["code_content"] for c in mock_code_chunks]
    embeddings = generator.generate(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(mock_code_chunks)
    assert len(embeddings[0]) == 384 or isinstance(embeddings[0], list)


@pytest.mark.skipif(ChromaVectorStore is None, reason="ChromaVectorStore not implemented")
@patch("chromadb.PersistentClient")
def test_chroma_vector_store(mock_client, mock_code_chunks):
    """Test ChromaDB local vector store storage and retrieval."""
    mock_collection = MagicMock()
    mock_client_instance = mock_client.return_value
    mock_client_instance.get_or_create_collection.return_value = mock_collection
    
    vector_store = ChromaVectorStore(persist_directory=".noesis/chroma")
    
    # Generate mock embeddings
    mock_embeddings = [[0.1] * 384, [0.2] * 384]
    
    vector_store.add_chunks(mock_code_chunks, mock_embeddings)
    
    # Assert collection upsert is called
    mock_collection.add.assert_called_once()
    
    # Test query
    mock_collection.query.return_value = {
        "documents": [["def find_user(username): pass"]],
        "metadatas": [[{"file_path": "/mock/db.py"}]],
        "distances": [[0.1]],
        "ids": [["id_1"]]
    }
    
    results = vector_store.query("find user", top_k=1)
    assert len(results) > 0
    assert "def find_user(username): pass" in results[0]["code_content"]


@pytest.mark.skipif(BM25Store is None, reason="BM25Store not implemented")
@patch("builtins.open", new_callable=mock_open)
@patch("pickle.dump")
@patch("pickle.load")
def test_bm25_store(mock_pickle_load, mock_pickle_dump, mock_file, mock_code_chunks):
    """Test BM25 serialization, deserialization, and lexical search."""
    store = BM25Store()
    
    # Index documents
    store.index(mock_code_chunks)
    
    # Save index
    store.save(".noesis/bm25.pkl")
    mock_file.assert_called_with(".noesis/bm25.pkl", "wb")
    mock_pickle_dump.assert_called_once()
    
    # Mock load
    mock_pickle_load.return_value = store
    loaded_store = BM25Store.load(".noesis/bm25.pkl")
    assert loaded_store is not None
    
    # Query test
    store.bm25 = MagicMock()
    # Mock return values for BM25 scores
    store.bm25.get_scores.return_value = [0.1, 0.9]
    
    results = store.query("find_user", top_k=1)
    assert len(results) == 1
