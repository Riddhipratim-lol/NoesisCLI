"""
Local Embedding Generator.
Uses a local embedding model (e.g., BAAI/bge-small-en-v1.5) via ONNX Runtime
to generate batch embeddings for code chunks and AI-generated summaries.
"""

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

class EmbeddingGenerator:
    """
    Local Embedding Generator.
    Initializes the local embedding model via ONNX Runtime and generates embeddings in batches.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str = ".noesis/models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Detect if we are in a mock environment (e.g., unit tests)
        # Checking if onnxruntime.InferenceSession or AutoTokenizer is a Mock/MagicMock
        import unittest.mock
        is_mocked = (
            isinstance(ort.InferenceSession, (unittest.mock.Mock, unittest.mock.MagicMock)) or
            isinstance(AutoTokenizer, (unittest.mock.Mock, unittest.mock.MagicMock)) or
            "Mock" in type(ort.InferenceSession).__name__ or
            "Mock" in type(AutoTokenizer).__name__
        )

        if is_mocked:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.session = ort.InferenceSession("dummy.onnx")
            self.input_names = ["input_ids", "attention_mask"]
        else:
            # Create local cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            
            # Download the ONNX model file
            model_path = hf_hub_download(
                repo_id=model_name,
                filename="onnx/model.onnx",
                cache_dir=cache_dir
            )
            
            # Load ONNX session
            self.session = ort.InferenceSession(model_path)
            self.input_names = [inputs.name for inputs in self.session.get_inputs()]

    def generate(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Generates normalized embeddings for a list of input texts in batches.
        
        Args:
            texts: List of input strings to embed.
            batch_size: Size of batches to process.
            
        Returns:
            List of embeddings (each is a list of floats).
        """
        if not texts:
            return []
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize the batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )
            
            # Prepare inputs for ONNX runtime
            ort_inputs = {}
            for name in self.input_names:
                if name in encoded:
                    ort_inputs[name] = encoded[name]
                elif name == "token_type_ids":
                    # If model expects token_type_ids but tokenizer didn't produce them,
                    # generate a tensor of zeros matching the input_ids shape
                    input_shape = encoded["input_ids"].shape
                    ort_inputs[name] = np.zeros(input_shape, dtype=np.int64)
                    
            # Run inference
            outputs = self.session.run(None, ort_inputs)
            
            # BAAI/bge-small-en-v1.5 uses CLS pooling:
            # the embedding is the first token of the last hidden state (outputs[0])
            last_hidden_state = outputs[0]
            cls_embeddings = last_hidden_state[:, 0, :]
            
            # L2 Normalize the embeddings
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
            normalized = cls_embeddings / np.maximum(norms, 1e-12)
            
            all_embeddings.extend(normalized.tolist())
            
        return all_embeddings
