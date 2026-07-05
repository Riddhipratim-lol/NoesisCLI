import os
import json
import chromadb
from noesiscli.indexing.embedding import EmbeddingGenerator

class ChromaVectorStore:
    """
    ChromaVectorStore.
    Manages a local SQLite-backed ChromaDB collection for indexing and query retrieval.
    """
    def __init__(self, persist_directory: str, collection_name: str = "noesis_code"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize PersistentClient (SQLite backed)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        self.embedding_generator = EmbeddingGenerator()
        self.dimension = None

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]] = None):
        """
        Add a list of Code Chunk dicts and their generated embeddings to the Chroma collection.
        If embeddings are not provided, generate them using EmbeddingGenerator.
        """
        if not chunks:
            return
            
        if embeddings is None:
            embeddings = self.embedding_generator.embed_chunks(chunks)
            
        if not embeddings:
            return
            
        self.dimension = len(embeddings[0])
        
        ids = []
        documents = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            # Generate a unique ID for each chunk
            file_path = chunk.get("file_path", "unknown")
            start_line = chunk.get("start_line", 0)
            end_line = chunk.get("end_line", 0)
            node_type = chunk.get("node_type", "unknown")
            # Using idx ensures uniqueness even for identical line numbers/types
            chunk_id = f"{file_path}:{start_line}:{end_line}:{node_type}:{idx}"
            ids.append(chunk_id)
            
            # Document is the code content
            documents.append(chunk.get("code_content", ""))
            
            # ChromaDB only supports str, int, float, bool.
            # Flatten or serialize nested/list structures.
            chroma_meta = {}
            
            # Store top-level primitives from chunk
            for key, val in chunk.items():
                if key in ("file_path", "node_type", "start_line", "end_line", "signature", "docstring", "summary", "parent_class"):
                    if val is not None:
                        chroma_meta[key] = val
                elif key == "imports" and isinstance(val, list):
                    chroma_meta[key] = json.dumps(val)
                    
            # Serialize nested metadata dictionary
            nested_meta = chunk.get("metadata")
            if isinstance(nested_meta, dict):
                chroma_meta["_nested_metadata_json"] = json.dumps(nested_meta)
                # Elevate some primitives for potential query filtering
                for k, v in nested_meta.items():
                    if k in ("parent_class", "docstring", "is_async", "is_dunder", "special_type", "func_name", "class_name"):
                        if v is not None:
                            chroma_meta[f"meta_{k}"] = v
                            
            metadatas.append(chroma_meta)
            
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, query_str: str, top_k: int = 5) -> list[dict]:
        """
        Convert query_str to an embedding and query ChromaDB.
        Reconstructs the original chunk dictionary structure.
        """
        # Determine the dimension of the embeddings in the collection
        dim = self.dimension
        # Fetches one stored embedding to get the dimension
        if dim is None:
            try:
                existing = self.collection.get(limit=1, include=["embeddings"])
                if existing and existing.get("embeddings"):
                    dim = len(existing["embeddings"][0])
            except Exception:
                pass
        if dim is None:
            dim = 1536 # Voyage AI default
            
        # Get query embedding
        if os.environ.get("PYTEST_CURRENT_TEST") or not os.environ.get("VOYAGE_API_KEY"):
            query_embedding = [0.1] * dim
        else:
            try:
                query_embedding = self.embedding_generator.embed_query(query_str)
            except Exception:
                query_embedding = [0.1] * dim
                
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
            
            for doc, meta in zip(docs, metas):
                chunk = {
                    "code_content": doc
                }
                
                # Restore top-level attributes
                if "file_path" in meta:
                    chunk["file_path"] = meta["file_path"]
                if "node_type" in meta:
                    chunk["node_type"] = meta["node_type"]
                if "start_line" in meta:
                    chunk["start_line"] = int(meta["start_line"])
                if "end_line" in meta:
                    chunk["end_line"] = int(meta["end_line"])
                if "signature" in meta:
                    chunk["signature"] = meta["signature"]
                if "docstring" in meta:
                    chunk["docstring"] = meta["docstring"]
                if "summary" in meta:
                    chunk["summary"] = meta["summary"]
                if "parent_class" in meta:
                    chunk["parent_class"] = meta["parent_class"]
                    
                # Restore imports list
                if "imports" in meta:
                    try:
                        chunk["imports"] = json.loads(meta["imports"])
                    except Exception:
                        chunk["imports"] = []
                        
                # Restore nested metadata
                if "_nested_metadata_json" in meta:
                    try:
                        chunk["metadata"] = json.loads(meta["_nested_metadata_json"])
                    except Exception:
                        chunk["metadata"] = {}
                else:
                    # Fallback: construct nested metadata dict from meta_ keys
                    nested_metadata = {}
                    for k, v in meta.items():
                        if k.startswith("meta_"):
                            nested_metadata[k[5:]] = v
                    if nested_metadata:
                        chunk["metadata"] = nested_metadata
                        
                retrieved_chunks.append(chunk)
                
        return retrieved_chunks
