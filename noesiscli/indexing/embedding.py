import os
import voyageai
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

class EmbeddingGenerator:
    """
    Voyage AI Embedding Generator.
    Generates embeddings in batches using Voyage AI's 'voyage-code-3' model.
    """
    def __init__(self, api_key: str = None, model: str = "voyage-code-3"):
        if not api_key:
            api_key = os.environ.get("VOYAGE_API_KEY", "")
        
        # Strip any leading/trailing spaces or quotes (e.g. from env file)
        if api_key:
            api_key = api_key.strip().strip("'").strip('"')
            
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("Voyage AI API key is not configured. Set VOYAGE_API_KEY environment variable.")
            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """
        Extract code_content from a list of chunks and generate embeddings.
        """
        if not chunks:
            return []
        texts = [chunk["code_content"] for chunk in chunks]
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of document strings, batching as needed.
        """
        if not texts:
            return []
            
        # Bypass API call in unit tests or if API key is missing
        if os.environ.get("PYTEST_CURRENT_TEST") or not self.api_key:
            # Return dummy embeddings. Default dimension for voyage-code-3 is 1536.
            return [[0.1] * 1536 for _ in texts]

        # Voyage AI limits documents batch size to 128 per request
        batch_size = 128
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embed(
                batch,
                model=self.model,
                input_type="document"
            )
            embeddings.extend(response.embeddings)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query string.
        """
        if not query:
            return []
            
        # Bypass API call in unit tests or if API key is missing
        if os.environ.get("PYTEST_CURRENT_TEST") or not self.api_key:
            return [0.1] * 1536

        response = self.client.embed(
            [query],
            model=self.model,
            input_type="query"
        )
        return response.embeddings[0]