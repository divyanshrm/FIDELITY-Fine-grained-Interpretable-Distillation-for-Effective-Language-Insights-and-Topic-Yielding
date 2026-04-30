import os
import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

class Embedder:
    """Class holding the active connection and embedding pipeline for words/documents."""
    
    def __init__(self, model_name: str | None = None):
        load_dotenv()
        self.model_name = model_name or os.getenv('EMB_MODEL', 'sfr-embedding-mistral')
        self.base_url = os.getenv('LLM_BASE_URL', 'https://api.ai.it.ufl.edu')
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv('OPENAI_API_KEY', 'dummy-key')
        )

    def find_embeddings_using_transformers(self, data: str | list[str]) -> np.ndarray:
        """Creates embeddings using the remote API setup."""
        if not data:
            return np.array([])
            
        data_to_send = [data] if isinstance(data, str) else data
        response = self.client.embeddings.create(
            input=data_to_send,
            model=self.model_name
        )
        embeddings = np.array([item.embedding for item in response.data])
        
        return embeddings[0] if isinstance(data, str) else embeddings

    def embed(self, text: list[str]) -> np.ndarray:
        """Converts document text to an embedding array."""
        return self.find_embeddings_using_transformers(text)

    @staticmethod
    def get_vector_similarity(x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Computes cosine similarity between two PyTorch vectors."""
        similarity = torch.nn.functional.cosine_similarity(x1=x1, x2=x2, dim=-1)
        return float(similarity)

    def embed_keywords(self, all_keywords: list[str]) -> tuple[dict, list[np.ndarray]]:
        """Embeds a list of keywords using the embedding API in batches."""
        keyword_embedding_dict = {}
        keyword_embeddings = []
        batch_size = 100
        
        for i in tqdm(range(0, len(all_keywords), batch_size), desc="Embedding Batches", leave=False):
            batch = all_keywords[i:i + batch_size]
            batch_emb = self.find_embeddings_using_transformers(batch)
            keyword_embeddings.extend(batch_emb)
            
        for kw, emb in zip(all_keywords, keyword_embeddings):
            keyword_embedding_dict[kw] = emb
            
        return keyword_embedding_dict, keyword_embeddings
