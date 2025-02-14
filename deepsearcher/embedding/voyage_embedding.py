import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

VOYAGE_MODEL_DIM_MAP = {
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
}


class VoyageEmbedding(BaseEmbedding):
    """
    https://docs.voyageai.com/embeddings/
    """

    def __init__(self, model_name="voyage-3", **kwargs):
        self.model_name = model_name
        self.voyageai_api_key = os.getenv("VOYAGE_API_KEY")

        import voyageai

        voyageai.api_key = self.voyageai_api_key
        self.vo = voyageai.Client(**kwargs)

    def embed_query(self, text: str) -> List[float]:
        """
        input_type (`str`): "query" or "document" for retrieval case.
        """
        embeddings = self.vo.embed(
            [text], model=self.model_name, input_type="query"
        )
        return embeddings.embeddings[0]
    
    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        embeddings = self.vo.embed(
            texts, model=self.model_name, input_type="document"
        )
        return embeddings.embeddings

    @property
    def dimension(self) -> int:
        return VOYAGE_MODEL_DIM_MAP[self.model_name]
