import torch
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.rag_chunker import RagChunk
from .base_retriever import BaseRetriever
from .retrieval_data_models import RetrievalResults


class DenseRetriever(BaseRetriever):

    def __init__(
            self,
            model_name: str = "intfloat/multilingual-e5-small",
            insert_metadata_into_text: bool = True,
            expansion_strategy: str = "none",
            top_k: int = 5,
            query_instruction: str = "query: ",
            text_instruction: str = "passage: ",
    ):
        super().__init__(insert_metadata_into_text, expansion_strategy, top_k)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.model_name = model_name

        self.model = HuggingFaceEmbedding(
            self.model_name,
            device=device,
            normalize=True,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

        self.vector_index = None
        self.retriever = None

    def index(self, items: list[RagChunk] | list[TextNode]) -> None:
        self.nodes = self._to_nodes(items)

        vector_index = VectorStoreIndex(
            nodes=self.nodes,
            embed_model=self.model,
            show_progress=False
        )
        vector_retriever = vector_index.as_retriever(similarity_top_k=self.top_k)
        self.retriever = vector_retriever
        self.vector_index = vector_index

    def retrieve(self, query: str) -> RetrievalResults:
        if self.vector_index is None:
            raise AttributeError("Please call index() before retrieving")
        if self.retriever is None:
            raise AttributeError("Please call index() before retrieving")
        query = query.strip()
        if not query:
            raise RuntimeError("Query cannot be empty")

        nodes_with_score = self.retriever.retrieve(query)

        retrieved_nodes = [item.node for item in nodes_with_score]
        retrieved_scores = [float(item.score) if item.score is not None else 0.0 for
                            item in nodes_with_score]
        context_nodes = self._get_context_nodes(retrieved_nodes)

        return RetrievalResults(
            results=retrieved_nodes,
            scores=retrieved_scores,
            context_chunks=context_nodes
        )
