from src.rag_chunker import RagChunk
from .bm25 import BM25Retriever
from .dense_retrieval import DenseRetriever
from .retrieval_data_models import RetrievalResults
from .base_retriever import BaseRetriever

from sklearn.preprocessing import MinMaxScaler
from llama_index.core.schema import TextNode

class HybridRetriever(BaseRetriever):
    ALLOWED_HYBRID_STRATEGIES = {"rrf", "weighted"}

    def __init__(
            self,
            hybrid_strategy: str = "rrf",
            top_k: int = 5,
            candidate_k: int = 20,
            alpha: float = 0.5,
            rrf_k: int = 60,
            dense_model: str = "intfloat/multilingual-e5-small",
            query_instruction: str = "query: ",
            text_instruction: str = "passage: ",
            insert_metadata_into_text: bool = True,
            expansion_strategy: str = "none",
            use_bm25_lemmatization: bool = True,
    ):
        super().__init__(
            insert_metadata_into_text=insert_metadata_into_text,
            expansion_strategy=expansion_strategy,
            top_k=top_k
        )
        if hybrid_strategy not in self.ALLOWED_HYBRID_STRATEGIES:
            raise ValueError(f"Hybrid strategy {hybrid_strategy} is not supported.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha {alpha} is not in [0, 1].")
        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        self.bm25 = BM25Retriever(
            expansion_strategy="none",
            insert_metadata_into_text=insert_metadata_into_text,
            top_k=candidate_k,
            use_lemmatization=use_bm25_lemmatization
        )
        self.dense = DenseRetriever(
            expansion_strategy="none",
            model_name=dense_model,
            insert_metadata_into_text=insert_metadata_into_text,
            top_k=candidate_k,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

        self.alpha = alpha
        self.rrf_k = rrf_k
        self.hybrid_strategy = hybrid_strategy


    def index(self, items: list[RagChunk] | list[TextNode]) -> None:
        if not items:
            raise ValueError("Cannot index empty list")
        nodes = self._to_nodes(items)
        self.bm25.index(nodes)
        self.dense.index(nodes)
        self.nodes = nodes

    def retrieve(
            self,
            query: str,
    ) -> RetrievalResults:
        if self.nodes is None:
            raise AttributeError("Please call index() before retrieving")

        query = query.strip()
        if not query:
            raise ValueError("Cannot retrieve empty query")

        bm_25_res = self.bm25.retrieve(query)
        dense_res = self.dense.retrieve(query)

        if self.hybrid_strategy == "rrf":
            return self._rrf(bm_25_res, dense_res)
        elif self.hybrid_strategy == "weighted":
            return self._weighted(bm_25_res, dense_res)
        else:
            raise ValueError(f"Hybrid strategy {self.hybrid_strategy} is not supported.")

    def _weighted(self, bm_25_res: RetrievalResults, dense_res: RetrievalResults) -> RetrievalResults:

        chunk_by_id: dict[str, TextNode] = {}
        fused_scores: dict[str, float] = {}

        normalized_bm25_scores = self.min_max_scaling(bm_25_res.scores)
        normalized_dense_scores = self.min_max_scaling(dense_res.scores)

        for chunk, score in zip(bm_25_res.results, normalized_bm25_scores):
            chunk_id = self._get_chunk_id(chunk)
            chunk_by_id[chunk_id] = chunk

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1 - self.alpha) * score

        for chunk, score in zip(dense_res.results, normalized_dense_scores):
            chunk_id = self._get_chunk_id(chunk)
            chunk_by_id[chunk_id] = chunk

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + self.alpha * score

        return self._build_results(chunk_by_id, fused_scores)


    def _rrf(self, bm_25_res: RetrievalResults, dense_res: RetrievalResults) -> RetrievalResults:

        chunk_by_id: dict[str, TextNode] = {}
        fused_scores: dict[str, float] = {}

        for rank, chunk in enumerate(bm_25_res.results, start=1):
            chunk_id = self._get_chunk_id(chunk)
            chunk_by_id[chunk_id] = chunk

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, chunk in enumerate(dense_res.results, start=1):
            chunk_id = self._get_chunk_id(chunk)
            chunk_by_id[chunk_id] = chunk

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        return self._build_results(chunk_by_id, fused_scores)

    def min_max_scaling(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0]
        if min(scores) == max(scores):
            return [1.0] * len(scores)
        scaler = MinMaxScaler()
        normalized_scores = scaler.fit_transform([[score] for score in scores])
        return normalized_scores.ravel().tolist()


    def _build_results(self, chunk_by_id: dict[str, TextNode], fused_scores: dict[str, float]) -> RetrievalResults:
        sorted_chunks_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:self.top_k]
        retrieved_nodes = [chunk_by_id[idx] for idx in sorted_chunks_ids]
        retrieved_scores = [fused_scores[idx] for idx in sorted_chunks_ids]
        context_nodes = self._get_context_nodes(retrieved_nodes)

        return RetrievalResults(
            results=retrieved_nodes,
            scores=retrieved_scores,
            context_chunks=context_nodes
        )
