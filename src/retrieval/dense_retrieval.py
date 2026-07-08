import torch
from sentence_transformers import SentenceTransformer

from src.rag_chunker import RagChunk
from src.retrieval import RetrievalResults


class DenseRetriever:

    def __init__(
            self,
            model_name: str = "intfloat/multilingual-e5-small",
            expansion_strategy: str = "none",
            use_section: bool = True,
            top_k: int = 5
    ):
        allowed_strategy = {
            "none", "neighbors", "first_section", "all_sections"
        }
        if expansion_strategy not in allowed_strategy:
            raise ValueError(f"Expansion strategy {expansion_strategy} is not supported.")

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=device)

        self.chunks = None
        self.embeddings = None

        self.expansion_strategy = expansion_strategy
        self.use_section = use_section
        self.top_k = top_k

    def _build_text(self, chunk: RagChunk) -> str:
        if self.use_section:
            text = " ".join([
                chunk.section or '',
                chunk.content or '',
            ])
        else:
            text = chunk.content or ""

        if "e5" in self.model_name.lower():
            text = "passage: " + text
        return text

    def _build_query(self, query: str) -> str:
        if "e5" in self.model_name.lower():
            return "query: " + query
        return query

    def index(self, chunks: list[RagChunk]) -> None:
        self.chunks = chunks
        corpus = [
            self._build_text(chunk) for chunk in chunks
        ]

        embeddings = self.model.encode(
            corpus,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        self.embeddings = embeddings

    def retrieve(self, query: str) -> RetrievalResults:
        if self.chunks is None:
            raise AttributeError("Please call index() before retrieving")

        query_text = self._build_query(query)

        query_embedding = self.model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        scores = query_embedding @ self.embeddings.T
        top_scores, top_indices = torch.topk(scores, k=self.top_k)

        retrieved_chunks = []
        retrieved_scores = []
        for idx, score in zip(top_indices, top_scores):
            chunk = self.chunks[idx.item()]
            retrieved_chunks.append(chunk)
            retrieved_scores.append(score.item())
        context_chunks = self._get_context_chunks(retrieved_chunks)

        return RetrievalResults(
            results=retrieved_chunks,
            scores=retrieved_scores,
            context_chunks=context_chunks
        )

    def _get_context_chunks(self, result_chunks: list[RagChunk]) -> list[RagChunk]:
        if self.expansion_strategy == "none":
            return result_chunks
        elif self.expansion_strategy == "first_section" or self.expansion_strategy == "all_section":
            relevant_chunks = self._get_section_chunks(result_chunks)
        elif self.expansion_strategy == "neighbors":
            relevant_chunks = self._get_neighbours_chunks(result_chunks)
        else:
            raise ValueError("Unknown expansion strategy: ", self.expansion_strategy)
        return relevant_chunks

    def _get_neighbours_chunks(
            self,
            result_chunks: list[RagChunk],
    ) -> list[RagChunk]:
        if self.chunks is None:
            raise AttributeError("Please call index() before retrieving")

        chunk_id_to_idx = {
            chunk.chunk_id: idx
            for idx, chunk in enumerate(self.chunks)
        }
        selected_indices = set()
        for chunk in result_chunks:

            idx = chunk_id_to_idx.get(chunk.chunk_id)
            if idx is None:
                continue

            for neighbour_idx in [idx - 1, idx, idx + 1]:
                if 0 <= neighbour_idx < len(self.chunks):
                    neighbour = self.chunks[neighbour_idx]

                    same_source = neighbour.source == chunk.source

                    if same_source:
                        selected_indices.add(neighbour_idx)
        return [
            self.chunks[idx]
            for idx in sorted(selected_indices)

        ]

    def _get_section_chunks(
            self,
            result_chunks: list[RagChunk],
    ) -> list[RagChunk]:
        if self.chunks is None:
            raise AttributeError("Please call index() before retrieving")

        if not result_chunks:
            return []

        if self.expansion_strategy == "first_section":
            first_chunk = result_chunks[0]

            if first_chunk.section is None or first_chunk.section == "":
                return self._get_neighbours_chunks(result_chunks)

            relevant_section_keys = {
                (first_chunk.source, first_chunk.section)
            }

        elif self.expansion_strategy == "all_section":
            chunks_with_sections = [
                chunk
                for chunk in result_chunks
                if chunk.section is not None and chunk.section != ""
            ]

            if not chunks_with_sections:
                return self._get_neighbours_chunks(result_chunks)

            relevant_section_keys = {
                (chunk.source, chunk.section)
                for chunk in chunks_with_sections
            }

        else:
            return result_chunks

        relevant_chunks = [
            chunk
            for chunk in self.chunks
            if (chunk.source, chunk.section) in relevant_section_keys
        ]

        return relevant_chunks