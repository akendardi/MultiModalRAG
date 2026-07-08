from bm25s import BM25, tokenize
from bm25s.stopwords import STOPWORDS_RUSSIAN
from src.rag_chunker import RagChunk
from .retrieval_data_models import RetrievalResults


class BM25Retriever:
    #expansion_strategy: none, neighbors, first_section, all_section
    def __init__(self, expansion_strategy: str = "none", use_section: bool = True, top_k: int = 5):

        allowed_strategy = {
            "none", "neighbors", "first_section", "all_sections"
        }
        if expansion_strategy not in allowed_strategy:
            raise ValueError(f"Expansion strategy {expansion_strategy} is not supported.")

        self.bm25 = BM25()
        self.chunks = None
        self.top_k = top_k
        self.use_section = use_section
        self.expansion_strategy = expansion_strategy

    def _build_text(self, chunk: RagChunk) -> str:
        if self.use_section:
            return " ".join([
                chunk.section or '',
                chunk.content or '',
            ])
        return chunk.content or ""

    def index(self, chunks: list[RagChunk]) -> None:

        corpus = [self._build_text(chunk) for chunk in chunks]
        corpus_tokens = tokenize(corpus, stopwords=STOPWORDS_RUSSIAN, show_progress=False)

        self.chunks = chunks
        self.bm25.index(corpus_tokens, show_progress=False)

    def retrieve(self, query: str) -> RetrievalResults:

        if self.chunks is None:
            raise AttributeError("Please call index() before retrieving")

        query_tokens = tokenize(query, stopwords=STOPWORDS_RUSSIAN, show_progress=False)
        results, scores = self.bm25.retrieve(query_tokens, corpus=self.chunks, k=self.top_k)

        retrieved_chunks = list(results[0])
        retrieved_scores = [float(score) for score in scores[0]]
        context_chunks = self._get_context_chunks(retrieved_chunks)

        return RetrievalResults(
            results=retrieved_chunks,
            scores=retrieved_scores,
            context_chunks=context_chunks,
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