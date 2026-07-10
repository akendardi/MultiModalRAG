import re

from bm25s import BM25, tokenize
from bm25s.stopwords import STOPWORDS_RUSSIAN
from llama_index.core.schema import TextNode
from pymorphy3 import MorphAnalyzer

from src.rag_chunker import RagChunk
from .base_retriever import BaseRetriever
from .retrieval_data_models import RetrievalResults


class BM25Retriever(BaseRetriever):

    def __init__(
            self,
            insert_metadata_into_text: bool = True,
            expansion_strategy: str = "none",
            top_k: int = 5,
            use_lemmatization: bool = True,
    ):
        super().__init__(insert_metadata_into_text, expansion_strategy, top_k)

        self.text_normalizer = _BM25TextNormalizer(
            use_lemmatization=use_lemmatization
        )
        self.bm25 = BM25()


    def _build_text(self, node: TextNode) -> str:
        content = node.text
        content = self.text_normalizer.preprocess_text(content)
        return content

    def index(self, items: list[RagChunk] | list[TextNode]) -> None:

        self.nodes = self._to_nodes(items)
        corpus = [self._build_text(node) for node in self.nodes]
        corpus_tokens = tokenize(corpus, stopwords=STOPWORDS_RUSSIAN, show_progress=False)

        self.bm25.index(corpus_tokens, show_progress=False)

    def retrieve(self, query: str) -> RetrievalResults:

        if self.nodes is None:
            raise AttributeError("Please call index() before retrieving")

        query = self.text_normalizer.preprocess_text(query)

        query_tokens = tokenize(query, stopwords=STOPWORDS_RUSSIAN, show_progress=False)
        results, scores = self.bm25.retrieve(
            query_tokens,
            corpus=self.nodes,
            k=min(self.top_k, len(self.nodes)),
            show_progress=False,
        )

        retrieved_nodes = list(results[0])
        retrieved_scores = [float(score) for score in scores[0]]
        context_nodes = self._get_context_nodes(retrieved_nodes)

        return RetrievalResults(
            results=retrieved_nodes,
            scores=retrieved_scores,
            context_chunks=context_nodes,
        )


class _BM25TextNormalizer:
    def __init__(
            self,
            use_lemmatization: bool = True,
    ):
        self.use_lemmatization = use_lemmatization
        self.morph = MorphAnalyzer() if use_lemmatization else None

    def _lemmatize_text(self, text: str) -> str:
        if not self.use_lemmatization or self.morph is None:
            return text
        words = text.split()
        lemmas = [
            self.morph.parse(word)[0].normal_form
            for word in words
        ]

        return " ".join(lemmas)

    def _normalize_text(self, text: str | None) -> str:
        if text is None:
            return ""
        text = str(text).lower()

        text = text.replace("ё", "е")
        text = re.sub(r"[^а-яa-z0-9\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess_text(self, text: str | None) -> str:
        text = self._normalize_text(text)
        text = self._lemmatize_text(text)
        return text
