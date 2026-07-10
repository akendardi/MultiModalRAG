from dataclasses import dataclass
from llama_index.core.schema import TextNode


@dataclass
class RetrievalResults:
    results: list[TextNode]
    scores: list[float]
    context_chunks: list[TextNode]
