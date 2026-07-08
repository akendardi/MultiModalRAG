from dataclasses import dataclass
from src.rag_chunker import RagChunk
@dataclass
class RetrievalResults:
    results: list[RagChunk]
    scores: list[float]
    context_chunks: list[RagChunk]