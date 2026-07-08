from dataclasses import dataclass, field

from src.text_processing.pdf_extracting import DocumentType


@dataclass(frozen=True)
class ExtraInformation:
    path: str | None
    content: str | None
    type: DocumentType


@dataclass
class RagChunk:
    source: str
    chunk_id: str
    content: str
    section: str | None
    pages: list[int]
    content_types: list[DocumentType]
    asset_paths: list[ExtraInformation] = field(default_factory=list)