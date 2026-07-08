from .document_models import DocumentType, HeaderPretend, Metadata
from .headers import ContextBuilder, HeadersProcessor
from .mineru import MineruLoader, MineruParser, MineruReader

__all__ = [
    "ContextBuilder",
    "DocumentType",
    "HeaderPretend",
    "HeadersExtractor",
    "HeadersProcessor",
    "Metadata",
    "MineruLoader",
    "MineruParser",
    "MineruReader",
    "PDFExtractor",
]


def __getattr__(name):
    if name == "HeadersExtractor":
        from .headers import HeadersExtractor

        return HeadersExtractor

    if name == "PDFExtractor":
        from .pdf_extractor import PDFExtractor

        return PDFExtractor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
