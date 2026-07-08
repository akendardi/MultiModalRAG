from .pdf_extracting import (
    ContextBuilder,
    DocumentType,
    HeaderPretend,
    HeadersProcessor,
    Metadata,
    MineruLoader,
    MineruParser,
    MineruReader,
)

__all__ = [
    "ContextBuilder",
    "DocumentType",
    "HeaderPretend",
    "HeadersEvaluator",
    "HeadersExtractor",
    "HeadersProcessor",
    "Metadata",
    "MineruLoader",
    "MineruParser",
    "MineruReader",
    "PDFExtractor",
]


def __getattr__(name):
    if name == "HeadersEvaluator":
        from .headers_evaluator import HeadersEvaluator

        return HeadersEvaluator

    if name == "HeadersExtractor":
        from .pdf_extracting import HeadersExtractor

        return HeadersExtractor

    if name == "PDFExtractor":
        from .pdf_extracting import PDFExtractor

        return PDFExtractor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
