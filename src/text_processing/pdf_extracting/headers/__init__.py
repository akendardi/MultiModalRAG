from .headers_processor import HeadersProcessor
from .section_context_builder import ContextBuilder

__all__ = [
    "ContextBuilder",
    "HeadersExtractor",
    "HeadersProcessor",
]


def __getattr__(name):
    if name == "HeadersExtractor":
        from .headers_extractor import HeadersExtractor

        return HeadersExtractor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
