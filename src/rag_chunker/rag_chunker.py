import re
from collections import defaultdict
from src.rag_chunker.data_models import RagChunk, ExtraInformation

from chonkie import (
    BaseChunker,
    Chunk,
    RecursiveLevel,
    RecursiveRules,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
    Visualizer,
)

from src.text_processing.pdf_extracting import DocumentType, Metadata




class RagChunker:

    def __init__(
            self,
            strategy: str = "token",
            chunk_size: int = 1000,
            overlap_size: int = 200
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def _get_default_recursive_rules(self) -> RecursiveRules:
        return RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=["\n\n"]),
                RecursiveLevel(delimiters=["\n"]),
                RecursiveLevel(delimiters=[". ", ".\n", "! ", "? "]),
                RecursiveLevel(whitespace=True),
            ]
        )

    def _get_main_section(
            self,
            relevant_docs: list[Metadata],
            chunk_start: int,
            chunk_end: int,
    ) -> str | None:
        section_scores = defaultdict(int)
        for doc in relevant_docs:
            if doc.section is None:
                continue
            if doc.start_idx is None or doc.end_idx is None:
                continue

            start = max(chunk_start, doc.start_idx)
            end = min(chunk_end, doc.end_idx)
            overlap = max(0, end - start)

            section_scores[doc.section] += overlap

        if not section_scores:
            return None

        return max(section_scores.items(), key=lambda x: x[1])[0]

    def get_chunks(
            self,
            documents: list[Metadata],
    ) -> list[RagChunk]:
        text, documents = self._format_text(documents)
        chunker = self._get_chunker_by_strategy(self.strategy, self.chunk_size, self.overlap_size)
        chunks = chunker.chunk(text)
        last_idx = 0
        rag_chunks = []

        for chunk in chunks:
            relevant_docs, idx = self._get_documents_from_chunk(chunk, documents, last_idx)
            if not relevant_docs:
                continue
            last_idx = idx
            source = relevant_docs[0].source
            pages = sorted(set(i.page for i in relevant_docs))
            content_types = list(dict.fromkeys(i.content_type for i in relevant_docs))
            asset_paths = self._get_asset_paths(relevant_docs)
            section = self._get_main_section(relevant_docs, chunk.start_index, chunk.end_index)
            start_page = min(pages)
            end_page = max(pages)
            rag_chunk = RagChunk(
                source=source,
                chunk_id=f"{source}:{start_page}-{end_page}:{chunk.id}",
                content=chunk.text,
                section=section,
                pages=pages,
                content_types=content_types,
                asset_paths=asset_paths
            )
            rag_chunks.append(rag_chunk)
        return rag_chunks

    def visualize_chunks(self, chunks: list[RagChunk]) -> None:
        visualizer = Visualizer()
        text = [chunk.content for chunk in chunks]
        visualizer.print(text)

    def _get_chunker_by_strategy(
            self,
            strategy: str = "token",
            chunk_size: int = 1000,
            overlap_size: int = 200,
            rules: RecursiveRules | None = None,
    ) -> BaseChunker:
        strategy = strategy.lower().strip()

        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть больше 0")

        if overlap_size < 0:
            raise ValueError("overlap_size не может быть отрицательным")

        if overlap_size >= chunk_size:
            raise ValueError("overlap_size должен быть меньше chunk_size")

        if strategy in {"token", "tokens"}:
            return TokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
            )

        if strategy in {"recursive", "rec"}:
            if rules is None:
                rules = self._get_default_recursive_rules()

            return RecursiveChunker(
                chunk_size=chunk_size,
                rules=rules,
            )

        if strategy in {"sentence", "sent"}:
            return SentenceChunker(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
            )

        if strategy in {"semantic", "sem"}:
            return SemanticChunker(
                chunk_size=chunk_size,
            )

        raise ValueError(
            f"Некорректная стратегия чанкинга: {strategy}. "
            f"Доступные стратегии: token, recursive, sentence, semantic"
        )

    def _get_documents_from_chunk(
            self,
            chunk: Chunk,
            documents: list[Metadata],
            last_idx: int = 0,
    ) -> tuple[list[Metadata], int]:
        ch_st = chunk.start_index
        ch_end = chunk.end_index
        relevant_docs = []
        idx = last_idx
        for i in range(last_idx, len(documents)):
            document = documents[i]
            if document.start_idx is None or document.end_idx is None:
                continue
            if document.start_idx >= ch_end:
                break
            if document.end_idx <= ch_st:
                continue
            if not relevant_docs:
                idx = i
            relevant_docs.append(document)
        return relevant_docs, idx

    def _format_text(
            self,
            documents: list[Metadata],
            separator: str = '\n\n'
    ) -> tuple[str, list[Metadata]]:
        cursor = 0
        blocks = []
        formatted_documents = []
        for document in documents:
            formatted_text = self._format_document(document)
            if not formatted_text:
                continue
            if blocks:
                blocks.append(separator)
                cursor += len(separator)
            document.start_idx = cursor
            cursor += len(formatted_text)
            document.end_idx = cursor
            blocks.append(formatted_text)
            formatted_documents.append(document)
        return "".join(blocks), formatted_documents

    def _format_document(self, document: Metadata) -> str:
        if document.content is None:
            return ""
        if document.content_type == DocumentType.PARAGRAPH:
            return f"{document.content}"
        elif document.content_type == DocumentType.CHART:
            return f"[График]{document.asset_path}\nОписание: {document.content}"
        elif document.content_type == DocumentType.TABLE:
            return f"[Таблица]{document.asset_path}\nОписание: {document.content}"
        elif document.content_type == DocumentType.IMAGE:
            return f"[Рисунок]{document.asset_path}\nОписание: {document.content}"
        elif document.content_type == DocumentType.FORMULA:
            formula_content = document.content.replace("\n", "")
            return f"[Формула]{document.asset_path}\nОписание: {formula_content}"
        return ""

    def _get_asset_paths(self, documents: list[Metadata]) -> list[ExtraInformation]:
        asset_paths = []
        seen_assets = set()
        for document in documents:
            if document.content_type == DocumentType.PARAGRAPH:
                continue

            asset = ExtraInformation(
                path=document.asset_path,
                content=document.content,
                type=document.content_type,
            )
            if asset in seen_assets:
                continue

            seen_assets.add(asset)
            asset_paths.append(asset)
        return asset_paths

    def _get_extra_information_from_text(self, text: str) -> list[tuple[str, str]]:
        pattern = (
            r"\[(?:Формула|График|Рисунок|Таблица)\]"
            r"(?P<path>[^\n]*)\nОписание:\s*(?P<description>[^\n]*)"
        )
        extra_information = []
        for match in re.finditer(pattern, text):
            path = match.group("path").strip()
            description = match.group("description").strip().replace("$$", "\n$$\n")
            extra_information.append((path, description))
        return extra_information
