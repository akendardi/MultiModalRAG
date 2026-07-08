from pathlib import Path

from src.text_processing.pdf_extracting.document_models import Metadata
from src.text_processing.pdf_extracting.headers.headers_processor import HeadersProcessor
from src.text_processing.pdf_extracting.headers.section_context_builder import ContextBuilder
from src.text_processing.pdf_extracting.mineru import MineruParser, MineruReader


class PDFExtractor:
    """
    Собирает документы из PDF и при необходимости проставляет контекст разделов.
    """

    def __init__(self):
        """
        Инициализирует внутренние компоненты для извлечения PDF.
        """
        from src.text_processing.pdf_extracting.headers.headers_extractor import (
            HeadersExtractor,
        )

        self.mineru_reader = MineruReader()
        self.mineru_parser = MineruParser()
        self.headers_extractor = HeadersExtractor(self.mineru_reader)
        self.headers_processor = HeadersProcessor()
        self.context_builder = ContextBuilder()

    def get_documents_from_pdf(
            self,
            pdf_path: str | Path,
            use_context: bool = True,
            headers_source: str = "golden",
            clean_headers: bool = True,
            keep_abstract_before_first_section: bool = True,
            remove_after_references: bool = True,
    ) -> list[Metadata]:
        """
        Извлекает структурированные Metadata из PDF.

        :param pdf_path: путь к PDF-документу
        :param use_context: проставлять ли section в Metadata
        :param headers_source: источник заголовков: golden, mineru или mupdf
        :param clean_headers: очищать ли заголовки через HeadersProcessor
        :param keep_abstract_before_first_section: оставить ли аннотацию до первого раздела
        :param remove_after_references: удалить ли библиографию и всё после неё
        :return: список Metadata
        """
        pdf_path = Path(pdf_path)
        json_file = self.mineru_reader.get_mineru_doc(str(pdf_path))
        json_file = self.mineru_parser.do_lists_to_paragraph(json_file)
        documents = self.mineru_parser.parse_json(
            json_file=json_file,
            source=pdf_path.stem,
        )

        if not use_context:
            return self.context_builder.remove_context(documents)

        headers = self.get_headers(
            pdf_path=pdf_path,
            headers_source=headers_source,
            clean_headers=clean_headers,
        )

        return self.context_builder.add_context(
            documents=documents,
            headers=headers,
            keep_abstract_before_first_section=keep_abstract_before_first_section,
            remove_after_references=remove_after_references,
        )

    def get_headers(
            self,
            pdf_path: str | Path,
            headers_source: str = "golden",
            clean_headers: bool = True,
    ) -> list[str]:
        """
        Получает заголовки из выбранного источника.

        :param pdf_path: путь к PDF-документу
        :param headers_source: источник заголовков: golden, mineru или mupdf
        :param clean_headers: очищать ли заголовки через HeadersProcessor
        :return: список заголовков
        """
        headers_source = headers_source.lower().strip()

        if headers_source in ("gold", "golden"):
            return self.headers_extractor.get_headers_by_golden(pdf_path)

        if headers_source == "mineru":
            headers = self.headers_extractor.get_headers_by_mineru(str(pdf_path))

        elif headers_source == "mupdf":
            headers = self.headers_extractor.get_headers_by_mupdf(str(pdf_path))

        else:
            raise ValueError(
                f"Неизвестный headers_source={headers_source}. "
                f"Доступные значения: golden, mineru, mupdf"
            )

        if clean_headers:
            return self.headers_processor.clear_headers(headers)

        return headers
