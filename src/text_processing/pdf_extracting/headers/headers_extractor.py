import json
import re
from pathlib import Path

import fitz
from src.paths import PROJECT_ROOT
from src.text_processing.pdf_extracting.mineru import MineruReader


class HeadersExtractor:

    def __init__(
            self,
            mineru_reader: MineruReader,
            golden_headers_path: str | Path | None = None,
    ):
        self.mineru_reader = mineru_reader
        self.golden_headers_path = (
            Path(golden_headers_path)
            if golden_headers_path
            else PROJECT_ROOT / "data" / "gold_headers.json"
        )

    def get_headers_by_mineru(self, file_path: str) -> list[str]:
        """
        Возвращает список заголовков из title-блоков MinerU.

        :param file_path: путь к PDF-документу
        :return: список строк-заголовков
        """
        json_file = self.mineru_reader.get_mineru_doc(file_path)

        headers = []

        for page in json_file:
            for block in page:
                block_type = block.get("type", "")

                if block_type != "title":
                    continue

                content = block.get("content", {}).get("title_content", "")
                text = self._join_inline_content(content)

                if text:
                    headers.append(text)

        return headers

    def get_headers_by_mupdf(self, file_path: str) -> list[str]:
        """
        Возвращает список заголовков, извлеченных MuPdf.

        :param file_path: путь к PDF-документу
        :return: список строк-заголовков
        """
        bold_titles = []

        with fitz.open(file_path) as file:
            for page_num, page in enumerate(file, start=1):
                data = page.get_text("dict")

                for block in data["blocks"]:
                    if block["type"] != 0:
                        continue

                    for line in block["lines"]:
                        spans = [
                            span for span in line["spans"]
                            if span["text"].strip()
                        ]

                        if not spans:
                            continue

                        line_text = " ".join(
                            span["text"].strip()
                            for span in spans
                        )

                        line_text = self._normalize_text(line_text)

                        if not line_text:
                            continue

                        is_bold_line = any(
                            "bold" in span["font"].lower()
                            or span["flags"] & 2 ** 4
                            or span["flags"] & 2
                            for span in spans
                        )

                        max_size = max(span["size"] for span in spans)
                        is_big_line = max_size >= 13.5

                        if not (is_bold_line or is_big_line):
                            continue

                        bbox = line["bbox"]

                        if bold_titles:
                            old_el = bold_titles[-1]
                            old_text, old_bbox, old_page = old_el

                            if (
                                old_page == page_num
                                and abs(bbox[1] - old_bbox[3]) <= 5
                            ):
                                new_el = [
                                    f"{old_text} {line_text}",
                                    [
                                        min(old_bbox[0], bbox[0]),
                                        min(old_bbox[1], bbox[1]),
                                        max(old_bbox[2], bbox[2]),
                                        max(old_bbox[3], bbox[3]),
                                    ],
                                    page_num,
                                ]
                                bold_titles[-1] = new_el
                                continue

                        bold_titles.append([line_text, bbox, page_num])

        headers = []

        for text, _, _ in bold_titles:
            text = self._normalize_text(text)

            if text:
                headers.append(text)

        return headers

    def get_headers_by_golden(self, file_path: str | Path) -> list[str]:
        """
        Возвращает эталонные заголовки из golden-файла.

        :param file_path: путь к PDF-документу или имя PDF-файла
        :return: список эталонных заголовков
        """
        document_name = Path(file_path).name
        golden_data = self._load_golden_headers()

        for item in golden_data:
            if item.get("document") == document_name:
                return item.get("headers", [])

        return []

    def _join_inline_content(self, contents: list[dict]) -> str:

        """
        Объединяет inline-элементы MinerU в одну текстовую строку.

        MinerU часто хранит содержимое абзаца не одной строкой, а списком
        отдельных элементов: обычный текст, inline-формулы, ссылки и т.п.
        Метод проходит по этим элементам, очищает их через clean_text(),
        inline-формулы оборачивает в LaTeX-формат $...$, после чего
        объединяет всё в одну строку.

        :param contents: список inline-элементов MinerU
        :return: очищенная строка с объединённым текстом
        """
        parts = []

        for part in contents:
            part_type = part.get("type", "")
            content = part.get("content", "")

            content = self._clean_text(content)

            if not content:
                continue

            if part_type == "equation_inline":
                content = f"${content}$"

            parts.append(content.strip())

        text = " ".join(parts)
        return self._clean_text(text)

    def _load_golden_headers(self) -> list[dict]:
        """
        Загружает JSON-файл с эталонными заголовками.

        :return: список документов с ручной разметкой заголовков
        """
        with open(self.golden_headers_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _clean_text(self, text: str) -> str:
        """
        Нормализует текст: убирает лишние пробелы, пробелы перед знаками
        препинания, ссылки вида [1] и служебные верхние индексы.

        :param text: текст для очистки
        :return: очищенный текст
        """
        text = re.sub(r"\s+", " ", str(text))
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(
            r"\[\s*\d+(?:\s*[-–—]\s*\d+)?"
            r"(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*\s*\]",
            "",
            text,
        )

        if re.fullmatch(r"^\^\{.+}$", text):
            return ""

        return text.strip()

    def _normalize_text(self, text: str) -> str:
        """
        Нормализует текст, извлечённый из PDF.

        :param text: исходный текст
        :return: нормализованный текст
        """
        text = re.sub(r"\s+", " ", str(text))
        text = text.replace("\u00a0", " ")
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        return self._clean_text(text)
