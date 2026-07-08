import re
from copy import deepcopy

from src.text_processing.pdf_extracting.document_models import DocumentType, Metadata


class MineruParser:

    def parse_json(self, json_file, source: str = "") -> list[Metadata]:
        """
        Преобразует JSON MinerU в список структурированных объектов Metadata.

        Метод обрабатывает текстовые абзацы, формулы, изображения и графики,
        сохраняет для каждого элемента страницу, секцию, тип содержимого,
        путь к связанному файлу и уникальный chunk_id. Также метод отслеживает
        текущую секцию документа по заголовкам, найденным в исходном PDF.

        :param json_file: JSON-документ MinerU в виде списка страниц и блоков
        :param source: название или путь источника документа
        :return: список объектов Metadata
        """
        documents = []

        block_id = 0
        for page_num, page in enumerate(json_file, start=1):
            old_desc = []
            for block in page:
                block_type = block["type"]

                if block_type == "paragraph":
                    content = block.get('content', {}).get("paragraph_content", [])
                    content = self._join_inline_content(content)

                    if not content:
                        continue
                    documents.append(Metadata(
                        source=source,
                        page=page_num,
                        section=None,
                        content_type=DocumentType.PARAGRAPH,
                        chunk_id=f"{source}:{page_num}:{block_id}",
                        asset_path=None,
                        content=content
                    ))

                elif block_type == "code":
                    content = block.get('content', {}).get("code_content", [])
                    content = self._join_inline_content(content)

                    if not content:
                        continue

                    documents.append(Metadata(
                        source=source,
                        page=page_num,
                        section=None,
                        content_type=DocumentType.PARAGRAPH,
                        chunk_id=f"{source}:{page_num}:{block_id}",
                        asset_path=None,
                        content=content
                    ))


                elif block_type == "equation_interline":
                    content = block.get('content', {}).get("math_content", "")
                    content = re.sub(r"\\\\tag\{\d+}$", "", content)
                    asset_img = block.get('content', {}).get('image_source', {}).get('path', "")

                    if not content:
                        continue

                    content = f"\n$$\n{content}\n$$\n"

                    documents.append(Metadata(
                        source=source,
                        page=page_num,
                        section=None,
                        content_type=DocumentType.FORMULA,
                        chunk_id=f"{source}:{page_num}:{block_id}",
                        asset_path=asset_img if asset_img else None,
                        content=content
                    ))

                elif block_type == "image":
                    asset_img = block.get('content', {}).get('image_source', {}).get('path', "")
                    image_desc = self._join_inline_content(block.get('content', {}).get("image_caption", {}))

                    if self._is_subfigure_caption(image_desc):
                        old_desc.append(Metadata(
                            source=source,
                            page=page_num,
                            section=None,
                            content_type=DocumentType.IMAGE,
                            chunk_id=f"{source}:{page_num}:{block_id}",
                            asset_path=asset_img if asset_img else None,
                            content=image_desc
                        ))
                    else:
                        if old_desc:
                            for old in old_desc:
                                old.content = self._clean_image_caption(f"{image_desc} {old.content}")
                                documents.append(old)
                            old_desc = []

                        documents.append(Metadata(
                            source=source,
                            page=page_num,
                            section=None,
                            content_type=DocumentType.IMAGE,
                            chunk_id=f"{source}:{page_num}:{block_id}",
                            asset_path=asset_img if asset_img else None,
                            content=image_desc
                        ))

                elif block_type == "chart":
                    asset_img = block.get('content', {}).get('image_source', {}).get('path', "")
                    image_desc = self._join_inline_content(block.get('content', {}).get("chart_caption", {}))

                    if self._is_subfigure_caption(image_desc):
                        old_desc.append(Metadata(
                            source=source,
                            page=page_num,
                            section=None,
                            content_type=DocumentType.CHART,
                            chunk_id=f"{source}:{page_num}:{block_id}",
                            asset_path=asset_img if asset_img else None,
                            content=image_desc
                        ))
                    else:
                        if old_desc:
                            for old in old_desc:
                                old.content = self._clean_image_caption(f"{image_desc} {old.content}")
                                documents.append(old)
                            old_desc = []

                        documents.append(Metadata(
                            source=source,
                            page=page_num,
                            section=None,
                            content_type=DocumentType.CHART,
                            chunk_id=f"{source}:{page_num}:{block_id}",
                            asset_path=asset_img if asset_img else None,
                            content=image_desc
                        ))

                elif block_type == "table":
                    content = block.get("content", {})
                    asset_img = content.get("image_source", {}).get("path", "")
                    table_caption = self._join_inline_content(content.get("table_caption", []))
                    table_footnote = self._join_inline_content(content.get("table_footnote", []))
                    table_html = content.get("html", "")
                    table_text = self._table_block_to_text(
                        caption=table_caption,
                        html=table_html,
                        footnote=table_footnote,
                    )

                    if not table_text:
                        continue

                    documents.append(Metadata(
                        source=source,
                        page=page_num,
                        section=None,
                        content_type=DocumentType.TABLE,
                        chunk_id=f"{source}:{page_num}:{block_id}",
                        asset_path=asset_img if asset_img else None,
                        content=table_text
                    ))

                elif block_type == "title":
                    content = block.get("content", {}).get("title_content", "")
                    content = self._join_inline_content(content)

                    if not content:
                        continue

                    documents.append(Metadata(
                        source=source,
                        page=page_num,
                        section=None,
                        content_type=DocumentType.TITLE,
                        chunk_id=f"{source}:{page_num}:{block_id}",
                        asset_path=None,
                        content=content
                    ))
                else:
                    pass
                block_id += 1
        return documents

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

    def _clean_text(self, text: str) -> str:
        """
        Нормализует текст: убирает лишние пробелы, пробелы перед знаками
        препинания, ссылки вида [1] и служебные верхние индексы.

        :param text: текст для очистки
        :return: очищенный текст
        """
        text = re.sub(r'\s+', " ", text)
        text = re.sub(r'\s+([.,;:!?])', r"\1", text)
        text = re.sub(r"\[\s*\d+(?:\s*[-–—]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*\s*]", "", text)
        if re.fullmatch(r'^\^\{.+}$', text):
            text = ""
        return text

    def _is_subfigure_caption(self, text: str) -> bool:
        """
        Проверяет, является ли подпись обозначением подрисунка.

        Иногда MinerU выделяет подписи вида "a)", "(б)", "1.", "А)"
        как отдельные подписи к частям одного общего рисунка.
        Такая подпись сама по себе не является полноценным описанием изображения,
        поэтому её нужно временно сохранить и затем объединить с общей подписью.

        Примеры таких подписей:
        - "a)"
        - "(б)"
        - "1."
        - "А)"

        :param text: текст подписи изображения или графика
        :return: True, если текст похож на обозначение подрисунка, иначе False
        """
        text = text.strip()
        return bool(re.fullmatch(r"[(\[]?[A-Za-zА-Яа-я0-9][)\].]?", text))

    def _clean_image_caption(self, text: str) -> str:
        """
        Удаляет обозначение подрисунка из начала подписи изображения.

        Например, если после объединения получилось:
        "Рис. 1. Примеры изображений a)"
        или
        "Рис. 1. Примеры изображений 1."

        метод очищает лишнее обозначение, чтобы итоговая подпись выглядела
        аккуратнее и лучше подходила для индексации в RAG.

        :param text: исходная подпись изображения или графика
        :return: очищенная подпись без начального обозначения подрисунка
        """
        text = text.strip()
        text = re.sub(r"^[A-Za-zА-Яа-я0-9][)\].]\s*", "", text)
        return text.strip()

    def _table_block_to_text(
            self,
            caption: str | None,
            html: str | None,
            footnote: str | None = None,
    ) -> str:
        """
        Преобразует HTML-таблицу MinerU в компактное текстовое описание.

        Для RAG важнее сохранить содержимое ячеек и подпись, чем точную верстку.
        """
        parts = []

        if caption:
            parts.append(f"Подпись: {caption}")

        if html:
            text = re.sub(r"</(tr|p|div|li|h\d)>", "\n", html, flags=re.I)
            text = re.sub(r"</(td|th)>", " | ", text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s*\|\s*", " | ", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s+", "\n", text)
            text = text.strip(" |\n\t ")

            if text:
                parts.append(f"Содержимое: {text}")

        if footnote:
            parts.append(f"Примечание: {footnote}")

        return "\n".join(parts).strip()

    def do_lists_to_paragraph(self, json_file):
        """
        Возвращает новый JSON-документ MinerU, в котором блоки типа "list"
        преобразованы в paragraph-блоки.

        Исходный json_file не изменяется.

        :param json_file: JSON-документ MinerU в виде списка страниц
        :return: новый JSON-документ MinerU с преобразованными списками
        """
        new_json_file = []

        for page in json_file:
            new_page = []

            for block in page:
                new_block = deepcopy(block)

                if new_block.get("type") != "list":
                    new_page.append(new_block)
                    continue

                items = new_block.get("content", {}).get("list_items", [])
                new_contents = []

                for item in items:
                    item_content = (
                            item.get("item_content")
                            or item.get("content")
                            or []
                    )

                    if isinstance(item_content, str):
                        new_contents.append({
                            "type": "text",
                            "content": item_content
                        })

                    elif isinstance(item_content, list):
                        new_contents.extend(deepcopy(item_content))

                new_block = {
                    **new_block,
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": new_contents
                    }
                }

                new_page.append(new_block)

            new_json_file.append(new_page)

        return new_json_file

    def _do_lists_to_paragraph(self, json_file):
        """
        Алиас для обратной совместимости.

        :param json_file: JSON-документ MinerU в виде списка страниц
        :return: новый JSON-документ MinerU с преобразованными списками
        """
        return self.do_lists_to_paragraph(json_file)
