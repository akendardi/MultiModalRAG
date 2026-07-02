from copy import deepcopy
from pathlib import Path
from enum import auto, Enum
import fitz
import re
from dataclasses import dataclass
from text_processing.mineru import run_mineru
import json


class DocumentType(Enum):
    """
    Тип содержимого, извлечённого из PDF-документа.
    Используется для разделения текста, формул, изображений,
    графиков и служебных заголовков.
    """

    PARAGRAPH = auto()
    FORMULA = auto()
    IMAGE = auto()
    CHART = auto()
    TITLE = auto()


@dataclass
class Metadata:
    """
    Структурированное представление одного элемента PDF-документа.

    Каждый объект хранит текст или описание отдельного элемента:
    абзаца, формулы, изображения, графика или заголовка.

    :param source: название или путь исходного документа
    :param page: номер страницы, на которой расположен элемент
    :param section: текущий раздел документа, к которому относится элемент
    :param content_type: тип содержимого элемента
    :param chunk_id: уникальный идентификатор элемента
    :param asset_path: путь к связанному файлу изображения/графика, если есть
    :param content: текстовое содержимое элемента
    """

    source: str
    page: int
    section: str | None
    content_type: DocumentType
    chunk_id: str
    asset_path: str | None
    content: str | None


@dataclass
class HeaderPretend:
    """
    Кандидат в заголовок раздела.

    Используется перед проверкой LLM. Хранит саму строку-кандидат,
    источник её обнаружения и соседние строки, чтобы модель могла отличить
    настоящий заголовок от названия статьи, автора, подписи рисунка
    или элемента списка.

    :param source: источник кандидата, например "mineru" или "pymupdf"
    :param content: текст кандидата в заголовок
    :param next_content: текст следующего блока или строки
    :param prev_content: текст предыдущего блока или строки
    """

    source: str
    content: str
    next_content: str | None = None
    prev_content: str | None = None


class PDFExtractor:
    """
    Извлекает структурированное содержимое из PDF-документов.

    Класс объединяет результаты MinerU и PyMuPDF, преобразует PDF
    в список объектов Metadata, собирает кандидаты в заголовки,
    а также может проставлять контекст разделов для каждого элемента.

    Основные задачи:
    - запуск или чтение результатов MinerU;
    - преобразование MinerU JSON в Metadata;
    - извлечение кандидатов в заголовки через MinerU и PyMuPDF;
    - обработка абзацев, формул, изображений и графиков;
    - добавление контекста разделов к документам.
    """

    def get_documents_from_text(
            self,
            pdf_path: str,
            output_mineru_dir_path: str = None,
            use_context: bool = True,
    ) -> list[Metadata]:
        """
        Извлекает структурированные документы из PDF-файла.

        Метод выполняет полный базовый пайплайн обработки PDF:
        1. Получает результат обработки PDF через MinerU.
           Если результат уже существует, он просто читается из JSON.
           Если результата нет, запускается MinerU.
        2. Преобразует блоки типа "list" в обычные paragraph-блоки,
           чтобы списки также попадали в текстовые документы.
        3. Преобразует MinerU JSON в список объектов Metadata:
           абзацы, формулы, изображения, графики и title-блоки.
        4. При необходимости добавляет контекст секций к каждому документу
           на основе найденных заголовков.

        :param pdf_path: путь к исходному PDF-файлу
        :param output_mineru_dir_path: путь к директории с результатами MinerU.
            Если None, используется директория по умолчанию "data/mineru_output"
        :param use_context: если True, к документам добавляется контекст разделов.
            Если False, документы возвращаются без заполнения поля section
        :return: список объектов Metadata, извлечённых из PDF
        """

        pdf_path = Path(pdf_path)

        json_file = self._get_mineru_doc(str(pdf_path), output_mineru_dir_path)
        json_file = self._do_lists_to_paragraph(json_file)

        documents = self._create_documents_from_json(json_file, pdf_path.stem)
        if use_context:
            documents = self._get_context(documents, [])
        return documents


    def _create_documents_from_json(self, json_file, source: str = "") -> list[Metadata]:
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

    def _get_context(
            self,
            documents: list[Metadata],
            headers_list: list[str],
    ) -> list[Metadata]:
        """
        Добавляет контекст разделов к извлечённым документам.

        Метод проходит по списку документов в исходном порядке и отслеживает
        текущий раздел статьи. Если содержимое документа начинается с одного
        из заголовков из headers_list, текущий стек секций обновляется.
        Все последующие документы получают значение section, соответствующее
        текущему разделу.

        Если заголовок находится в начале абзаца, он удаляется из content,
        чтобы в тексте чанка остался только основной текст без дублирования
        заголовка. Сам заголовок при этом сохраняется в поле section.

        Документы типа TITLE используются только как служебные маркеры смены
        раздела. В итоговый список документов они не добавляются, чтобы в RAG
        не попадали отдельные чанки, состоящие только из заголовков.

        :param documents: список объектов Metadata, полученных из MinerU JSON
        :param headers_list: список подтверждённых заголовков разделов.
            Обычно формируется через PyMuPDF/MinerU-кандидаты и LLM
        :return: список объектов Metadata с заполненным полем section
        """
        sections = []
        result = []

        for doc in documents:
            content = doc.content or ""

            header = self._find_matching_heading_prefix(content, headers_list)

            if header is not None:
                sections = self._update_sections(sections, header)
                content = self._remove_heading_prefix(content, header)


            if doc.content_type == DocumentType.TITLE:
                continue

            if not content.strip():
                continue

            doc.section = self._current_section(sections)
            doc.content = content.strip()

            result.append(doc)

        return result

    @staticmethod
    def _remove_heading_prefix(text: str, header: str) -> str:
        """
        Удаляет найденный заголовок из начала текста.

        Нужно, чтобы после обновления section в content остался только основной
        текст абзаца без дублирования заголовка.

        :param text: исходный текст абзаца
        :param header: заголовок, найденный в начале текста
        :return: текст без заголовка в начале
        """
        text = text.strip()
        header = header.strip()

        if text.startswith(header):
            return text[len(header):].strip()

        if not header.endswith(".") and text.startswith(header + "."):
            return text[len(header) + 1:].strip()

        if header.endswith(".") and text.startswith(header[:-1]):
            return text[len(header[:-1]):].strip()

        return text

    @staticmethod
    def _get_mineru_doc(pdf_path: str, output_mineru_dir_path: str | None = None):
        """
        Загружает результат обработки PDF из MinerU.
        Если результата ещё нет, запускает MinerU и затем читает JSON.

        :param pdf_path: путь к PDF-файлу
        :param output_mineru_dir_path: путь к директории, куда MinerU сохраняет результаты
        :return: содержимое content_list JSON в виде Python-объекта
        """
        pdf_path = Path(pdf_path)
        if output_mineru_dir_path is None:
            output_mineru_dir_path = Path("data/mineru_output")
        else:
            output_mineru_dir_path = Path(output_mineru_dir_path)
        pdf_stem = pdf_path.stem
        content_json_path = (
                output_mineru_dir_path
                / pdf_stem
                / "hybrid_auto"
                / f"{pdf_stem}_content_list_v2.json"
        )
        if not content_json_path.exists():
            run_mineru(str(pdf_path), str(output_mineru_dir_path))
        if not content_json_path.exists():
            raise FileNotFoundError(
                f"MinerU отработал, но файл не найден: {content_json_path}"
            )
        with open(content_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_headers_by_mupdf(self, file_path) -> list[HeaderPretend]:
        """
        Извлекает список заголовков разделов из PDF-файла.

        Функция открывает PDF через PyMuPDF, проходит по всем страницам,
        текстовым блокам, строкам и span-элементам. В качестве кандидатов
        в заголовки берутся фрагменты, выделенные жирным начертанием.

        Если несколько жирных фрагментов расположены рядом по вертикали
        на одной странице, они объединяются в один заголовок. Это нужно
        для случаев, когда длинный заголовок PDF разбивает на несколько
        отдельных span-элементов.

        После сбора жирных фрагментов функция оставляет только те строки,
        которые похожи на заголовки разделов: "Введение", "Заключение",
        "1. Название", "2.1 Название" и т.п.

        :param file_path: путь к PDF-файлу
        :return: список нормализованных заголовков разделов
        """
        bold_titles = []
        all_lines = []
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
                        all_lines.append(line_text)
                        # Проверяем признаки всей строки
                        is_bold_line = any(
                            "bold" in span["font"].lower() or span["flags"] & 2 ** 4
                            for span in spans
                        )

                        max_size = max(span["size"] for span in spans)
                        is_big_line = max_size >= 13.5

                        if not (is_bold_line or is_big_line):
                            continue

                        bbox = line["bbox"]
                        text = line_text

                        if bold_titles:
                            old_el = bold_titles[-1]
                            old_text, old_bbox, old_page, idx = old_el

                            if old_page == page_num and abs(bbox[1] - old_bbox[3]) <= 5:
                                new_el = [
                                    f"{old_text} {text}",
                                    [
                                        min(old_bbox[0], bbox[0]),
                                        min(old_bbox[1], bbox[1]),
                                        max(old_bbox[2], bbox[2]),
                                        max(old_bbox[3], bbox[3]),
                                    ],
                                    page_num,
                                    idx
                                ]
                                bold_titles[-1] = new_el
                                continue

                        bold_titles.append([text, bbox, page_num, len(all_lines) - 1])

        headers = []
        for bold_title in bold_titles:
            text, bbox, page, idx = bold_title
            prev_text = all_lines[max(0, idx - 1)]
            next_text = all_lines[min(len(all_lines) - 1, idx + 1)]
            headers.append(
                HeaderPretend(
                    "pupdf",
                    self._normalize_text(text),
                    self._normalize_text(next_text),
                    self._normalize_text(prev_text)
                )
            )
        return headers

    def _get_headers_by_mineru(self, json_file) -> list[HeaderPretend]:
        """
        Собирает кандидаты в заголовки из title-блоков MinerU.

        Метод проходит по текстовым блокам MinerU, сохраняет их порядок,
        а затем для каждого блока типа "title" создаёт HeaderPretend.
        Вместе с кандидатом сохраняются соседние текстовые блоки, чтобы LLM
        могла отличать заголовки разделов от названий статей, авторов и служебных строк.

        :param json_file: JSON-документ MinerU в виде списка страниц и блоков
        :return: список кандидатов HeaderPretend, найденных по title-блокам MinerU
        """

        text_blocks = []
        global_block_id = 0

        for page_num, page in enumerate(json_file, start=1):
            for block in page:
                block_type = block.get("type", "")

                if block_type == "paragraph":
                    content = block.get("content", {}).get("paragraph_content", [])
                    text = self._join_inline_content(content)

                elif block_type == "title":
                    content = block.get("content", {}).get("title_content", "")
                    text = self._join_inline_content(content)

                elif block_type == "code":
                    content = block.get("content", {}).get("code_content", [])
                    text = self._join_inline_content(content)

                else:
                    text = ""

                if text:
                    text_blocks.append({
                        "text": text,
                        "type": block_type,
                        "page": page_num,
                        "block_id": global_block_id,
                    })

                global_block_id += 1

        pretends = []

        for i, block in enumerate(text_blocks):
            if block["type"] != "title":
                continue

            prev_text = text_blocks[i - 1]["text"] if i > 0 else None
            next_text = text_blocks[i + 1]["text"] if i + 1 < len(text_blocks) else None

            pretends.append(
                HeaderPretend(
                    source="mineru",
                    content=block["text"],
                    next_content=next_text,
                    prev_content=prev_text,
                )
            )

        return pretends

    @staticmethod
    def _do_lists_to_paragraph(json_file):
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

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Нормализует текст: убирает лишние пробелы, пробелы перед знаками
        препинания, ссылки вида [1] и служебные верхние индексы.

        :param text: текст для очистки
        :return: очищенный текст
        """
        text = re.sub(r'\s+', " ", text)
        text = re.sub(r'\s+([.,;:!?])', r"\1", text)
        text = re.sub(r'\[\d+]', "", text)
        if re.fullmatch(r'^\^\{.+}$', text):
            text = ""
        return text

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

    @staticmethod
    def _is_subfigure_caption(text: str) -> bool:
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

    @staticmethod
    def _clean_image_caption(text: str) -> str:
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

    @staticmethod
    def _current_section(sections: list[str]) -> str | None:
        """
        Возвращает текущий путь секций в виде строки.

        Секции хранятся как стек заголовков. Например:
        ["2. Обзор методов", "2.1 Классические методы"]

        Метод преобразует этот список в строку:
        "2. Обзор методов -> 2.1 Классические методы"

        Если секций пока нет, возвращает None.

        :param sections: список текущих заголовков
        :return: строка с текущей секцией или None
        """
        return " -> ".join(sections) if sections else None

    @staticmethod
    def _update_sections(sections: list[str], header: str) -> list[str]:
        """
        Обновляет стек секций на основе найденного заголовка.

        Метод определяет уровень заголовка по его номеру:
        - "2. Обзор..."      -> уровень 1
        - "2.1 Что-то..."    -> уровень 2
        - "2.1.3 Что-то..."  -> уровень 3
        - "Введение"         -> уровень 1
        - "Заключение"       -> уровень 1

        Если заголовок не имеет числового номера, он считается заголовком
        первого уровня и заменяет весь текущий стек секций.

        :param sections: текущий стек секций
        :param header: найденный заголовок
        :return: обновлённый стек секций
        """
        m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", header)

        if not m:
            return [header]

        number = m.group(1)
        level = number.count(".") + 1

        return sections[:level - 1] + [header]

    @staticmethod
    def _find_matching_heading_prefix(text: str, headers_list: list[str]) -> str | None:
        """
        Ищет заголовок, с которого начинается переданный текст.

        Заголовки заранее извлекаются из PDF через get_headers().
        Затем при обработке MinerU JSON метод проверяет, начинается ли
        текущий абзац с одного из этих заголовков.

        Например:
        text = "2. Обзор существующих методов. Стоит отметить..."
        header = "2. Обзор существующих методов."

        В таком случае метод вернёт найденный заголовок.

        :param text: текст текущего абзаца
        :param headers_list: список заголовков, найденных в PDF
        :return: найденный заголовок или None
        """
        for header in headers_list:
            if text.startswith(header):
                return header

        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Нормализует текст, извлечённый из PDF.

        Функция приводит последовательности пробелов, переносов строк и табуляций
        к одному пробелу, заменяет неразрывные пробелы на обычные, а также
        склеивает слова, которые были разорваны переносом через дефис.

        Например:
        "свёрточ-\nная нейронная сеть" -> "свёрточная нейронная сеть"

        :param text: исходный текст
        :return: нормализованный текст
        """
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\u00a0", " ")
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        return text

    @staticmethod
    def _looks_like_section_heading(text: str) -> bool:
        """
        Проверяет, похожа ли строка на заголовок раздела.
        """
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            return False

        text_lower = text.lower()

        # Отсекаем десятичные числа: 32.66, 0.959
        if re.fullmatch(r"\d+(?:[.,]\d+)+", text):
            return False

        bad_prefixes = (
            "ключевые слова",
            "keywords",
            "удк",
            "udc",
            "doi",
            "цитирование",
            "citation",
            "рецензенты",
            "рецензент",
            "поступила в редакцию",
            "рис",
            "табл"
        )

        if text_lower.startswith(bad_prefixes):
            return False

        # Нумерованные заголовки:
        # 1 Введение
        # 1. Введение
        # 3.1 Предлагаемый КЛ-ДР
        pattern = (
            r"^(?:"
            r"литература|список литературы|использованная литература|"
            r"\d*\.?\s*введение|\d*\.?\s*заключение|\d*\.?\s*выводы|"
            r"\d*(?:\.\d+)*\.?\s*[A-Za-zА-Яа-яЁё0-9\s\-–—(),]{2,160}\.?"
            r")$"
        )

        if re.match(pattern, text, flags=re.I):
            return True

        return False


