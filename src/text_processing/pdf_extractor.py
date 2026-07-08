from copy import deepcopy
from pathlib import Path
from enum import auto, Enum
import re
from dataclasses import dataclass
from src.text_processing.mineru import run_mineru
import json
from src.paths import MINERU_OUTPUT_DIR


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

    @staticmethod
    def _remove_references_section(documents: list[Metadata]) -> list[Metadata]:
        """
        Удаляет список литературы, список источников и references из документов.

        Удаление начинается с документа, который похож на заголовок библиографии,
        и продолжается до конца списка документов.

        :param documents: список Metadata
        :return: список Metadata без библиографического раздела
        """
        result = []

        for doc in documents:
            content = doc.content or ""
            content_norm = content.strip().lower()
            content_norm = content_norm.replace("ё", "е")
            content_norm = re.sub(r"\s+", " ", content_norm)

            if PDFExtractor._is_references_heading(content_norm):
                break

            result.append(doc)

        return result

    @staticmethod
    def _is_references_heading(text: str) -> bool:
        """
        Проверяет, является ли строка заголовком списка литературы.

        :param text: нормализованный текст
        :return: True, если это заголовок библиографии
        """
        patterns = [
            r"^литература$",
            r"^список литературы$",
            r"^список использованной литературы$",
            r"^использованная литература$",
            r"^список источников$",
            r"^список использованных источников$",
            r"^источники$",
            r"^references$",
            r"^bibliography$",
        ]

        return any(re.fullmatch(pattern, text, flags=re.I) for pattern in patterns)

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

    def get_documents_from_text(
            self,
            pdf_path: str | Path,
            use_context: bool = False,
            type_context: str = "gold",
            keep_abstract_before_first_section: bool = True,
            remove_after_references: bool = True,
    ) -> list[Metadata]:
        """
        Извлекает структурированные документы из PDF-файла.

        Метод выполняет полный базовый пайплайн обработки PDF:
        1. Получает результат обработки PDF через MinerU.
        2. Преобразует блоки типа "list" в paragraph-блоки.
        3. Преобразует MinerU JSON в список объектов Metadata.
        4. При необходимости добавляет контекст секций.
        5. До первого настоящего заголовка оставляет только аннотацию.
        6. После списка литературы может обрезать документ.

        :param pdf_path: путь к исходному PDF-файлу
        :param use_context: использовать ли контекст разделов
        :param type_context: источник заголовков: gold, mineru, mupdf
        :param keep_abstract_before_first_section: оставить ли аннотацию до первого раздела
        :param remove_after_references: удалить ли список литературы и всё после него
        :return: список Metadata
        """
        pdf_path = Path(pdf_path)

        json_file = self._get_mineru_doc(str(pdf_path))
        json_file = self._do_lists_to_paragraph(json_file)

        documents = self._create_documents_from_json(json_file, pdf_path.stem)

        if use_context:
            documents = self._get_context(
                documents=documents,
                file_path=str(pdf_path),
                type_context=type_context,
                keep_abstract_before_first_section=keep_abstract_before_first_section,
                remove_after_references=remove_after_references,
            )

        return documents

    def _get_context(
            self,
            documents: list[Metadata],
            file_path: str | Path,
            type_context: str = "gold",
            keep_abstract_before_first_section: bool = True,
            remove_after_references: bool = True,
    ) -> list[Metadata]:
        """
        Добавляет каждому Metadata название раздела, в котором находится элемент.

        До первого настоящего заголовка сохраняется только аннотация.

        Если явного блока "Аннотация" нет, но найден блок "Ключевые слова",
        то в качестве аннотации берётся предыдущий содержательный абзац.

        :param documents: список Metadata для обработки
        :param file_path: путь к PDF-файлу
        :param type_context: источник заголовков: gold, mineru или mupdf
        :param keep_abstract_before_first_section: оставить ли аннотацию до первого раздела
        :param remove_after_references: удалить ли библиографию и всё после неё
        :return: список Metadata с заполненным section
        """
        file_path = Path(file_path)

        headers = self._get_headers_for_context(
            file_path=file_path,
            type_context=type_context,
        )

        if not headers:
            return documents

        start_header = self._choose_start_header(headers)

        sections = []
        result = []

        started = False

        pre_section_docs = []
        abstract_added = False

        for doc in documents:
            content = doc.content or ""
            content = content.strip()

            if not content:
                continue

            matched_header = self._find_matching_heading_prefix_robust(
                text=content,
                headers_list=headers,
            )

            title_header = None

            if doc.content_type == DocumentType.TITLE:
                title_header = self._find_best_header_match(
                    text=content,
                    headers_list=headers,
                )

            header = title_header or matched_header

            if not started:
                if header and self._headers_are_same(header, start_header):
                    started = True

                    if (
                            keep_abstract_before_first_section
                            and not abstract_added
                    ):
                        fallback_abstract_doc = self._extract_abstract_before_keywords(
                            pre_section_docs
                        )

                        if fallback_abstract_doc:
                            result.append(fallback_abstract_doc)
                            abstract_added = True

                    sections = self._update_sections(
                        sections=sections,
                        header=header,
                    )

                    doc.section = self._current_section(sections)

                    if doc.content_type == DocumentType.PARAGRAPH:
                        cleaned_content = self._remove_heading_prefix(
                            text=content,
                            header=header,
                        )

                        if cleaned_content:
                            doc.content = cleaned_content

                    result.append(doc)
                    continue

                pre_section_docs.append(doc)

                if (
                        keep_abstract_before_first_section
                        and not abstract_added
                        and self._is_russian_abstract_block(content)
                ):
                    abstract_doc = deepcopy(doc)
                    abstract_doc.section = "Аннотация"
                    abstract_doc.content_type = DocumentType.PARAGRAPH
                    abstract_doc.content = self._clean_abstract_text(content)

                    if abstract_doc.content:
                        result.append(abstract_doc)
                        abstract_added = True

                continue

            if header:
                if self._is_context_references_header(header):
                    if remove_after_references:
                        break

                    sections = self._update_sections(
                        sections=sections,
                        header=header,
                    )

                    doc.section = self._current_section(sections)
                    result.append(doc)
                    continue

                if self._is_service_context_header(header):
                    continue

                sections = self._update_sections(
                    sections=sections,
                    header=header,
                )

                doc.section = self._current_section(sections)

                if doc.content_type == DocumentType.PARAGRAPH:
                    cleaned_content = self._remove_heading_prefix(
                        text=content,
                        header=header,
                    )

                    if cleaned_content:
                        doc.content = cleaned_content

                result.append(doc)
                continue

            doc.section = self._current_section(sections)
            result.append(doc)

        return result

    @staticmethod
    def _extract_abstract_before_keywords(
            pre_section_docs: list[Metadata],
    ) -> Metadata | None:
        """
        Извлекает аннотацию из блоков до первого основного заголовка.

        Если явного блока "Аннотация" нет, но есть блок "Ключевые слова",
        то аннотацией считается ближайший предыдущий содержательный абзац.

        Пример:
        [
            "Название статьи",
            "Авторы",
            "В статье рассматривается ...",
            "Ключевые слова: нейронные сети, классификация"
        ]

        В этом случае будет взят:
        "В статье рассматривается ..."

        :param pre_section_docs: документы до первого настоящего раздела
        :return: Metadata с section="Аннотация" или None
        """
        if not pre_section_docs:
            return None

        for index, doc in enumerate(pre_section_docs):
            content = doc.content or ""

            if not PDFExtractor._is_keywords_block(content):
                continue

            for prev_doc in reversed(pre_section_docs[:index]):
                prev_content = prev_doc.content or ""
                prev_content = prev_content.strip()

                if PDFExtractor._is_good_fallback_abstract(prev_content):
                    abstract_doc = deepcopy(prev_doc)
                    abstract_doc.section = "Аннотация"
                    abstract_doc.content_type = DocumentType.PARAGRAPH
                    abstract_doc.content = PDFExtractor._clean_fallback_abstract_text(
                        prev_content
                    )

                    return abstract_doc

        return None

    @staticmethod
    def _is_keywords_block(text: str) -> bool:
        """
        Проверяет, является ли блок блоком ключевых слов.

        :param text: текст блока
        :return: True, если это ключевые слова
        """
        text_norm = str(text).strip().lower()
        text_norm = text_norm.replace("ё", "е")
        text_norm = re.sub(r"\s+", " ", text_norm)

        patterns = [
            r"^ключевые слова\s*[:.].+",
            r"^ключевые слова\s+.+",
            r"^keywords\s*[:.].+",
            r"^key words\s*[:.].+",
        ]

        return any(
            re.match(pattern, text_norm, flags=re.I)
            for pattern in patterns
        )

    @staticmethod
    def _is_good_fallback_abstract(text: str) -> bool:
        """
        Проверяет, похож ли абзац перед ключевыми словами на аннотацию.

        Отсекает:
        - УДК, DOI;
        - авторов;
        - название статьи;
        - слишком короткие блоки;
        - английские блоки;
        - служебные строки.

        :param text: текст-кандидат
        :return: True, если блок можно считать аннотацией
        """
        text = str(text).strip()
        text_norm = text.lower()
        text_norm = text_norm.replace("ё", "е")
        text_norm = re.sub(r"\s+", " ", text_norm)

        if len(text_norm) < 120:
            return False

        bad_prefixes = [
            "удк",
            "doi",
            "для цитирования",
            "for citation",
            "content",
            "license",
            "©",
            "issn",
            "edn",
            "грнти",
        ]

        if any(text_norm.startswith(prefix) for prefix in bad_prefixes):
            return False

        bad_contains = [
            "e-mail",
            "email",
            "orcid",
            "университет",
            "институт",
            "академия",
            "кафедра",
            "россия",
            "russia",
            "moscow",
            "abstract",
            "keywords",
            "key words",
        ]

        if any(item in text_norm for item in bad_contains):
            return False

        russian_letters = re.findall(r"[а-яё]", text_norm)
        if len(russian_letters) < 50:
            return False

        abstract_markers = [
            "рассмотр",
            "предлож",
            "представлен",
            "исследован",
            "проведен",
            "показан",
            "описан",
            "разработан",
            "цель",
            "задач",
            "метод",
            "результат",
            "анализ",
        ]

        if not any(marker in text_norm for marker in abstract_markers):
            return False

        return True

    @staticmethod
    def _clean_fallback_abstract_text(text: str) -> str:
        """
        Очищает fallback-аннотацию.

        :param text: исходный текст
        :return: очищенный текст
        """
        text = str(text).strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _is_russian_abstract_block(text: str) -> bool:
        """
        Проверяет, является ли блок русской аннотацией.

        Метод специально не считает английский Abstract аннотацией,
        чтобы до первого раздела сохранялась только русская аннотация.

        :param text: текст блока
        :return: True, если блок похож на русскую аннотацию
        """
        text = str(text).strip()
        text_norm = text.lower()
        text_norm = text_norm.replace("ё", "е")
        text_norm = re.sub(r"\s+", " ", text_norm)

        if re.match(r"^аннотация\s*[:.]", text_norm):
            return True

        if re.match(r"^аннотация\s+", text_norm):
            return True

        return False

    @staticmethod
    def _clean_abstract_text(text: str) -> str:
        """
        Очищает текст аннотации от служебного префикса.

        :param text: исходный текст аннотации
        :return: очищенный текст аннотации
        """
        text = str(text).strip()

        text = re.sub(
            r"^аннотация\s*[:.]\s*",
            "",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _choose_start_header(headers: list[str]) -> str | None:
        """
        Выбирает заголовок, с которого начинается основная часть статьи.

        Если есть "Введение", выбирается оно.
        Если "Введения" нет, выбирается первый содержательный заголовок.

        :param headers: список заголовков
        :return: стартовый заголовок или None
        """
        if not headers:
            return None

        for header in headers:
            if PDFExtractor._is_intro_header(header):
                return header

        for header in headers:
            if PDFExtractor._is_service_context_header(header):
                continue

            if PDFExtractor._is_context_references_header(header):
                continue

            return header

        return headers[0]

    @staticmethod
    def _headers_are_same(header_1: str | None, header_2: str | None) -> bool:
        """
        Проверяет, совпадают ли два заголовка после нормализации.

        :param header_1: первый заголовок
        :param header_2: второй заголовок
        :return: True, если заголовки совпадают
        """
        if not header_1 or not header_2:
            return False

        return (
                PDFExtractor._normalize_for_context_match(header_1)
                == PDFExtractor._normalize_for_context_match(header_2)
        )

    @staticmethod
    def _is_intro_header(header: str) -> bool:
        """
        Проверяет, является ли заголовок введением.

        :param header: заголовок
        :return: True, если это введение
        """
        header_norm = PDFExtractor._normalize_for_context_match(header)

        patterns = [
            r"^введение$",
            r"^1 введение$",
            r"^1\. введение$",
            r"^1 введение и постановка задачи$",
            r"^1\. введение и постановка задачи$",
        ]

        return any(
            re.fullmatch(pattern, header_norm, flags=re.I)
            for pattern in patterns
        )

    @staticmethod
    def _is_service_context_header(header: str) -> bool:
        """
        Проверяет, является ли заголовок служебным блоком.

        :param header: заголовок
        :return: True, если это служебный заголовок
        """
        header_norm = PDFExtractor._normalize_for_context_match(header)

        patterns = [
            r"^аннотация$",
            r"^abstract$",
            r"^ключевые слова$",
            r"^keywords$",
            r"^key words$",
            r"^для цитирования$",
            r"^for citation$",
            r"^цель исследования$",
            r"^материалы и методы исследования$",
            r"^результаты$",
        ]

        return any(
            re.fullmatch(pattern, header_norm, flags=re.I)
            for pattern in patterns
        )

    @staticmethod
    def _is_context_references_header(header: str) -> bool:
        """
        Проверяет, является ли заголовок началом библиографического раздела.

        :param header: заголовок
        :return: True, если это литература / references
        """
        header_norm = PDFExtractor._normalize_for_context_match(header)

        patterns = [
            r"^литература$",
            r"^список литературы$",
            r"^список источников$",
            r"^список использованной литературы$",
            r"^список использованных источников$",
            r"^источники$",
            r"^references$",
            r"^bibliography$",
            r"^список литературы references$",
            r"^список литературы / references$",
        ]

        return any(
            re.fullmatch(pattern, header_norm, flags=re.I)
            for pattern in patterns
        )

    def _get_headers_for_context(
            self,
            file_path: Path,
            type_context: str = "gold",
    ) -> list[str]:
        """
        Получает список заголовков для построения контекста.

        :param file_path: путь к PDF-файлу
        :param type_context: источник заголовков: gold, mineru или mupdf
        :return: список заголовков
        """
        type_context = type_context.lower().strip()

        if type_context == "gold":
            return self._get_gold_headers(file_path)

        if type_context == "mineru":
            pretends = self.get_headers_by_mineru(str(file_path))
            return [
                item.content
                for item in pretends
                if item.content
            ]

        if type_context == "mupdf":
            pretends = self.get_headers_by_mupdf(str(file_path))
            return [
                item.content
                for item in pretends
                if item.content
            ]

        raise ValueError(
            f"Неизвестный type_context={type_context}. "
            f"Доступные значения: gold, mineru, mupdf"
        )

    @staticmethod
    def _get_gold_headers(file_path: str | Path) -> list[str]:
        """
        Загружает ручные заголовки документа из data/gold_headers.json.

        Метод ищет запись по имени PDF-файла.

        :param file_path: путь к PDF-файлу
        :return: список эталонных заголовков
        """
        file_path = Path(file_path)
        document_name = file_path.name

        gold_path = Path(MINERU_OUTPUT_DIR).parent / "gold_headers.json"

        if not gold_path.exists():
            raise FileNotFoundError(
                f"Файл ручной разметки не найден: {gold_path}"
            )

        with open(gold_path, "r", encoding="utf-8") as file:
            gold_data = json.load(file)

        for item in gold_data:
            if item.get("document") == document_name:
                return item.get("headers", [])

        return []

    @staticmethod
    def _normalize_for_context_match(text: str) -> str:
        """
        Нормализует текст для сравнения заголовков.

        :param text: исходный текст
        :return: нормализованный текст
        """
        text = str(text).lower()
        text = text.replace("ё", "е")
        text = text.replace("\u00ad", "")
        text = text.replace("\ufeff", "")

        text = text.replace("–", "-")
        text = text.replace("—", "-")

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        text = re.sub(r"^(\d+(?:\.\d+)*)\s+", r"\1. ", text)
        text = re.sub(r"^(\d+(?:\.\d+)*)\.\.\s+", r"\1. ", text)

        text = re.sub(r"[:.;,\s]+$", "", text)
        text = re.sub(r"[^\w\s.\-]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _find_best_header_match(
            self,
            text: str,
            headers_list: list[str],
    ) -> str | None:
        """
        Проверяет, совпадает ли текст с одним из заголовков.

        Метод используется в первую очередь для title-блоков MinerU.

        :param text: текст блока
        :param headers_list: список известных заголовков
        :return: найденный заголовок или None
        """
        text_norm = self._normalize_for_context_match(text)

        for header in headers_list:
            header_norm = self._normalize_for_context_match(header)

            if text_norm == header_norm:
                return header

        return None

    def _find_matching_heading_prefix_robust(
            self,
            text: str,
            headers_list: list[str],
    ) -> str | None:
        """
        Ищет заголовок, с которого начинается текст.

        Метод нужен для случаев:
        - "1. Введение. Далее текст..."
        - "Введение. Далее текст..."
        - "2. Метод исследования Далее текст..."
        - title-блок отдельно совпадает с заголовком.

        :param text: текст текущего Metadata
        :param headers_list: список известных заголовков
        :return: найденный заголовок или None
        """
        text = str(text).strip()
        text_norm = self._normalize_for_context_match(text)

        headers_sorted = sorted(
            headers_list,
            key=lambda h: len(self._normalize_for_context_match(h)),
            reverse=True,
        )

        for header in headers_sorted:
            header_norm = self._normalize_for_context_match(header)

            if not header_norm:
                continue

            if text_norm == header_norm:
                return header

            if text_norm.startswith(header_norm + " "):
                return header

            if text_norm.startswith(header_norm + ". "):
                return header

            if text_norm.startswith(header_norm + ": "):
                return header

        return None

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
        text = str(text).strip()
        header = str(header).strip()

        if not text or not header:
            return text

        escaped_header = re.escape(header)

        patterns = [
            rf"^{escaped_header}\s*[\.:]\s*",
            rf"^{escaped_header}\s+",
        ]

        if header.endswith("."):
            header_without_dot = re.escape(header.rstrip("."))
            patterns.extend([
                rf"^{header_without_dot}\s*[\.:]\s*",
                rf"^{header_without_dot}\s+",
            ])

        for pattern in patterns:
            new_text = re.sub(
                pattern,
                "",
                text,
                count=1,
                flags=re.I,
            ).strip()

            if new_text != text:
                return new_text

        return text

    @staticmethod
    def _get_mineru_doc(pdf_path: str):
        """
        Загружает результат обработки PDF из MinerU.
        Если результата ещё нет, запускает MinerU и затем читает JSON.

        :param pdf_path: путь к PDF-файлу
        :param output_mineru_dir_path: путь к директории, куда MinerU сохраняет результаты
        :return: содержимое content_list JSON в виде Python-объекта
        """
        pdf_path = Path(pdf_path)
        output_mineru_dir_path = Path(MINERU_OUTPUT_DIR)
        pdf_stem = pdf_path.stem
        content_json_path = (
                output_mineru_dir_path
                / pdf_stem
                / "hybrid_auto"
                / f"{pdf_stem}_content_list_v2.json"
        )
        if not content_json_path.exists():
            run_mineru(str(pdf_path))
        if not content_json_path.exists():
            raise FileNotFoundError(
                f"MinerU отработал, но файл не найден: {content_json_path}"
            )
        with open(content_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_headers_by_mupdf(self, file_path) -> list[HeaderPretend]:
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
        import fitz

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
                            "bold" in span["font"].lower() or span["flags"] & 2 ** 4 or span["flags"] & 2
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

    def get_headers_by_mineru(self, file_path: str) -> list[HeaderPretend]:
        """
        Собирает кандидаты в заголовки из title-блоков MinerU.

        Метод проходит по текстовым блокам MinerU, сохраняет их порядок,
        а затем для каждого блока типа "title" создаёт HeaderPretend.
        Вместе с кандидатом сохраняются соседние текстовые блоки, чтобы LLM
        могла отличать заголовки разделов от названий статей, авторов и служебных строк.

        :param file_path: путь к пдф документу
        :return: список кандидатов HeaderPretend, найденных по title-блокам MinerU
        """

        text_blocks = []
        global_block_id = 0

        json_file = self._get_mineru_doc(file_path)

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
