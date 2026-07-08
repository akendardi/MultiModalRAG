from copy import deepcopy
import re

from src.text_processing.pdf_extracting.document_models import DocumentType, Metadata


class ContextBuilder:
    """
    Проставляет section в Metadata на основе списка заголовков документа.
    """

    def add_context(
            self,
            documents: list[Metadata],
            headers: list[str],
            keep_abstract_before_first_section: bool = True,
            remove_after_references: bool = True,
    ) -> list[Metadata]:
        """
        Добавляет каждому Metadata название раздела, в котором находится элемент.

        :param documents: список Metadata для обработки
        :param headers: список известных заголовков документа
        :param keep_abstract_before_first_section: оставить ли аннотацию до первого раздела
        :param remove_after_references: удалить ли библиографию и всё после неё
        :return: список Metadata с заполненным section
        """
        if not headers:
            return [
                doc for doc in documents
                if doc.content_type != DocumentType.TITLE
            ]

        start_header = self._choose_start_header(headers)

        sections = []
        result = []
        started = False
        pre_section_docs = []
        abstract_added = False

        for doc in documents:
            content = (doc.content or "").strip()

            if not content:
                continue

            matched_header = self._find_matching_heading_prefix(
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
                            fallback_abstract_doc = deepcopy(fallback_abstract_doc)
                            fallback_abstract_doc.section = "Аннотация"
                            fallback_abstract_doc.content_type = DocumentType.PARAGRAPH
                            fallback_abstract_doc.content = self._clean_abstract_text(
                                fallback_abstract_doc.content or ""
                            )

                            if fallback_abstract_doc.content:
                                result.append(fallback_abstract_doc)
                                abstract_added = True

                    sections = self._update_sections(
                        sections=sections,
                        header=header,
                    )

                    doc.section = self._current_section(sections)

                    if doc.content_type == DocumentType.TITLE:
                        continue

                    if doc.content_type == DocumentType.PARAGRAPH:
                        cleaned_content = self._remove_heading_prefix(
                            text=content,
                            header=header,
                        )

                        if cleaned_content:
                            doc.content = cleaned_content
                        else:
                            continue

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
                if self._is_references_header(header):
                    if remove_after_references:
                        break

                    sections = self._update_sections(
                        sections=sections,
                        header=header,
                    )

                    doc.section = self._current_section(sections)

                    if doc.content_type == DocumentType.TITLE:
                        continue

                    result.append(doc)
                    continue

                if self._is_service_header(header):
                    continue

                sections = self._update_sections(
                    sections=sections,
                    header=header,
                )

                doc.section = self._current_section(sections)

                if doc.content_type == DocumentType.TITLE:
                    continue

                if doc.content_type == DocumentType.PARAGRAPH:
                    cleaned_content = self._remove_heading_prefix(
                        text=content,
                        header=header,
                    )

                    if cleaned_content:
                        doc.content = cleaned_content
                    else:
                        continue

                result.append(doc)
                continue

            doc.section = self._current_section(sections)

            if doc.content_type != DocumentType.TITLE:
                result.append(doc)

        return result

    def remove_context(self, documents: list[Metadata]) -> list[Metadata]:
        """
        Удаляет section из документов.

        :param documents: список Metadata
        :return: список Metadata без section
        """
        result = []

        for doc in documents:
            new_doc = deepcopy(doc)
            new_doc.section = None
            result.append(new_doc)

        return result

    def _extract_abstract_before_keywords(
            self,
            pre_section_docs: list[Metadata],
    ) -> Metadata | None:
        """
        Извлекает аннотацию из блоков до первого основного заголовка.

        :param pre_section_docs: документы до первого настоящего раздела
        :return: Metadata с section="Аннотация" или None
        """
        if not pre_section_docs:
            return None

        for index, doc in enumerate(pre_section_docs):
            content = doc.content or ""

            if not self._is_keywords_block(content):
                continue

            for prev_doc in reversed(pre_section_docs[:index]):
                prev_content = (prev_doc.content or "").strip()

                if self._is_good_fallback_abstract(prev_content):
                    abstract_doc = deepcopy(prev_doc)
                    abstract_doc.section = "Аннотация"
                    abstract_doc.content_type = DocumentType.PARAGRAPH
                    abstract_doc.content = self._clean_fallback_abstract_text(
                        prev_content
                    )

                    return abstract_doc

        return None

    def _is_keywords_block(self, text: str) -> bool:
        """
        Проверяет, является ли блок блоком ключевых слов.

        :param text: текст блока
        :return: True, если это ключевые слова
        """
        text_norm = self._normalize_for_match(text)

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

    def _is_good_fallback_abstract(self, text: str) -> bool:
        """
        Проверяет, похож ли абзац перед ключевыми словами на аннотацию.

        :param text: текст-кандидат
        :return: True, если блок можно считать аннотацией
        """
        text = str(text).strip()
        text_norm = self._normalize_for_match(text)

        if len(text_norm) < 120:
            return False

        bad_prefixes = (
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
        )

        if text_norm.startswith(bad_prefixes):
            return False

        bad_contains = (
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
        )

        if any(item in text_norm for item in bad_contains):
            return False

        russian_letters = re.findall(r"[а-яё]", text_norm)

        if len(russian_letters) < 50:
            return False

        abstract_markers = (
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
        )

        return any(marker in text_norm for marker in abstract_markers)

    def _clean_fallback_abstract_text(self, text: str) -> str:
        """
        Очищает fallback-аннотацию.

        :param text: исходный текст
        :return: очищенный текст
        """
        return re.sub(r"\s+", " ", str(text).strip()).strip()

    def _is_russian_abstract_block(self, text: str) -> bool:
        """
        Проверяет, является ли блок русской аннотацией.

        :param text: текст блока
        :return: True, если блок похож на русскую аннотацию
        """
        text_norm = self._normalize_for_match(text)

        if re.match(r"^аннотация\s*[:.]", text_norm):
            return True

        if re.match(r"^аннотация\s+", text_norm):
            return True

        return False

    def _clean_abstract_text(self, text: str) -> str:
        """
        Очищает текст аннотации от служебного префикса.

        :param text: исходный текст аннотации
        :return: очищенный текст аннотации
        """
        text = str(text).strip()
        text = re.sub(r"^аннотация\s*[:.]\s*", "", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    def _choose_start_header(self, headers: list[str]) -> str | None:
        """
        Выбирает заголовок, с которого начинается основная часть статьи.

        :param headers: список заголовков
        :return: стартовый заголовок или None
        """
        if not headers:
            return None

        for header in headers:
            if self._is_intro_header(header):
                return header

        for header in headers:
            if self._is_service_header(header):
                continue

            if self._is_references_header(header):
                continue

            return header

        return headers[0]

    def _headers_are_same(
            self,
            header_1: str | None,
            header_2: str | None,
    ) -> bool:
        """
        Проверяет, совпадают ли два заголовка после нормализации.

        :param header_1: первый заголовок
        :param header_2: второй заголовок
        :return: True, если заголовки совпадают
        """
        if not header_1 or not header_2:
            return False

        return (
            self._normalize_for_context_match(header_1)
            == self._normalize_for_context_match(header_2)
        )

    def _is_intro_header(self, header: str) -> bool:
        """
        Проверяет, является ли заголовок введением.

        :param header: заголовок
        :return: True, если это введение
        """
        header_norm = self._normalize_for_context_match(header)

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

    def _is_service_header(self, header: str) -> bool:
        """
        Проверяет, является ли заголовок служебным блоком.

        :param header: заголовок
        :return: True, если это служебный заголовок
        """
        header_norm = self._normalize_for_context_match(header)

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

    def _is_references_header(self, header: str) -> bool:
        """
        Проверяет, является ли заголовок началом библиографического раздела.

        :param header: заголовок
        :return: True, если это литература / references
        """
        header_norm = self._normalize_for_context_match(header)

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

    def _normalize_for_context_match(self, text: str) -> str:
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
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^(\d+(?:\.\d+)*)\s+", r"\1. ", text)
        text = re.sub(r"^(\d+(?:\.\d+)*)\.\.\s+", r"\1. ", text)
        text = re.sub(r"[:.;,\s]+$", "", text)
        text = re.sub(r"[^\w\s.\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_for_match(self, text: str) -> str:
        """
        Нормализует текст для простых проверок.

        :param text: исходный текст
        :return: нормализованный текст
        """
        text_norm = str(text).strip().lower()
        text_norm = text_norm.replace("ё", "е")
        return re.sub(r"\s+", " ", text_norm)

    def _find_best_header_match(
            self,
            text: str,
            headers_list: list[str],
    ) -> str | None:
        """
        Проверяет, совпадает ли текст с одним из заголовков.

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

    def _find_matching_heading_prefix(
            self,
            text: str,
            headers_list: list[str],
    ) -> str | None:
        """
        Ищет заголовок, с которого начинается текст.

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

    def _remove_heading_prefix(self, text: str, header: str) -> str:
        """
        Удаляет найденный заголовок из начала текста.

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

    def _current_section(self, sections: list[str]) -> str | None:
        """
        Возвращает текущий путь секций в виде строки.

        :param sections: список текущих заголовков
        :return: строка с текущей секцией или None
        """
        return " -> ".join(sections) if sections else None

    def _update_sections(self, sections: list[str], header: str) -> list[str]:
        """
        Обновляет стек секций на основе найденного заголовка.

        :param sections: текущий стек секций
        :param header: найденный заголовок
        :return: обновлённый стек секций
        """
        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", header)

        if not match:
            return [header]

        number = match.group(1)
        level = number.count(".") + 1

        return sections[:level - 1] + [header]
