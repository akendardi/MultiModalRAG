import re


class HeadersProcessor:
    """Очищает и фильтрует кандидаты в заголовки из PDF."""

    MOJIBAKE_CHARS = (
        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß"
        "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
    )

    PDF_CYRILLIC_FONT_MAP = {
        "Ⱥ": "А", "Ȼ": "Б", "ȼ": "В", "Ƚ": "Г", "Ⱦ": "Д", "ȿ": "Е",
        "ɀ": "Ж", "Ɂ": "З", "ɂ": "И", "Ƀ": "Й", "Ʉ": "К", "Ʌ": "Л",
        "Ɇ": "М", "ɇ": "Н", "Ɉ": "О", "ɉ": "П", "Ɋ": "Р", "ɋ": "С",
        "Ɍ": "Т", "ɍ": "У", "Ɏ": "Ф", "ɏ": "Х", "ɐ": "Ц", "ɑ": "ч",
        "ɒ": "Ш", "ɓ": "щ", "ɔ": "ъ", "ɕ": "ы", "ɖ": "ь", "ɗ": "э",
        "ɘ": "Ю", "ə": "я", "ɚ": "а", "ɛ": "б", "ɜ": "в", "ɝ": "г",
        "ɞ": "д", "ɟ": "е", "ɠ": "ж", "ɡ": "з", "ɢ": "и", "ɣ": "й",
        "ɤ": "к", "ɥ": "л", "ɦ": "м", "ɧ": "н", "ɨ": "о", "ɩ": "п",
        "ɪ": "р", "ɫ": "с", "ɬ": "т", "ɭ": "у", "ɮ": "ф", "ɯ": "х",
        "ɰ": "ц", "ɱ": "ч", "ɲ": "ш", "ɳ": "щ", "ɴ": "ъ", "ɵ": "ы",
        "ɶ": "ь", "ɷ": "э", "ɸ": "ю", "ɹ": "я",
    }

    LITERATURE_RE = re.compile(
        r"(список\s+литературы|литература|библиографический\s+список|"
        r"библиографические\s+ссылки|ссылки\s+на\s+источники|"
        r"список\s+источников|references|sources)\s*:?",
        re.I,
    )

    SERVICE_HEADER_RE = re.compile(
        r"(введение|заключение|основная часть|список литературы|"
        r"литература|библиографический список|библиографические ссылки|"
        r"благодарности|references|sources)",
        re.I,
    )

    JOURNAL_NOISE = (
        "педагогика и психология", "наука. инновации. технологии",
        "scientific journal", "компьютерная оптика", "технические науки",
        "технічні науки", "удк", "doi", "issn", "том", "vol.", "no.",
    )

    PERSON_INFO = (
        "доцент", "профессор", "магистрант", "аспирант", "студент",
        "кандидат", "д-р", "доктор", "кафедра", "associate professor",
        "professor", "student", "department", "сведения об авторе",
        "сведения об авторах",
    )

    AFFILIATION_WORDS = (
        "университет", "институт", "академия", "фгаоу", "фгбоу", "впо",
        "кафедра", "г.", "e-mail", "email", "university", "institute",
        "department", "academy", "russia", "россия", "северо-кавказский",
        "north-caucasus",
    )

    ABSTRACT_NOISE = (
        "аннотация", "ключевые слова", "abstract", "keywords", "key words",
        "the educational robotics", "engineering education",
        "robotics, educational robotics",
    )

    SENTENCE_MARKERS = (
        "является", "являются", "внедрила", "внедрил", "внедрившей",
        "используется", "позволяет", "представляет", "рассматривается",
        "состоит", "имеет", "может быть", "следует", "необходимо",
        "таким образом",
    )

    def clear_headers(self, headers: list[str]) -> list[str]:
        """
        Очищает список кандидатов и оставляет только заголовки разделов.

        :param headers: список строк-кандидатов
        :return: очищенный список заголовков
        """
        clean_headers = []
        is_start = False

        for raw_text in headers:
            text = self._prepare_header_text(raw_text)

            if not text or len(text) < 5:
                continue

            low = text.lower()

            if self._is_literature_header(text):
                clean_headers.append(text)
                break

            if self._is_noise(text):
                continue

            if text.replace(" ", "").isalpha() and text.islower():
                continue

            upper_letters = [ch for ch in text if ch.isalpha() and ch.isupper()]
            all_letters = [ch for ch in text if ch.isalpha()]
            upper_coef = len(upper_letters) / max(len(all_letters), 1)

            if text.isupper() or upper_coef > 0.85:
                is_start = True
                continue

            if "ключевые слова" in low or "keywords" in low:
                is_start = True
                continue

            if not is_start:
                if (
                    low.startswith("введение")
                    or self._is_numbered_header(text)
                    or self._is_service_header(text)
                ):
                    is_start = True
                else:
                    continue

            if self._has_sentence_noise(text):
                continue

            header = self._extract_header(text)

            if header:
                clean_headers.append(header)

        clean_headers = self._merge_broken_headers(clean_headers)
        return self._deduplicate_headers(clean_headers)

    def _prepare_header_text(self, text: str) -> str:
        """
        Выполняет базовую очистку и нормализацию строки-кандидата.

        :param text: исходная строка
        :return: нормализованная строка
        """
        text = self._clean_text(text)
        text = self._remove_page_number_tail(text)
        text = self._cut_after_literature_word(text)
        return self._normalize_numbered_header(text)

    def _cyrillic_count(self, text: str) -> int:
        """
        Считает количество кириллических символов в строке.

        :param text: исходный текст
        :return: количество русских букв
        """
        return sum(
            "а" <= ch.lower() <= "я" or ch.lower() == "ё"
            for ch in text
        )

    def _looks_like_mojibake(self, text: str) -> bool:
        """
        Проверяет, похож ли текст на неправильно декодированную кириллицу.

        :param text: исходный текст
        :return: True, если текст похож на битую кодировку
        """
        return sum(ch in self.MOJIBAKE_CHARS for ch in text) >= 3

    def _fix_mojibake(self, text: str) -> str:
        """
        Исправляет кириллицу, ошибочно прочитанную как latin1.

        :param text: исходный текст
        :return: исправленный текст или исходная строка
        """
        if not self._looks_like_mojibake(text):
            return text

        try:
            fixed = text.encode("latin1", errors="ignore").decode(
                "cp1251",
                errors="ignore",
            )
        except Exception:
            return text

        if self._cyrillic_count(fixed) > self._cyrillic_count(text):
            return fixed

        return text

    def _fix_pdf_cyrillic_font_map(self, text: str) -> str:
        """
        Заменяет нестандартные PDF-символы на кириллические буквы.

        :param text: исходный текст
        :return: текст с восстановленными символами
        """
        return "".join(self.PDF_CYRILLIC_FONT_MAP.get(ch, ch) for ch in text)

    def _clean_text(self, text: str) -> str:
        """
        Нормализует строку заголовка и убирает служебные символы.

        :param text: текст для очистки
        :return: очищенный текст
        """
        text = self._fix_mojibake(str(text))
        text = self._fix_pdf_cyrillic_font_map(text)
        text = re.sub(r"[\uf000-\uf8ff]", " ", text)

        for char in ("\u00ad", "\x0e", "\x19", "\x1a", "\ufeff"):
            text = text.replace(char, "")

        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text).strip()

        if re.match(r"^(©|\(c\)|copyright)\s*", text, flags=re.I):
            return ""

        return text

    def _is_literature_header(self, text: str) -> bool:
        """
        Проверяет, является ли строка заголовком списка литературы.

        :param text: проверяемый текст
        :return: True, если строка обозначает литературу или references
        """
        text = self._clean_text(text).strip().lower()
        return bool(self.LITERATURE_RE.fullmatch(text))

    def _is_reference_item(self, text: str) -> bool:
        """
        Проверяет, похожа ли строка на пункт библиографического списка.

        :param text: проверяемый текст
        :return: True, если строка похожа на ссылку из литературы
        """
        text = text.strip()

        if re.match(
            r"^\d+\.\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-]+[, ]+\s*[A-ZА-ЯЁ]\.?",
            text,
        ):
            return True

        return bool(
            re.match(r"^\d+\.\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-]+", text)
            and "," in text
        )

    def _is_journal_noise(self, text: str) -> bool:
        """
        Проверяет, является ли строка журнальным или служебным шумом.

        :param text: проверяемый текст
        :return: True, если строку нужно исключить
        """
        text = text.strip()
        low = text.lower()

        if re.fullmatch(r"№\s*\d+\s*,?\s*\d{4}", text):
            return True

        if re.fullmatch(r"\d+\s*№\s*\d+|№\s*\d+|\d{1,4}", text):
            return True

        return any(word in low for word in self.JOURNAL_NOISE)

    def _is_author_or_person_info(self, text: str) -> bool:
        """
        Проверяет, похожа ли строка на ФИО или сведения об авторе.

        :param text: проверяемый текст
        :return: True, если строка содержит автора или должность
        """
        text = text.strip()
        low = text.lower()

        if re.fullmatch(
            r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+,?.*",
            text,
        ):
            return True

        if re.fullmatch(r"[А-ЯЁ][а-яё]+ [А-ЯЁ]\.\s*[А-ЯЁ]\..*", text):
            return True

        if re.fullmatch(r"[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+.*", text):
            return True

        if any(word in low for word in self.PERSON_INFO):
            return True

        english_author_markers = (
            "associate professor", "department", "university", "institute",
            "student", "professor", "candidate",
        )

        return bool(
            re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+,?.*", text)
            and any(marker in low for marker in english_author_markers)
        )

    def _is_affiliation(self, text: str) -> bool:
        """
        Проверяет, похожа ли строка на организацию или аффилиацию.

        :param text: проверяемый текст
        :return: True, если строка содержит место работы или контакты
        """
        low = text.lower()

        if any(word in low for word in self.AFFILIATION_WORDS):
            return True

        return bool(re.match(r"^\d+\s+[A-ZА-ЯЁ]", text) and "," in text)

    def _is_abstract_noise(self, text: str) -> bool:
        """
        Проверяет, относится ли строка к аннотации или ключевым словам.

        :param text: проверяемый текст
        :return: True, если строка является abstract/keywords-шумом
        """
        low = text.lower().strip()

        if any(phrase in low for phrase in self.ABSTRACT_NOISE):
            return True

        latin_count = sum("a" <= ch.lower() <= "z" for ch in text)
        return latin_count > 25 and self._cyrillic_count(text) == 0

    def _is_math_or_table_noise(self, text: str) -> bool:
        """
        Проверяет, похожа ли строка на формулу, число или элемент таблицы.

        :param text: проверяемый текст
        :return: True, если строка является табличным или математическим шумом
        """
        if re.fullmatch(r"[\d\s.,%]+", text):
            return True

        if sum(ch in "∑∏∫×÷√≈≠≤≥∞{}[]" for ch in text) >= 2:
            return True

        return bool(
            re.fullmatch(r"[A-Za-zА-Яа-яЁё]\s+[A-Za-zА-Яа-яЁё].*", text)
            and any(ch in text for ch in ":=∑∏×")
        )

    def _is_figure_or_table_caption(self, text: str) -> bool:
        """
        Проверяет, является ли строка подписью рисунка или таблицы.

        :param text: проверяемый текст
        :return: True, если строка похожа на подпись
        """
        low = text.lower().strip()
        return any(
            re.search(pattern, low)
            for pattern in (
                r"\bрис\.?\s*\d+",
                r"\bрисунок\s*\d+",
                r"\bтабл\.?\s*\d+",
                r"\bтаблица\s*\d+",
            )
        )

    def _has_sentence_noise(self, text: str) -> bool:
        """
        Проверяет, похожа ли строка на обычное предложение.

        :param text: проверяемый текст
        :return: True, если строка содержит признаки абзаца
        """
        low = text.lower()
        return any(marker in low for marker in self.SENTENCE_MARKERS)

    def _remove_page_number_tail(self, text: str) -> str:
        """
        Удаляет номер страницы из конца строки.

        :param text: исходный текст
        :return: текст без номера страницы в конце
        """
        return re.sub(
            r"(\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^.\d]{3,160})\s+\d+$",
            r"\1",
            text,
        ).strip()

    def _cut_after_literature_word(self, text: str) -> str:
        """
        Обрезает строку после упоминания блока литературы.

        :param text: исходный текст
        :return: текст до начала библиографического блока
        """
        return re.sub(
            r"\s+(литература|список\s+литературы|references|sources)\s*:?.*$",
            "",
            text,
            flags=re.I,
        ).strip()

    def _normalize_numbered_header(self, text: str) -> str:
        """
        Приводит нумерацию заголовка к единому виду.

        :param text: исходный текст
        :return: текст с нормализованной нумерацией
        """
        text = re.sub(r"^(\d+(?:\.\d+)*)\s+", r"\1. ", text)
        return re.sub(r"^(\d+(?:\.\d+)*)\.\.\s+", r"\1. ", text).strip()

    def _is_service_header(self, text: str) -> bool:
        """
        Проверяет, является ли строка типовым служебным заголовком.

        :param text: проверяемый текст
        :return: True, если строка является стандартным заголовком
        """
        return bool(self.SERVICE_HEADER_RE.fullmatch(text.strip().lower()))

    def _is_numbered_header(self, text: str) -> bool:
        """
        Проверяет, является ли строка нумерованным заголовком.

        :param text: проверяемый текст
        :return: True, если строка начинается с номера раздела
        """
        return bool(re.match(r"^\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]", text.strip()))

    def _should_merge_with_next(self, current: str, next_text: str) -> bool:
        """
        Проверяет, нужно ли склеить две соседние строки заголовка.

        :param current: текущая строка
        :param next_text: следующая строка
        :return: True, если строки выглядят как части одного заголовка
        """
        current = current.strip()
        next_text = next_text.strip()

        if not current or not next_text:
            return False

        if (
            self._is_service_header(current)
            or self._is_service_header(next_text)
            or self._is_numbered_header(current)
            or self._is_numbered_header(next_text)
            or next_text[0].isupper()
        ):
            return False

        if len(current.split()) <= 3 and next_text[0].islower():
            return True

        return (
            "." not in current
            and "." not in next_text
            and len(current.split()) <= 4
            and len(next_text.split()) <= 8
        )

    def _merge_broken_headers(self, headers: list[str]) -> list[str]:
        """
        Склеивает заголовки, разбитые PDF-парсером на несколько строк.

        :param headers: список заголовков-кандидатов
        :return: список заголовков после склейки
        """
        merged_headers = []
        buffer = ""

        for header in headers:
            header = header.strip()

            if not header:
                continue

            if not buffer:
                buffer = header
                continue

            if self._should_merge_with_next(buffer, header):
                buffer = f"{buffer} {header}"
            else:
                merged_headers.append(buffer)
                buffer = header

        if buffer:
            merged_headers.append(buffer)

        return merged_headers

    def _deduplicate_headers(self, headers: list[str]) -> list[str]:
        """
        Удаляет повторяющиеся заголовки с сохранением порядка.

        :param headers: список заголовков
        :return: список уникальных заголовков
        """
        result = []
        seen = set()

        for header in headers:
            key = header.lower().replace("ё", "е").strip()

            if key in seen:
                continue

            seen.add(key)
            result.append(header)

        return result

    def _is_noise(self, text: str) -> bool:
        """
        Проверяет, относится ли строка к любому известному типу шума.

        :param text: проверяемый текст
        :return: True, если строку нужно отбросить
        """
        return (
            self._is_journal_noise(text)
            or self._is_author_or_person_info(text)
            or self._is_affiliation(text)
            or self._is_abstract_noise(text)
            or self._is_reference_item(text)
            or self._is_math_or_table_noise(text)
            or self._is_figure_or_table_caption(text)
            or re.fullmatch(r"\d+(?:\.\d+)?", text) is not None
            or re.fullmatch(r"[\d\s.,%]+", text) is not None
        )

    def _extract_header(self, text: str) -> str | None:
        """
        Извлекает заголовок из очищенной строки.

        :param text: очищенный текст
        :return: найденный заголовок или None
        """
        match = re.match(r"^(Введение|Заключение)\.?\s+", text, flags=re.I)

        if match:
            return match.group(1).capitalize()

        for pattern in (
            r"^(\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^.]{3,180}?)(?:\.|\s{2,}|$)",
            r"(\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^.]{3,180}?)(?:\.|\s{2,}|$)",
        ):
            match = re.search(pattern, text)

            if match:
                found = match.group(1).strip().rstrip(".")
                return self._normalize_numbered_header(found)

        if "." not in text and len(text.split()) <= 12:
            return text

        return None
