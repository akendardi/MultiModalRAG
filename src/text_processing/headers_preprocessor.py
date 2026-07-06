import re


class HeadersProcessor:
    """
    Обработка и фильтрация заголовков, извлечённых из PDF-документов.

    Класс выполняет постобработку заголовков-кандидатов:
    1. Исправляет часть проблем с кодировкой и PDF-шрифтами.
    2. Удаляет служебные символы и нормализует пробелы.
    3. Отфильтровывает шум: авторов, аффилиации, DOI, УДК, ISSN, колонтитулы.
    4. Удаляет аннотации, ключевые слова, подписи рисунков и таблиц.
    5. Удаляет элементы формул, таблиц, схем и списка литературы.
    6. Нормализует нумерованные заголовки.
    7. Склеивает заголовки, разбитые на несколько строк.
    8. Удаляет дубликаты с сохранением порядка.
    """

    @staticmethod
    def looks_like_mojibake(text: str) -> bool:
        """
        Проверяет, похожа ли строка на текст с битой кодировкой.

        Метод используется для обнаружения ситуации, когда русский текст был
        ошибочно прочитан как latin1 вместо cp1251.

        :param text: исходная строка
        :return: True, если строка похожа на битую кодировку
        """
        mojibake_chars = (
            "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß"
            "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
        )

        count = sum(ch in mojibake_chars for ch in text)

        return count >= 3

    @staticmethod
    def fix_mojibake(text: str) -> str:
        """
        Восстанавливает русский текст при ошибочном декодировании latin1.

        Метод пытается перекодировать строку из latin1 в cp1251.
        Исправленная версия возвращается только в том случае, если после
        преобразования количество кириллических символов увеличилось.

        :param text: исходная строка
        :return: исправленная строка или исходная строка, если восстановление не требуется
        """
        if not HeadersProcessor.looks_like_mojibake(text):
            return text

        try:
            fixed = text.encode(
                "latin1",
                errors="ignore",
            ).decode(
                "cp1251",
                errors="ignore",
            )
        except Exception:
            return text

        old_cyr = sum(
            "а" <= ch.lower() <= "я" or ch.lower() == "ё"
            for ch in text
        )

        new_cyr = sum(
            "а" <= ch.lower() <= "я" or ch.lower() == "ё"
            for ch in fixed
        )

        if new_cyr > old_cyr:
            return fixed

        return text

    @staticmethod
    def fix_pdf_cyrillic_font_map(text: str) -> str:
        """
        Исправляет часть символов, повреждённых при извлечении текста из PDF.

        Метод заменяет некоторые нестандартные символы, которые появляются
        из-за особенностей встроенных PDF-шрифтов, на соответствующие буквы
        кириллицы.

        :param text: исходная строка
        :return: строка с частично восстановленной кириллицей
        """
        char_map = {
            "Ⱥ": "А",
            "Ȼ": "Б",
            "ȼ": "В",
            "Ƚ": "Г",
            "Ⱦ": "Д",
            "ȿ": "Е",
            "ɀ": "Ж",
            "Ɂ": "З",
            "ɂ": "И",
            "Ƀ": "Й",
            "Ʉ": "К",
            "Ʌ": "Л",
            "Ɇ": "М",
            "ɇ": "Н",
            "Ɉ": "О",
            "ɉ": "П",
            "Ɋ": "Р",
            "ɋ": "С",
            "Ɍ": "Т",
            "ɍ": "У",
            "Ɏ": "Ф",
            "ɏ": "Х",
            "ɐ": "Ц",
            "ɑ": "ч",
            "ɒ": "Ш",
            "ɓ": "щ",
            "ɔ": "ъ",
            "ɕ": "ы",
            "ɖ": "ь",
            "ɗ": "э",
            "ɘ": "Ю",
            "ə": "я",

            "ɚ": "а",
            "ɛ": "б",
            "ɜ": "в",
            "ɝ": "г",
            "ɞ": "д",
            "ɟ": "е",
            "ɠ": "ж",
            "ɡ": "з",
            "ɢ": "и",
            "ɣ": "й",
            "ɤ": "к",
            "ɥ": "л",
            "ɦ": "м",
            "ɧ": "н",
            "ɨ": "о",
            "ɩ": "п",
            "ɪ": "р",
            "ɫ": "с",
            "ɬ": "т",
            "ɭ": "у",
            "ɮ": "ф",
            "ɯ": "х",
            "ɰ": "ц",
            "ɱ": "ч",
            "ɲ": "ш",
            "ɳ": "щ",
            "ɴ": "ъ",
            "ɵ": "ы",
            "ɶ": "ь",
            "ɷ": "э",
            "ɸ": "ю",
            "ɹ": "я",
        }

        return "".join(char_map.get(ch, ch) for ch in text)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Очищает и нормализует одну строку текста.

        Метод выполняет базовую нормализацию:
        1. Исправляет битую кодировку.
        2. Исправляет часть PDF-символов кириллицы.
        3. Удаляет служебные unicode-символы.
        4. Нормализует тире.
        5. Схлопывает повторяющиеся пробелы.
        6. Удаляет copyright-строки.

        :param text: исходная строка
        :return: очищенная строка
        """
        text = str(text)

        text = HeadersProcessor.fix_mojibake(text)
        text = HeadersProcessor.fix_pdf_cyrillic_font_map(text)

        text = re.sub(r"[\uf000-\uf8ff]", " ", text)
        text = text.replace("\u00ad", "")
        text = text.replace("\x0e", "")
        text = text.replace("\x19", "")
        text = text.replace("\x1a", "")
        text = text.replace("\ufeff", "")

        text = text.replace("–", "-")
        text = text.replace("—", "-")

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        if re.match(r"^(©|\(c\)|copyright)\s*", text, flags=re.I):
            return ""

        return text

    @staticmethod
    def is_literature_header(text: str) -> bool:
        """
        Проверяет, является ли строка заголовком списка литературы.

        Метод определяет стандартные варианты названий раздела литературы:
        "Литература", "Список литературы", "References", "Sources" и т.п.

        :param text: проверяемая строка
        :return: True, если строка является заголовком литературы
        """
        text = HeadersProcessor.clean_text(text).strip().lower()

        return bool(
            re.fullmatch(
                r"(список\s+литературы|литература|библиографический\s+список|"
                r"библиографические\s+ссылки|ссылки\s+на\s+источники|"
                r"список\s+источников|references|sources)\s*:?",
                text,
                flags=re.I,
            )
        )

    @staticmethod
    def is_reference_item(text: str) -> bool:
        """
        Проверяет, похожа ли строка на пункт списка литературы.

        Метод ищет строки вида:
        "1. Иванов И.И. ..."
        "2. Petrov A. ..."

        :param text: проверяемая строка
        :return: True, если строка похожа на библиографическую запись
        """
        text = text.strip()

        if re.match(
            r"^\d+\.\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-]+[, ]+\s*[A-ZА-ЯЁ]\.?",
            text,
        ):
            return True

        if re.match(
            r"^\d+\.\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-]+",
            text,
        ) and "," in text:
            return True

        return False

    @staticmethod
    def is_journal_noise(text: str) -> bool:
        """
        Проверяет, является ли строка журнальным или служебным шумом.

        Метод удаляет номера страниц, номера выпусков, DOI, УДК, ISSN,
        названия журналов, колонтитулы и другие служебные элементы.

        :param text: проверяемая строка
        :return: True, если строка является служебным шумом
        """
        text = text.strip()
        low = text.lower()

        if re.fullmatch(r"№\s*\d+\s*,?\s*\d{4}", text):
            return True

        if re.fullmatch(r"\d+\s*№\s*\d+", text):
            return True

        if re.fullmatch(r"№\s*\d+", text):
            return True

        if re.fullmatch(r"\d{1,4}", text):
            return True

        journal_words = [
            "педагогика и психология",
            "наука. инновации. технологии",
            "scientific journal",
            "компьютерная оптика",
            "технические науки",
            "технічні науки",
            "удк",
            "doi",
            "issn",
            "том",
            "vol.",
            "no.",
        ]

        if any(word in low for word in journal_words):
            return True

        return False

    @staticmethod
    def is_author_or_person_info(text: str) -> bool:
        """
        Проверяет, похожа ли строка на ФИО автора или сведения об авторе.

        Метод отфильтровывает:
        1. ФИО в русском формате.
        2. Фамилию с инициалами.
        3. Инициалы с фамилией.
        4. Должности и научные степени.
        5. Англоязычные сведения об авторах.

        :param text: проверяемая строка
        :return: True, если строка похожа на автора или сведения об авторе
        """
        text = text.strip()
        low = text.lower()

        if re.fullmatch(
            r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+,?.*",
            text,
        ):
            return True

        if re.fullmatch(
            r"[А-ЯЁ][а-яё]+ [А-ЯЁ]\.\s*[А-ЯЁ]\..*",
            text,
        ):
            return True

        if re.fullmatch(
            r"[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+.*",
            text,
        ):
            return True

        person_info_words = [
            "доцент",
            "профессор",
            "магистрант",
            "аспирант",
            "студент",
            "кандидат",
            "д-р",
            "доктор",
            "кафедра",
            "associate professor",
            "professor",
            "student",
            "department",
            "сведения об авторе",
            "сведения об авторах",
        ]

        if any(word in low for word in person_info_words):
            return True

        if re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+,?.*", text):
            english_author_markers = [
                "associate professor",
                "department",
                "university",
                "institute",
                "student",
                "professor",
                "candidate",
            ]

            if any(marker in low for marker in english_author_markers):
                return True

        return False

    @staticmethod
    def is_affiliation(text: str) -> bool:
        """
        Проверяет, похожа ли строка на аффилиацию автора.

        Метод отфильтровывает названия университетов, институтов, кафедр,
        академий, e-mail, города и страны.

        :param text: проверяемая строка
        :return: True, если строка похожа на организацию или аффилиацию
        """
        low = text.lower()

        affiliation_words = [
            "университет",
            "институт",
            "академия",
            "фгаоу",
            "фгбоу",
            "впо",
            "кафедра",
            "г.",
            "e-mail",
            "email",
            "university",
            "institute",
            "department",
            "academy",
            "russia",
            "россия",
            "северо-кавказский",
            "north-caucasus",
        ]

        if any(word in low for word in affiliation_words):
            return True

        if re.match(r"^\d+\s+[A-ZА-ЯЁ]", text) and "," in text:
            return True

        return False

    @staticmethod
    def is_abstract_noise(text: str) -> bool:
        """
        Проверяет, относится ли строка к аннотации или ключевым словам.

        Метод удаляет строки с маркерами "Аннотация", "Ключевые слова",
        "Abstract", "Keywords", а также длинные англоязычные строки,
        которые часто являются переводом названия или abstract-блоком.

        :param text: проверяемая строка
        :return: True, если строка похожа на аннотацию, keywords или abstract
        """
        low = text.lower().strip()

        bad_phrases = [
            "аннотация",
            "ключевые слова",
            "abstract",
            "keywords",
            "key words",
            "the educational robotics",
            "engineering education",
            "robotics, educational robotics",
        ]

        if any(phrase in low for phrase in bad_phrases):
            return True

        latin_count = sum("a" <= ch.lower() <= "z" for ch in text)

        cyr_count = sum(
            "а" <= ch.lower() <= "я" or ch.lower() == "ё"
            for ch in text
        )

        if latin_count > 25 and cyr_count == 0:
            return True

        return False

    @staticmethod
    def is_math_or_table_noise(text: str) -> bool:
        """
        Проверяет, похожа ли строка на формулу, числовую строку или элемент таблицы.

        Метод удаляет:
        1. Типовые заголовки столбцов таблиц.
        2. Строки, состоящие только из чисел и знаков.
        3. Строки с большим количеством математических символов.
        4. Короткие формульные выражения.

        :param text: проверяемая строка
        :return: True, если строка похожа на формулу или элемент таблицы
        """
        low = text.lower().strip()

        table_items = [
            "№ п/п",
            "наименование и содержание операции",
            "оборудование",
            "время",
            "время (мкс)",
            "критерий",
            "метод",
        ]

        if low in table_items:
            return True

        if re.fullmatch(r"[\d\s.,%]+", text):
            return True

        math_chars = "∑∏∫×÷√≈≠≤≥∞{}[]"

        math_count = sum(ch in math_chars for ch in text)

        if math_count >= 2:
            return True

        if re.fullmatch(r"[A-Za-zА-Яа-яЁё]\s+[A-Za-zА-Яа-яЁё].*", text):
            if any(ch in text for ch in ":=∑∏×"):
                return True

        return False

    @staticmethod
    def is_diagram_or_list_item(text: str) -> bool:
        """
        Проверяет, похожа ли строка на элемент схемы или внутреннего списка.

        Метод удаляет отдельные элементы диаграмм, схем и списков,
        которые PDF-парсер мог ошибочно принять за заголовки.

        :param text: проверяемая строка
        :return: True, если строка похожа на элемент схемы или списка
        """
        low = text.lower().strip().rstrip(";:.,")

        diagram_items = [
            "по характеру обучения",
            "обучение с учителем",
            "обучение без учителя",
            "по типу настройки весов",
            "по типу входной информации делит их",
            "аналоговые - входная информация представлена в форме действительных чисел",
            "входной слой",
            "скрытые слои",
            "выходной слой",
        ]

        if low in diagram_items:
            return True

        if text.strip().endswith(";"):
            return True

        return False

    @staticmethod
    def is_figure_or_table_caption(text: str) -> bool:
        """
        Проверяет, является ли строка подписью рисунка или таблицы.

        Метод удаляет строки вида:
        "Рис. 1 ...", "Рисунок 2 ...", "Табл. 1 ...", "Таблица 3 ...".

        :param text: проверяемая строка
        :return: True, если строка является подписью рисунка или таблицы
        """
        low = text.lower().strip()

        if re.search(r"\bрис\.?\s*\d+", low):
            return True

        if re.search(r"\bрисунок\s*\d+", low):
            return True

        if re.search(r"\bтабл\.?\s*\d+", low):
            return True

        if re.search(r"\bтаблица\s*\d+", low):
            return True

        return False

    @staticmethod
    def has_sentence_noise(text: str) -> bool:
        """
        Проверяет, похожа ли строка на обычное предложение.

        Метод ищет глагольные и вводные маркеры, характерные для обычного
        текста, а не для заголовков.

        :param text: проверяемая строка
        :return: True, если строка похожа на фрагмент обычного текста
        """
        low = text.lower()

        bad_sentence_markers = [
            "является",
            "являются",
            "внедрила",
            "внедрил",
            "внедрившей",
            "используется",
            "позволяет",
            "представляет",
            "рассматривается",
            "состоит",
            "имеет",
            "может быть",
            "следует",
            "необходимо",
            "таким образом",
        ]

        if any(marker in low for marker in bad_sentence_markers):
            return True

        return False

    @staticmethod
    def remove_page_number_tail(text: str) -> str:
        """
        Удаляет номер страницы в конце строки.

        Метод используется для случаев, когда заголовок был извлечён вместе
        с номером страницы из оглавления или колонтитула.

        :param text: исходная строка
        :return: строка без номера страницы в конце
        """
        return re.sub(
            r"(\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^.\d]{3,160})\s+\d+$",
            r"\1",
            text,
        ).strip()

    @staticmethod
    def cut_after_literature_word(text: str) -> str:
        """
        Обрезает строку после начала блока литературы.

        Метод используется для случаев, когда в одной строке после заголовка
        случайно оказался текст списка литературы.

        :param text: исходная строка
        :return: строка, обрезанная перед блоком литературы
        """
        return re.sub(
            r"\s+(литература|список\s+литературы|references|sources)\s*:?.*$",
            "",
            text,
            flags=re.I,
        ).strip()

    @staticmethod
    def normalize_numbered_header(text: str) -> str:
        """
        Нормализует формат нумерованного заголовка.

        Метод приводит заголовки к виду:
        "1 Введение" -> "1. Введение"
        "1.. Введение" -> "1. Введение"

        :param text: исходная строка
        :return: строка с нормализованной нумерацией
        """
        text = re.sub(r"^(\d+(?:\.\d+)*)\s+", r"\1. ", text)

        text = re.sub(
            r"^(\d+(?:\.\d+)*)\.\.\s+",
            r"\1. ",
            text,
        )

        return text.strip()

    @staticmethod
    def is_service_header(text: str) -> bool:
        """
        Проверяет, является ли строка стандартным служебным заголовком.

        Метод определяет типовые названия разделов:
        "Введение", "Заключение", "Основная часть",
        "Список литературы", "References" и т.п.

        :param text: проверяемая строка
        :return: True, если строка является служебным заголовком
        """
        return bool(
            re.fullmatch(
                r"(введение|заключение|основная часть|список литературы|"
                r"литература|библиографический список|библиографические ссылки|"
                r"благодарности|references|sources)",
                text.strip().lower(),
                flags=re.I,
            )
        )

    @staticmethod
    def is_numbered_header(text: str) -> bool:
        """
        Проверяет, является ли строка нумерованным заголовком.

        Метод определяет строки вида:
        "1. Введение", "2 Методы", "3.1 Постановка задачи".

        :param text: проверяемая строка
        :return: True, если строка является нумерованным заголовком
        """
        return bool(
            re.match(
                r"^\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]",
                text.strip(),
            )
        )

    @staticmethod
    def should_merge_with_next(current: str, next_text: str) -> bool:
        """
        Проверяет, нужно ли объединить текущую строку со следующей.

        Метод используется для восстановления заголовков, которые были
        разбиты PDF-парсером на несколько строк.

        Строки не объединяются, если одна из них является служебным или
        нумерованным заголовком.

        :param current: текущая строка
        :param next_text: следующая строка
        :return: True, если строки нужно объединить
        """
        current = current.strip()
        next_text = next_text.strip()

        if not current or not next_text:
            return False

        if HeadersProcessor.is_service_header(current):
            return False

        if HeadersProcessor.is_service_header(next_text):
            return False

        if HeadersProcessor.is_numbered_header(current):
            return False

        if HeadersProcessor.is_numbered_header(next_text):
            return False

        if next_text[0].isupper():
            return False

        if len(current.split()) <= 3 and next_text[0].islower():
            return True

        if (
            "." not in current
            and "." not in next_text
            and len(current.split()) <= 4
            and len(next_text.split()) <= 8
        ):
            return True

        return False

    @staticmethod
    def merge_broken_headers(headers: list[str]) -> list[str]:
        """
        Склеивает заголовки, разбитые на несколько строк.

        Метод последовательно проходит по списку заголовков и объединяет
        соседние строки, если они выглядят как части одного заголовка.

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

            if HeadersProcessor.should_merge_with_next(buffer, header):
                buffer = f"{buffer} {header}"
            else:
                merged_headers.append(buffer)
                buffer = header

        if buffer:
            merged_headers.append(buffer)

        return merged_headers

    @staticmethod
    def deduplicate_headers(headers: list[str]) -> list[str]:
        """
        Удаляет повторяющиеся заголовки с сохранением исходного порядка.

        Для сравнения дубликатов используется упрощённый ключ:
        строка приводится к нижнему регистру, буква "ё" заменяется на "е",
        а пробелы по краям удаляются.

        :param headers: список заголовков
        :return: список заголовков без дубликатов
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

    @staticmethod
    def clear_headers(headers: list[str]) -> list[str]:
        """
        Очищает список заголовков-кандидатов.

        Метод выполняет полный пайплайн постобработки:
        1. Очищает каждую строку от служебных символов.
        2. Удаляет номера страниц в конце строк.
        3. Обрезает строки при обнаружении блока литературы.
        4. Нормализует нумерованные заголовки.
        5. Удаляет служебный шум, авторов, аффилиации и abstract-блоки.
        6. Удаляет пункты списка литературы, формулы, таблицы и подписи рисунков.
        7. Определяет начало основной структуры документа.
        8. Извлекает обычные и нумерованные заголовки.
        9. Склеивает заголовки, разбитые на несколько строк.
        10. Удаляет дубликаты.

        :param headers: список заголовков-кандидатов
        :return: очищенный список заголовков
        """
        clean_headers = []
        is_start = False

        for text in headers:
            text = HeadersProcessor.clean_text(text)
            text = HeadersProcessor.remove_page_number_tail(text)
            text = HeadersProcessor.cut_after_literature_word(text)
            text = HeadersProcessor.normalize_numbered_header(text)

            if not text or len(text) < 5:
                continue

            low = text.lower()

            if HeadersProcessor.is_literature_header(text):
                if is_start:
                    clean_headers.append(text)
                    break

                clean_headers.append(text)
                break

            if HeadersProcessor.is_journal_noise(text):
                continue

            if HeadersProcessor.is_author_or_person_info(text):
                continue

            if HeadersProcessor.is_affiliation(text):
                continue

            if HeadersProcessor.is_abstract_noise(text):
                continue

            if HeadersProcessor.is_reference_item(text):
                continue

            if HeadersProcessor.is_math_or_table_noise(text):
                continue

            if HeadersProcessor.is_diagram_or_list_item(text):
                continue

            if HeadersProcessor.is_figure_or_table_caption(text):
                continue

            if re.fullmatch(r"\d+(?:\.\d+)?", text):
                continue

            if re.fullmatch(r"[\d\s.,%]+", text):
                continue

            if text.replace(" ", "").isalpha() and text.islower():
                continue

            upper_letters = [
                ch for ch in text
                if ch.isalpha() and ch.isupper()
            ]

            all_letters = [
                ch for ch in text
                if ch.isalpha()
            ]

            upper_coef = len(upper_letters) / max(len(all_letters), 1)

            if text.isupper() or upper_coef > 0.85:
                is_start = True
                continue

            if "ключевые слова" in low or "keywords" in low:
                is_start = True
                continue

            if not is_start:
                if (
                    text.lower().startswith("введение")
                    or HeadersProcessor.is_numbered_header(text)
                    or HeadersProcessor.is_service_header(text)
                ):
                    is_start = True
                else:
                    continue

            if HeadersProcessor.has_sentence_noise(text):
                continue

            m = re.match(r"^(Введение|Заключение)\.?\s+", text, flags=re.I)

            if m:
                clean_headers.append(m.group(1).capitalize())
                continue

            m = re.match(
                r"^(\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^.]{3,180}?)(?:\.|\s{2,}|$)",
                text,
            )

            if m:
                found = m.group(1).strip().rstrip(".")
                found = HeadersProcessor.normalize_numbered_header(found)
                clean_headers.append(found)
                continue

            m = re.search(
                r"(\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^.]{3,180}?)(?:\.|\s{2,}|$)",
                text,
            )

            if m:
                found = m.group(1).strip().rstrip(".")
                found = HeadersProcessor.normalize_numbered_header(found)
                clean_headers.append(found)
                continue

            if "." not in text and len(text.split()) <= 12:
                clean_headers.append(text)
                continue

        clean_headers = HeadersProcessor.merge_broken_headers(clean_headers)
        clean_headers = HeadersProcessor.deduplicate_headers(clean_headers)

        return clean_headers