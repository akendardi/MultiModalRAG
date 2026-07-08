from dataclasses import dataclass
from enum import Enum, auto


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
    TABLE = auto()
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
    :param start_idx: индекс начала контента
    :param end_idx: индекс конца контента
    """

    source: str
    page: int
    section: str | None
    content_type: DocumentType
    chunk_id: str
    asset_path: str | None
    content: str | None

    start_idx: int|None = None
    end_idx: int|None = None


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