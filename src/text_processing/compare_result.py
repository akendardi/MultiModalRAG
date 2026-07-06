import json
import re
from pathlib import Path
from dataclasses import dataclass

from rapidfuzz import fuzz


@dataclass
class HeadersCompareResult:
    """
    Результат сравнения найденных заголовков с эталонной разметкой.

    Объект хранит подробную информацию по одному документу и одному методу:
    1. Количество правильно найденных заголовков.
    2. Количество лишних найденных заголовков.
    3. Количество пропущенных эталонных заголовков.
    4. Метрики Precision, Recall и F1.
    5. Списки пропущенных, лишних и сопоставленных заголовков.
    """

    document: str
    method: str
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    missing: list[str]
    extra: list[str]
    matches: list[dict]


class HeadersComparator:
    """
    Сравнение автоматически найденных заголовков с ручной эталонной разметкой.

    Класс используется для экспериментальной оценки качества методов
    извлечения заголовков из PDF-документов.

    Основной пайплайн работы:
    1. Загружает ручную разметку заголовков из JSON-файла.
    2. Нормализует эталонные и найденные заголовки.
    3. Сравнивает найденные заголовки с эталоном через fuzzy matching.
    4. Считает TP, FP, FN.
    5. Считает Precision, Recall и F1.
    6. Формирует списки пропущенных, лишних и совпавших заголовков.
    7. Может агрегировать результаты по нескольким документам и методам.
    8. Может сохранять подробные и агрегированные результаты в JSON.
    """

    def __init__(
        self,
        gold_path: str | Path,
        threshold: int = 85,
    ):
        """
        Инициализирует компаратор заголовков.

        При создании объекта загружается JSON-файл с ручной эталонной
        разметкой заголовков. После этого объект можно использовать для
        сравнения результатов разных методов извлечения заголовков.

        :param gold_path: путь к JSON-файлу с ручной разметкой
        :param threshold: минимальный процент похожести для fuzzy-сравнения
        """
        self.gold_path = Path(gold_path)
        self.threshold = threshold
        self.gold_data = self._load_gold_headers(self.gold_path)

    @staticmethod
    def _load_gold_headers(gold_path: Path) -> dict[str, list[str]]:
        """
        Загружает эталонные заголовки из JSON-файла.

        Метод ожидает JSON следующего формата:
        [
            {
                "document": "file.pdf",
                "headers": ["Введение", "Заключение"]
            }
        ]

        После загрузки данные преобразуются в словарь вида:
        {
            "file.pdf": ["Введение", "Заключение"]
        }

        :param gold_path: путь к JSON-файлу с ручной разметкой
        :return: словарь, где ключ — имя документа, значение — список эталонных заголовков
        """
        with open(gold_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        gold_data = {}

        for item in data:
            document = item["document"]
            headers = item["headers"]
            gold_data[document] = headers

        return gold_data

    @staticmethod
    def normalize_header(text: str) -> str:
        """
        Нормализует заголовок перед сравнением.

        Метод приводит заголовок к единому виду:
        1. Переводит текст в нижний регистр.
        2. Заменяет "ё" на "е".
        3. Удаляет служебные unicode-символы.
        4. Нормализует тире.
        5. Исправляет пробелы в нумерованных заголовках.
        6. Удаляет пунктуацию в конце строки.
        7. Удаляет лишние символы.
        8. Схлопывает повторяющиеся пробелы.

        Это нужно, чтобы небольшие различия в форматировании не мешали
        сравнению заголовков.

        :param text: исходный заголовок
        :return: нормализованный заголовок
        """
        text = str(text).lower()
        text = text.replace("ё", "е")

        text = re.sub(r"[\uf000-\uf8ff]", " ", text)
        text = text.replace("\u00ad", "")
        text = text.replace("\x0e", "")
        text = text.replace("\x19", "")
        text = text.replace("\x1a", "")

        text = text.replace("–", "-")
        text = text.replace("—", "-")

        text = text.strip()

        text = re.sub(r"(\d+)\s+\.", r"\1.", text)
        text = re.sub(r"^(\d+)\.([а-яa-z])", r"\1. \2", text)
        text = re.sub(r"^(\d+(?:\.\d+)+)([а-яa-z])", r"\1 \2", text)

        text = re.sub(r"[:.;,\s]+$", "", text)
        text = re.sub(r"[^\w\s.\-]", " ", text)

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    @staticmethod
    def get_document_name(document_path: str | Path) -> str:
        """
        Возвращает имя документа без полного пути.

        Метод нужен, чтобы можно было передавать как полный путь к PDF-файлу,
        так и просто имя файла. В обоих случаях для поиска в ручной разметке
        используется только имя документа.

        :param document_path: путь к документу или имя файла
        :return: имя файла документа
        """
        return Path(document_path).name

    def _make_items(self, headers: list[str]) -> list[dict]:
        """
        Преобразует список заголовков в список элементов для сравнения.

        Каждый заголовок преобразуется в структуру:
        {
            "original": исходный заголовок,
            "norm": нормализованный заголовок
        }

        Исходный заголовок нужен для отчёта, а нормализованный — для сравнения.
        Пустые строки после нормализации удаляются.

        :param headers: список исходных заголовков
        :return: список элементов с исходной и нормализованной формой заголовка
        """
        items = []

        for header in headers:
            norm = self.normalize_header(header)

            if not norm:
                continue

            items.append({
                "original": header,
                "norm": norm,
            })

        return items

    def compare_headers(
        self,
        document: str | Path,
        predicted_headers: list[str],
        method: str,
    ) -> HeadersCompareResult:
        """
        Сравнивает заголовки одного метода с ручной эталонной разметкой.

        Метод выполняет сравнение для одного документа:
        1. Находит эталонные заголовки по имени документа.
        2. Нормализует эталонные и найденные заголовки.
        3. Для каждого найденного заголовка ищет наиболее похожий эталонный.
        4. Засчитывает совпадение, если score >= threshold.
        5. Не позволяет одному эталонному заголовку быть сопоставленным несколько раз.
        6. Считает TP, FP, FN.
        7. Считает Precision, Recall и F1.
        8. Формирует списки missing, extra и matches.

        :param document: имя документа или путь к документу
        :param predicted_headers: список заголовков, найденных методом
        :param method: название метода извлечения заголовков
        :return: результат сравнения с метриками и подробностями ошибок
        """
        document_name = self.get_document_name(document)

        if document_name not in self.gold_data:
            raise ValueError(f"Для документа {document_name} нет ручной разметки")

        gold_headers = self.gold_data[document_name]

        gold_items = self._make_items(gold_headers)
        predicted_items = self._make_items(predicted_headers)

        matched_gold = set()
        matched_predicted = set()
        matches = []

        for pred_idx, pred_item in enumerate(predicted_items):
            pred_norm = pred_item["norm"]

            best_score = 0
            best_gold_idx = None

            for gold_idx, gold_item in enumerate(gold_items):
                if gold_idx in matched_gold:
                    continue

                gold_norm = gold_item["norm"]

                score = fuzz.token_sort_ratio(pred_norm, gold_norm)

                if score > best_score:
                    best_score = score
                    best_gold_idx = gold_idx

            if best_gold_idx is not None and best_score >= self.threshold:
                matched_predicted.add(pred_idx)
                matched_gold.add(best_gold_idx)

                matches.append({
                    "gold": gold_items[best_gold_idx]["original"],
                    "predicted": pred_item["original"],
                    "gold_norm": gold_items[best_gold_idx]["norm"],
                    "predicted_norm": pred_item["norm"],
                    "score": round(best_score, 2),
                })

        true_positive = len(matched_predicted)
        false_positive = len(predicted_items) - true_positive
        false_negative = len(gold_items) - true_positive

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        missing = [
            item["original"]
            for idx, item in enumerate(gold_items)
            if idx not in matched_gold
        ]

        extra = [
            item["original"]
            for idx, item in enumerate(predicted_items)
            if idx not in matched_predicted
        ]

        return HeadersCompareResult(
            document=document_name,
            method=method,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            missing=missing,
            extra=extra,
            matches=matches,
        )

    def compare_many_methods(
        self,
        document: str | Path,
        methods_headers: dict[str, list[str]],
    ) -> list[HeadersCompareResult]:
        """
        Сравнивает несколько методов извлечения заголовков на одном документе.

        Метод принимает словарь вида:
        {
            "MinerU": [...],
            "MuPDF": [...],
            "LLM": [...]
        }

        Для каждого метода вызывается compare_headers, после чего результаты
        возвращаются одним списком.

        :param document: имя документа или путь к документу
        :param methods_headers: словарь, где ключ — название метода, значение — список найденных заголовков
        :return: список результатов сравнения для каждого метода
        """
        results = []

        for method, headers in methods_headers.items():
            result = self.compare_headers(
                document=document,
                predicted_headers=headers,
                method=method,
            )
            results.append(result)

        return results

    @staticmethod
    def summarize_results(
        results: list[HeadersCompareResult],
    ) -> dict[str, dict]:
        """
        Агрегирует результаты сравнения по методам.

        Метод группирует результаты по названию метода и считает:
        1. Общее количество документов.
        2. Суммарные TP, FP и FN.
        3. Micro-метрики по суммарным TP, FP и FN.
        4. Macro-метрики как среднее значение метрик по документам.

        Micro-метрики показывают общее качество метода на всём наборе данных.
        Macro-метрики дают одинаковый вес каждому документу независимо от
        количества заголовков в нём.

        :param results: список результатов сравнения
        :return: словарь с агрегированной статистикой по каждому методу
        """
        grouped = {}

        for result in results:
            method = result.method

            if method not in grouped:
                grouped[method] = {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "documents": 0,
                    "precision_values": [],
                    "recall_values": [],
                    "f1_values": [],
                }

            grouped[method]["true_positive"] += result.true_positive
            grouped[method]["false_positive"] += result.false_positive
            grouped[method]["false_negative"] += result.false_negative
            grouped[method]["documents"] += 1

            grouped[method]["precision_values"].append(result.precision)
            grouped[method]["recall_values"].append(result.recall)
            grouped[method]["f1_values"].append(result.f1)

        summary = {}

        for method, values in grouped.items():
            tp = values["true_positive"]
            fp = values["false_positive"]
            fn = values["false_negative"]

            micro_precision = tp / max(tp + fp, 1)
            micro_recall = tp / max(tp + fn, 1)
            micro_f1 = 2 * micro_precision * micro_recall / max(
                micro_precision + micro_recall,
                1e-9,
            )

            macro_precision = sum(values["precision_values"]) / max(values["documents"], 1)
            macro_recall = sum(values["recall_values"]) / max(values["documents"], 1)
            macro_f1 = sum(values["f1_values"]) / max(values["documents"], 1)

            summary[method] = {
                "documents": values["documents"],
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,

                "precision": round(micro_precision, 4),
                "recall": round(micro_recall, 4),
                "f1": round(micro_f1, 4),

                "macro_precision": round(macro_precision, 4),
                "macro_recall": round(macro_recall, 4),
                "macro_f1": round(macro_f1, 4),
            }

        return summary

    @staticmethod
    def result_to_dict(result: HeadersCompareResult) -> dict:
        """
        Преобразует объект HeadersCompareResult в словарь.

        Метод нужен перед сохранением результатов в JSON, так как dataclass
        напрямую не сериализуется через json.dump.

        :param result: объект с результатом сравнения
        :return: словарь с результатами сравнения
        """
        return {
            "document": result.document,
            "method": result.method,
            "true_positive": result.true_positive,
            "false_positive": result.false_positive,
            "false_negative": result.false_negative,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "missing": result.missing,
            "extra": result.extra,
            "matches": result.matches,
        }

    @staticmethod
    def save_results(
        results: list[HeadersCompareResult],
        output_path: str | Path,
    ) -> None:
        """
        Сохраняет подробные результаты сравнения в JSON-файл.

        В файл сохраняется список результатов по каждому документу и методу.
        Для каждого результата сохраняются TP, FP, FN, Precision, Recall, F1,
        а также списки missing, extra и matches.

        :param results: список результатов сравнения
        :param output_path: путь к JSON-файлу для сохранения
        :return: None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            HeadersComparator.result_to_dict(result)
            for result in results
        ]

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def save_summary(
        summary: dict[str, dict],
        output_path: str | Path,
    ) -> None:
        """
        Сохраняет агрегированную статистику в JSON-файл.

        В файл сохраняются итоговые показатели по каждому методу:
        1. Количество документов.
        2. Суммарные TP, FP и FN.
        3. Micro Precision, Recall и F1.
        4. Macro Precision, Recall и F1.

        :param summary: словарь с агрегированной статистикой
        :param output_path: путь к JSON-файлу для сохранения
        :return: None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=4)