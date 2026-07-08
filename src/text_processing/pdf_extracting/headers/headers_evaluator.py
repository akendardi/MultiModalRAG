from __future__ import annotations

import json
import re
from pathlib import Path

from src.paths import PROJECT_ROOT
from src.text_processing.pdf_extracting.headers.headers_extractor import HeadersExtractor
from src.text_processing.pdf_extracting.headers.headers_processor import HeadersProcessor
from src.text_processing.pdf_extracting.mineru import MineruReader


class HeadersEvaluator:
    """
    Оценивает качество извлечения заголовков из PDF-документов.

    Класс выполняет полный экспериментальный пайплайн:
    1. Загружает список документов с ручной разметкой из gold_headers.json.
    2. Берёт PDF-файлы из директории data/pdfs.
    3. Извлекает заголовки разными способами:
       - MinerU raw;
       - MinerU + rules;
       - MuPDF raw;
       - MuPDF + rules;
       - mixed_clean.
    4. Сравнивает найденные заголовки с эталонной разметкой.
    5. Считает Precision, Recall, F1, TP, FP, FN.
    6. Возвращает pandas-таблицы с подробными и итоговыми результатами.
    7. Строит графики по итоговым метрикам.
    """

    def __init__(
            self,
            pdfs_dir: str | Path | None = None,
            gold_path: str | Path | None = None,
            threshold: int = 85,
    ):
        """
        Инициализирует объект для оценки качества извлечения заголовков.

        :param pdfs_dir: директория с PDF-файлами
        :param gold_path: путь к JSON-файлу с ручной разметкой заголовков
        :param threshold: минимальный процент похожести для fuzzy-сравнения
        """
        self.pdfs_dir = Path(pdfs_dir) if pdfs_dir else PROJECT_ROOT / "data" / "pdfs"
        self.gold_path = Path(gold_path) if gold_path else PROJECT_ROOT / "data" / "gold_headers.json"

        self.threshold = threshold

        self.headers_extractor = HeadersExtractor(
            mineru_reader=MineruReader(),
            golden_headers_path=self.gold_path,
        )
        self.header_processor = HeadersProcessor()
        self.gold_data = self._load_gold_headers()

    def _load_gold_headers(self) -> dict[str, list[str]]:
        """
        Загружает эталонные заголовки из JSON-файла.

        :return: словарь, где ключ — имя документа, значение — список заголовков
        """
        with open(self.gold_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            item["document"]: item["headers"]
            for item in data
        }

    def _normalize_header(self, text: str) -> str:
        """
        Нормализует заголовок перед fuzzy-сравнением.

        :param text: исходный заголовок
        :return: нормализованный заголовок
        """
        text = str(text).lower()
        text = text.replace("ё", "е")
        text = re.sub(r"[\uf000-\uf8ff]", " ", text)

        for char in ("\u00ad", "\x0e", "\x19", "\x1a"):
            text = text.replace(char, "")

        text = text.replace("–", "-").replace("—", "-")
        text = text.strip()
        text = re.sub(r"(\d+)\s+\.", r"\1.", text)
        text = re.sub(r"^(\d+)\.([а-яa-z])", r"\1. \2", text)
        text = re.sub(r"^(\d+(?:\.\d+)+)([а-яa-z])", r"\1 \2", text)
        text = re.sub(r"[:.;,\s]+$", "", text)
        text = re.sub(r"[^\w\s.\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _make_compare_items(self, headers: list[str]) -> list[dict]:
        """
        Готовит заголовки к сравнению.

        :param headers: список исходных заголовков
        :return: список элементов с исходной и нормализованной формой
        """
        items = []

        for header in headers:
            normalized = self._normalize_header(header)

            if normalized:
                items.append({
                    "original": header,
                    "normalized": normalized,
                })

        return items

    def _compare_headers(
            self,
            document: str | Path,
            predicted_headers: list[str],
            method: str,
    ) -> dict:
        """
        Сравнивает найденные заголовки с эталонной разметкой.

        :param document: имя документа или путь к PDF-файлу
        :param predicted_headers: найденные заголовки
        :param method: название метода извлечения
        :return: словарь с метриками и списками ошибок
        """
        from rapidfuzz import fuzz

        document_name = Path(document).name

        if document_name not in self.gold_data:
            raise ValueError(f"Для документа {document_name} нет ручной разметки")

        gold_items = self._make_compare_items(self.gold_data[document_name])
        predicted_items = self._make_compare_items(predicted_headers)

        matched_gold = set()
        matched_predicted = set()
        matches = []

        for pred_idx, pred_item in enumerate(predicted_items):
            best_score = 0
            best_gold_idx = None

            for gold_idx, gold_item in enumerate(gold_items):
                if gold_idx in matched_gold:
                    continue

                score = fuzz.token_sort_ratio(
                    pred_item["normalized"],
                    gold_item["normalized"],
                )

                if score > best_score:
                    best_score = score
                    best_gold_idx = gold_idx

            if best_gold_idx is not None and best_score >= self.threshold:
                matched_predicted.add(pred_idx)
                matched_gold.add(best_gold_idx)
                matches.append({
                    "gold": gold_items[best_gold_idx]["original"],
                    "predicted": pred_item["original"],
                    "gold_norm": gold_items[best_gold_idx]["normalized"],
                    "predicted_norm": pred_item["normalized"],
                    "score": round(best_score, 2),
                })

        true_positive = len(matched_predicted)
        false_positive = len(predicted_items) - true_positive
        false_negative = len(gold_items) - true_positive

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        return {
            "document": document_name,
            "method": method,
            "TP": true_positive,
            "FP": false_positive,
            "FN": false_negative,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "missing": [
                item["original"]
                for idx, item in enumerate(gold_items)
                if idx not in matched_gold
            ],
            "extra": [
                item["original"]
                for idx, item in enumerate(predicted_items)
                if idx not in matched_predicted
            ],
            "matches": matches,
        }

    def _compare_many_methods(
            self,
            document: str | Path,
            methods_headers: dict[str, list[str]],
    ) -> list[dict]:
        """
        Сравнивает несколько методов извлечения заголовков.

        :param document: имя документа или путь к PDF-файлу
        :param methods_headers: словарь method_name -> headers
        :return: список результатов сравнения
        """
        return [
            self._compare_headers(
                document=document,
                predicted_headers=headers,
                method=method,
            )
            for method, headers in methods_headers.items()
        ]

    def _summarize_results(self, results: list[dict]) -> dict[str, dict]:
        """
        Агрегирует результаты сравнения по методам.

        :param results: список результатов сравнения
        :return: итоговая статистика по каждому методу
        """
        grouped = {}

        for result in results:
            method = result["method"]

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

            grouped[method]["true_positive"] += result["TP"]
            grouped[method]["false_positive"] += result["FP"]
            grouped[method]["false_negative"] += result["FN"]
            grouped[method]["documents"] += 1
            grouped[method]["precision_values"].append(result["precision"])
            grouped[method]["recall_values"].append(result["recall"])
            grouped[method]["f1_values"].append(result["f1"])

        summary = {}

        for method, values in grouped.items():
            true_positive = values["true_positive"]
            false_positive = values["false_positive"]
            false_negative = values["false_negative"]

            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)

            documents_count = max(values["documents"], 1)
            macro_precision = sum(values["precision_values"]) / documents_count
            macro_recall = sum(values["recall_values"]) / documents_count
            macro_f1 = sum(values["f1_values"]) / documents_count

            summary[method] = {
                "documents": values["documents"],
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "macro_precision": round(macro_precision, 4),
                "macro_recall": round(macro_recall, 4),
                "macro_f1": round(macro_f1, 4),
            }

        return summary

    def collect_headers_for_document(
            self,
            pdf_path: str | Path,
    ) -> dict[str, list[str]]:
        """
        Извлекает заголовки из одного PDF-документа разными методами.

        Метод получает кандидаты из MinerU и MuPDF, затем дополнительно
        применяет эвристическую очистку через HeadersProcessor.

        :param pdf_path: путь к PDF-файлу
        :return: словарь method_name -> headers
        """
        pdf_path = Path(pdf_path)

        mineru_raw = self.headers_extractor.get_headers_by_mineru(str(pdf_path))
        mupdf_raw = self.headers_extractor.get_headers_by_mupdf(str(pdf_path))

        mineru_clean = self.header_processor.clear_headers(mineru_raw)
        mupdf_clean = self.header_processor.clear_headers(mupdf_raw)

        mixed_clean = self.header_processor.clear_headers(
            mineru_clean + mupdf_clean
        )

        return {
            "MinerU raw": mineru_raw,
            "MinerU + rules": mineru_clean,
            "MuPDF raw": mupdf_raw,
            "MuPDF + rules": mupdf_clean,
            "mixed_clean": mixed_clean,
        }

    def results_to_dataframe(self, results) -> pd.DataFrame:
        """
        Преобразует список HeadersCompareResult в pandas DataFrame.

        :param results: список результатов сравнения
        :return: таблица с результатами по документам и методам
        """
        import pandas as pd

        rows = []

        for result in results:
            rows.append({
                "document": result["document"],
                "method": result["method"],
                "TP": result["TP"],
                "FP": result["FP"],
                "FN": result["FN"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "missing": result["missing"],
                "extra": result["extra"],
                "matches": result["matches"],
            })

        return pd.DataFrame(rows)

    def summary_to_dataframe(self, summary: dict[str, dict]) -> pd.DataFrame:
        """
        Преобразует итоговую статистику в pandas DataFrame.

        :param summary: словарь с агрегированными метриками
        :return: таблица с итоговыми метриками по методам
        """
        import pandas as pd

        rows = []

        for method, values in summary.items():
            rows.append({
                "method": method,
                "documents": values["documents"],
                "TP": values["true_positive"],
                "FP": values["false_positive"],
                "FN": values["false_negative"],
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "macro_precision": values["macro_precision"],
                "macro_recall": values["macro_recall"],
                "macro_f1": values["macro_f1"],
            })

        return pd.DataFrame(rows)

    def evaluate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Считает метрики качества извлечения заголовков.

        Метод выполняет пайплайн:
        1. Берёт все PDF-файлы из указанной директории.
        2. Оставляет только файлы, которые есть в gold_headers.json.
        3. Извлекает заголовки через MinerU и MuPDF.
        4. Применяет HeadersProcessor.
        5. Сравнивает результаты с ручной разметкой.
        6. Возвращает две pandas-таблицы.

        :return: results_df, summary_df
        """
        gold_documents = set(self.gold_data)
        pdf_paths = sorted(self.pdfs_dir.glob("*.pdf"))

        all_results = []

        for pdf_path in pdf_paths:
            if pdf_path.name not in gold_documents:
                print(f"Пропуск: для {pdf_path.name} нет ручной разметки")
                continue

            print(f"Обработка: {pdf_path.name}")

            methods_headers = self.collect_headers_for_document(
                pdf_path=pdf_path,
            )

            document_results = self._compare_many_methods(
                document=pdf_path.name,
                methods_headers=methods_headers,
            )

            all_results.extend(document_results)

        if not all_results:
            raise RuntimeError(
                "Нет результатов для сравнения. Проверь data/pdfs и gold_headers.json"
            )

        summary = self._summarize_results(all_results)

        results_df = self.results_to_dataframe(all_results)
        summary_df = self.summary_to_dataframe(summary)

        return results_df, summary_df

    def evaluate_document(
            self,
            pdf_path: str | Path,
    ) -> pd.DataFrame:
        """
        Считает метрики качества извлечения заголовков для одного PDF-документа.

        :param pdf_path: путь к PDF-файлу
        :return: pandas-таблица с результатами по одному документу
        """
        pdf_path = Path(pdf_path)

        methods_headers = self.collect_headers_for_document(
            pdf_path=pdf_path,
        )

        results = self._compare_many_methods(
            document=pdf_path.name,
            methods_headers=methods_headers,
        )

        return self.results_to_dataframe(results)

    def plot_summary_metrics(self, summary_df: pd.DataFrame) -> None:
        """
        Строит график Precision, Recall и F1 по методам.

        :param summary_df: итоговая таблица метрик
        :return: None
        """
        import matplotlib.pyplot as plt

        plot_df = summary_df.copy()
        plot_df = plot_df.sort_values("f1", ascending=False)

        ax = plot_df.set_index("method")[
            ["precision", "recall", "f1"]
        ].plot(
            kind="bar",
            figsize=(11, 6),
            ylim=(0, 1),
            title="Precision / Recall / F1 по методам",
        )

        ax.set_xlabel("Метод")
        ax.set_ylabel("Значение метрики")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_macro_metrics(self, summary_df: pd.DataFrame) -> None:
        """
        Строит график Macro Precision, Macro Recall и Macro F1 по методам.

        :param summary_df: итоговая таблица метрик
        :return: None
        """
        import matplotlib.pyplot as plt

        plot_df = summary_df.copy()
        plot_df = plot_df.sort_values("macro_f1", ascending=False)

        ax = plot_df.set_index("method")[
            ["macro_precision", "macro_recall", "macro_f1"]
        ].plot(
            kind="bar",
            figsize=(11, 6),
            ylim=(0, 1),
            title="Macro Precision / Macro Recall / Macro F1 по методам",
        )

        ax.set_xlabel("Метод")
        ax.set_ylabel("Значение метрики")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_f1_by_method(self, summary_df: pd.DataFrame) -> None:
        """
        Строит отдельный график F1 по методам.

        :param summary_df: итоговая таблица метрик
        :return: None
        """
        import matplotlib.pyplot as plt

        plot_df = summary_df.copy()
        plot_df = plot_df.sort_values("f1", ascending=True)

        ax = plot_df.set_index("method")["f1"].plot(
            kind="barh",
            figsize=(9, 5),
            xlim=(0, 1),
            title="F1 по методам",
        )

        ax.set_xlabel("F1")
        ax.set_ylabel("Метод")
        plt.tight_layout()
        plt.show()

    def plot_errors(self, summary_df: pd.DataFrame) -> None:
        """
        Строит график TP, FP и FN по методам.

        :param summary_df: Итоговая таблица метрик
        :return: None
        """
        import matplotlib.pyplot as plt

        plot_df = summary_df.copy()
        plot_df = plot_df.sort_values("f1", ascending=False)

        ax = plot_df.set_index("method")[
            ["TP", "FP", "FN"]
        ].plot(
            kind="bar",
            figsize=(11, 6),
            title="TP / FP / FN по методам",
        )

        ax.set_xlabel("Метод")
        ax.set_ylabel("Количество заголовков")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_document_f1(self, results_df: pd.DataFrame) -> None:
        """
        Строит график F1 по каждому документу и методу.

        :param results_df: подробная таблица результатов
        :return: None
        """
        import matplotlib.pyplot as plt

        pivot_df = results_df.pivot(
            index="document",
            columns="method",
            values="f1",
        )

        ax = pivot_df.plot(
            kind="bar",
            figsize=(15, 7),
            ylim=(0, 1),
            title="F1 по документам",
        )

        ax.set_xlabel("Документ")
        ax.set_ylabel("F1")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_missing_extra(self, results_df: pd.DataFrame) -> None:
        """
        Строит график количества пропущенных и лишних заголовков по документам.

        :param results_df: подробная таблица результатов
        :return: None
        """
        import matplotlib.pyplot as plt

        plot_df = results_df.copy()

        plot_df["missing_count"] = plot_df["missing"].apply(len)
        plot_df["extra_count"] = plot_df["extra"].apply(len)

        grouped_df = plot_df.groupby("method")[
            ["missing_count", "extra_count"]
        ].sum()

        ax = grouped_df.plot(
            kind="bar",
            figsize=(11, 6),
            title="Пропущенные и лишние заголовки по методам",
        )

        ax.set_xlabel("Метод")
        ax.set_ylabel("Количество")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_all(
            self,
            results_df: pd.DataFrame,
            summary_df: pd.DataFrame,
    ) -> None:
        """
        Строит все основные графики.

        :param results_df: подробная таблица результатов
        :param summary_df: итоговая таблица метрик
        :return: None
        """
        self.plot_summary_metrics(summary_df)
        self.plot_macro_metrics(summary_df)
        self.plot_f1_by_method(summary_df)
        self.plot_errors(summary_df)
        self.plot_document_f1(results_df)
        self.plot_missing_extra(results_df)
