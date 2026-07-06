from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from src.paths import PROJECT_ROOT
from src.text_processing.pdf_extractor import PDFExtractor
from src.text_processing.headers_preprocessor import HeadersProcessor
from src.text_processing.compare_result import HeadersComparator


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

        self.extractor = PDFExtractor()
        self.comparator = HeadersComparator(
            gold_path=self.gold_path,
            threshold=self.threshold,
        )

    @staticmethod
    def header_pretends_to_texts(headers) -> list[str]:
        """
        Преобразует список HeaderPretend в список строк.

        :param headers: список объектов HeaderPretend
        :return: список текстов заголовков
        """
        return [
            header.content
            for header in headers
            if header.content
        ]

    @staticmethod
    def load_gold_documents(gold_path: str | Path) -> set[str]:
        """
        Загружает имена документов, для которых есть ручная разметка.

        :param gold_path: путь к JSON-файлу с ручной разметкой
        :return: множество имён PDF-файлов
        """
        gold_path = Path(gold_path)

        with open(gold_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            item["document"]
            for item in data
        }

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

        mineru_pretends = self.extractor.get_headers_by_mineru(str(pdf_path))
        mupdf_pretends = self.extractor.get_headers_by_mupdf(str(pdf_path))

        mineru_raw = self.header_pretends_to_texts(mineru_pretends)
        mupdf_raw = self.header_pretends_to_texts(mupdf_pretends)

        mineru_clean = HeadersProcessor.clear_headers(mineru_raw)
        mupdf_clean = HeadersProcessor.clear_headers(mupdf_raw)

        mixed_clean = HeadersProcessor.deduplicate_headers(
            mineru_clean + mupdf_clean
        )

        return {
            "MinerU raw": mineru_raw,
            "MinerU + rules": mineru_clean,
            "MuPDF raw": mupdf_raw,
            "MuPDF + rules": mupdf_clean,
            "mixed_clean": mixed_clean,
        }

    @staticmethod
    def results_to_dataframe(results) -> pd.DataFrame:
        """
        Преобразует список HeadersCompareResult в pandas DataFrame.

        :param results: список результатов сравнения
        :return: таблица с результатами по документам и методам
        """
        rows = []

        for result in results:
            rows.append({
                "document": result.document,
                "method": result.method,
                "TP": result.true_positive,
                "FP": result.false_positive,
                "FN": result.false_negative,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "missing": result.missing,
                "extra": result.extra,
                "matches": result.matches,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def summary_to_dataframe(summary: dict[str, dict]) -> pd.DataFrame:
        """
        Преобразует итоговую статистику в pandas DataFrame.

        :param summary: словарь с агрегированными метриками
        :return: таблица с итоговыми метриками по методам
        """
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
        gold_documents = self.load_gold_documents(self.gold_path)
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

            document_results = self.comparator.compare_many_methods(
                document=pdf_path.name,
                methods_headers=methods_headers,
            )

            all_results.extend(document_results)

        if not all_results:
            raise RuntimeError(
                "Нет результатов для сравнения. Проверь data/pdfs и gold_headers.json"
            )

        summary = HeadersComparator.summarize_results(all_results)

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

        results = self.comparator.compare_many_methods(
            document=pdf_path.name,
            methods_headers=methods_headers,
        )

        return self.results_to_dataframe(results)

    @staticmethod
    def plot_summary_metrics(summary_df: pd.DataFrame) -> None:
        """
        Строит график Precision, Recall и F1 по методам.

        :param summary_df: итоговая таблица метрик
        :return: None
        """
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

    @staticmethod
    def plot_macro_metrics(summary_df: pd.DataFrame) -> None:
        """
        Строит график Macro Precision, Macro Recall и Macro F1 по методам.

        :param summary_df: итоговая таблица метрик
        :return: None
        """
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

    @staticmethod
    def plot_f1_by_method(summary_df: pd.DataFrame) -> None:
        """
        Строит отдельный график F1 по методам.

        :param summary_df: итоговая таблица метрик
        :return: None
        """
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

    @staticmethod
    def plot_errors(summary_df: pd.DataFrame) -> None:
        """
        Строит график TP, FP и FN по методам.

        :param summary_df: итоговая таблица метрик
        :return: None
        """
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

    @staticmethod
    def plot_document_f1(results_df: pd.DataFrame) -> None:
        """
        Строит график F1 по каждому документу и методу.

        :param results_df: подробная таблица результатов
        :return: None
        """
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

    @staticmethod
    def plot_missing_extra(results_df: pd.DataFrame) -> None:
        """
        Строит график количества пропущенных и лишних заголовков по документам.

        :param results_df: подробная таблица результатов
        :return: None
        """
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

    @staticmethod
    def plot_all(
            results_df: pd.DataFrame,
            summary_df: pd.DataFrame,
    ) -> None:
        """
        Строит все основные графики.

        :param results_df: подробная таблица результатов
        :param summary_df: итоговая таблица метрик
        :return: None
        """
        HeadersEvaluator.plot_summary_metrics(summary_df)
        HeadersEvaluator.plot_macro_metrics(summary_df)
        HeadersEvaluator.plot_f1_by_method(summary_df)
        HeadersEvaluator.plot_errors(summary_df)
        HeadersEvaluator.plot_document_f1(results_df)
        HeadersEvaluator.plot_missing_extra(results_df)