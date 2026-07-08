import json
from pathlib import Path
from src.paths import MINERU_OUTPUT_DIR
from src.text_processing.pdf_extracting.mineru import MineruLoader

class MineruReader:
    def __init__(self):
        self.loader = MineruLoader()

    def get_mineru_doc(self, pdf_path: str):
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
            self.loader.run_mineru(str(pdf_path))
        if not content_json_path.exists():
            raise FileNotFoundError(
                f"MinerU отработал, но файл не найден: {content_json_path}"
            )
        with open(content_json_path, "r", encoding="utf-8") as f:
            return json.load(f)