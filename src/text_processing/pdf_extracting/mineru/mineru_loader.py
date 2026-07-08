import subprocess
from pathlib import Path

from src.paths import MINERU_OUTPUT_DIR


class MineruLoader:
    def run_mineru(self, pdf_path: str | Path | list[str | Path] | tuple[str | Path]) -> None:
        """
        Запускает MinerU для одного PDF-файла, папки с PDF или списка PDF-файлов.
        :param pdf_path: путь к PDF-файлу, папке с PDF или список путей
        :return: None
        """
        output_dir = Path(MINERU_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(pdf_path, (str, Path)):
            pdf_path = Path(pdf_path)

            if pdf_path.is_dir():
                pdfs = sorted(pdf_path.glob("*.pdf"))
            else:
                pdfs = [pdf_path]

        elif isinstance(pdf_path, (tuple, list)):
            pdfs = [Path(path) for path in pdf_path]

        else:
            raise TypeError("Incorrect type of pdf_path")

        if not pdfs:
            raise FileNotFoundError(f"PDF-файлы не найдены: {pdf_path}")

        for pdf in pdfs:
            command = [
                "mineru",
                "-p", str(pdf),
                "-o", str(output_dir),

                "-m", "auto",
                "-b", "hybrid-engine",
                "--effort", "medium",
                "-l", "cyrillic",

                "-f", "true",
                "-t", "true",
                "--image-analysis", "false",
            ]

            print("Запуск MinerU:")
            print(" ".join(command))

            subprocess.run(
                command,
                check=True,
            )