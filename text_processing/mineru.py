import subprocess
from pathlib import Path
from langchain_core.documents import Document

def run_mineru(pdf_path, output_dir = "output"):
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        if pdf_path.is_dir():
            pdfs = sorted(pdf_path.glob("*.pdf"))
        else:
            pdfs = [pdf_path]
    elif isinstance(pdf_path, (tuple, list)):
        pdfs = pdf_path
    else:
        raise TypeError("Incorrect type of pdf_path")
    for pdf in pdfs:
        subprocess.run(
            [
                "mineru",
                "-p", str(pdf),
                "-o", str(output_dir)
            ],
            check=True
        )


