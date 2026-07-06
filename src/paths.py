from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MINERU_OUTPUT_DIR = DATA_DIR / "mineru_output"
PDFS_DIR = DATA_DIR / "pdfs"

print(MINERU_OUTPUT_DIR)