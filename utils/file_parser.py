# Read lecture text from disk paths or from raw upload bytes (.txt / .pdf).

import io
from pathlib import Path

from pypdf import PdfReader


# Load a UTF-8 text file from disk (CLI or tests); raises if path is not a file.
def read_text_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    return p.read_text(encoding="utf-8", errors="replace").strip()


# Decode uploads: .txt as UTF-8 (replace errors); .pdf via pypdf page text extraction.
def extract_text_from_bytes(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="replace").strip()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts).strip()
    raise ValueError(f"Unsupported file type (use .txt or .pdf): {filename}")
