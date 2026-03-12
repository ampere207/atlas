from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(slots=True)
class ParsedDocument:
    text: str
    filename: str
    content_type: str


class DocumentParser:
    async def parse(self, raw_bytes: bytes, filename: str, content_type: str) -> ParsedDocument:
        lowered = filename.lower()

        if lowered.endswith(".pdf") or content_type == "application/pdf":
            text = self._parse_pdf(raw_bytes)
        elif lowered.endswith((".md", ".markdown")):
            text = raw_bytes.decode("utf-8", errors="ignore")
        elif lowered.endswith((".txt", ".rst", ".log", ".py", ".json", ".yaml", ".yml")):
            text = raw_bytes.decode("utf-8", errors="ignore")
        else:
            text = raw_bytes.decode("utf-8", errors="ignore")

        return ParsedDocument(text=text, filename=filename, content_type=content_type)

    def _parse_pdf(self, raw_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(raw_bytes))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
