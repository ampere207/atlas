from dataclasses import dataclass


@dataclass(slots=True)
class ParsedDocument:
    text: str
    filename: str
    content_type: str


class DocumentParser:
    async def parse(self, raw_bytes: bytes, filename: str, content_type: str) -> ParsedDocument:
        text = raw_bytes.decode("utf-8", errors="ignore")
        return ParsedDocument(text=text, filename=filename, content_type=content_type)
