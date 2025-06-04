from dataclasses import dataclass, asdict

@dataclass
class DocItem:
    path: str          # full file system path
    doc_id: str | None  # id of the document
    url: str | None    # URL to the file