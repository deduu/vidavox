from dataclasses import dataclass, asdict

@dataclass
class DocItem:
    path: str          # full file system path
    db_id: str | None  # UUID we got back from upload_files