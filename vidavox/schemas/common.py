from dataclasses import dataclass, asdict

@dataclass
class DocItem:
    path: str
    doc_id: str | None = None   
    url: str | None = None        
    folder_id: str | None = None 