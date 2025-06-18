from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import threading, re

# matches “…_chunk0”, “…_chunk17”, etc
_CHUNK_SUFFIX_RE = re.compile(r"_chunk\d+$")

@dataclass
class Doc:
    doc_id: str
    text: str
    meta_data: Dict[str, any]
    folder_ids: Set[str] = field(default_factory=set)

class DocumentManager:
    """
    Single-engine, multi-tenant document registry with shared documents and optional folders.
    """
    def __init__(self) -> None:
        self.documents: Dict[str, Doc] = {}
        self.user_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        self.shared_doc_ids: Set[str] = set()
        self.folder_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.RLock()
        self._just_added: Dict[Optional[str], List[str]] = defaultdict(list)

    def unique_parent_ids(
        self, user_id: Optional[str] = None
    ) -> List[str]:
        raw_ids = self.doc_ids(user_id)
        parents: Set[str] = set()
        for cid in raw_ids:
            parent = _CHUNK_SUFFIX_RE.sub("", cid)
            parents.add(parent)
        return sorted(parents)

    # ---------- INGEST ----------
    def add_document(
        self,
        doc_id: str,
        text: str,
        meta_data: Optional[Dict] = None,
        user_id: Optional[str] = None,
        folder_id: Optional[str] = None,
    ) -> None:
        meta = (meta_data or {}).copy()
        if user_id is not None:
            meta["owner_id"] = user_id
        with self.lock:
            doc = Doc(doc_id, text, meta)
            if folder_id:
                doc.folder_ids.add(folder_id)
                self.folder_to_doc_ids[folder_id].add(doc_id)
            self.documents[doc_id] = doc
            if user_id is not None:
                self.user_to_doc_ids[user_id].add(doc_id)
            else:
                self.shared_doc_ids.add(doc_id)
            self._just_added[user_id].append(doc_id)

    def add_documents(
        self,
        docs: List[Union[Tuple[str, str, Dict], Tuple[str, str, Dict, Optional[str]]]],
        user_id: Optional[str] = None,
    ) -> None:
        with self.lock:
            new_docs: Dict[str, Doc] = {}
            newly_added: List[str] = []
            for item in docs:
                if len(item) == 3:
                    doc_id, text, meta_data = item
                    folder_id: Optional[str] = None
                else:
                    doc_id, text, meta_data, folder_id = item  # type: ignore
                meta = (meta_data or {}).copy()
                if user_id is not None:
                    meta["owner_id"] = user_id
                doc = Doc(doc_id, text, meta)
                if folder_id:
                    doc.folder_ids.add(folder_id)
                    self.folder_to_doc_ids[folder_id].add(doc_id)
                new_docs[doc_id] = doc
                newly_added.append(doc_id)
            self.documents.update(new_docs)
            if user_id is not None:
                self.user_to_doc_ids[user_id].update(newly_added)
            else:
                self.shared_doc_ids.update(newly_added)
            self._just_added[user_id].extend(newly_added)

    # ---------- READ ----------
    def get_document(self, user_id: str, doc_id: str) -> Optional[str]:
        with self.lock:
            if doc_id not in self.user_to_doc_ids[user_id] and doc_id not in self.shared_doc_ids:
                return None
            return self.documents[doc_id].text

    def get_metadata(self, user_id: str, doc_id: str) -> Optional[Dict]:
        with self.lock:
            if doc_id not in self.user_to_doc_ids[user_id] and doc_id not in self.shared_doc_ids:
                return None
            return self.documents[doc_id].meta_data

    def get_user_docs(self, user_id: str) -> List[str]:
        with self.lock:
            user_docs = set(self.user_to_doc_ids.get(user_id, set()))
            return list(user_docs | self.shared_doc_ids)

    def get_documents(
        self,
        user_id: Optional[str] = None,
        folder_id: Optional[str] = None,
    ) -> List[Doc]:
        with self.lock:
            base = set(self.user_to_doc_ids.get(user_id, set())) | self.shared_doc_ids
            if folder_id:
                base &= self.folder_to_doc_ids.get(folder_id, set())
            return [self.documents[d] for d in sorted(base)]

    def get_all_documents(self) -> Dict[str, str]:
        with self.lock:
            return {doc_id: doc.text for doc_id, doc in self.documents.items()}

    def get_all_just_added_documents(
        self,
        user_id: Optional[str],
        clear_after: bool = True
    ) -> Dict[str, str]:
        with self.lock:
            just_ids = list(self._just_added.get(user_id, []))
            result = {doc_id: self.documents[doc_id].text for doc_id in just_ids}
            if clear_after:
                self._just_added[user_id].clear()
            return result

    def get_all_documents_by_user(self) -> Dict[Optional[str], List[Doc]]:
        with self.lock:
            result: Dict[Optional[str], List[Doc]] = {}
            for uid, ids in self.user_to_doc_ids.items():
                result[uid] = [self.documents[d] for d in ids]
            result[None] = [self.documents[d] for d in self.shared_doc_ids]
            return result

    # ---------- DELETE ----------
    def _delete_document_nolock(self, doc_id: str, user_id: Optional[str] = None) -> bool:
        if doc_id not in self.documents:
            return False
        if user_id is not None and doc_id not in self.user_to_doc_ids[user_id] and doc_id not in self.shared_doc_ids:
            return False
        for uid, docs in self.user_to_doc_ids.items():
            docs.discard(doc_id)
        self.shared_doc_ids.discard(doc_id)
        self.documents.pop(doc_id, None)
        return True

    def delete_document(self, doc_id: str, user_id: Optional[str] = None) -> bool:
        with self.lock:
            success = self._delete_document_nolock(doc_id, user_id)
            if success:
                for fid, docs in self.folder_to_doc_ids.items():
                    docs.discard(doc_id)
            return success

    def delete_documents(self, doc_ids: List[str], user_id: Optional[str] = None) -> int:
        deleted = 0
        with self.lock:
            for doc_id in doc_ids:
                if self._delete_document_nolock(doc_id, user_id):
                    deleted += 1
                    for fid, docs in self.folder_to_doc_ids.items():
                        docs.discard(doc_id)
        return deleted

    # ---------- UTIL ----------
    def clear_user(self, user_id: str) -> None:
        with self.lock:
            doc_ids = self.user_to_doc_ids.pop(user_id, set())
            for doc_id in doc_ids:
                used_elsewhere = any(doc_id in docs for docs in self.user_to_doc_ids.values())
                if not used_elsewhere and doc_id not in self.shared_doc_ids:
                    self.documents.pop(doc_id, None)

    def doc_count(self, user_id: Optional[str] = None) -> int:
        with self.lock:
            if user_id is None:
                return len(self.documents)
            return len(self.user_to_doc_ids.get(user_id, set())) + len(self.shared_doc_ids)

    def doc_ids(self, user_id: Optional[str] = None) -> List[str]:
        with self.lock:
            if user_id is None:
                return sorted(self.documents.keys())
            return sorted(set(self.user_to_doc_ids.get(user_id, set())) | self.shared_doc_ids)

    # ---------- FOLDER MANAGEMENT ----------
    def create_folder(self, folder_id: str) -> None:
        with self.lock:
            self.folder_to_doc_ids.setdefault(folder_id, set())

    def delete_folder(self, folder_id: str, delete_docs: bool = False) -> int:
        with self.lock:
            docs = list(self.folder_to_doc_ids.pop(folder_id, []))
            if delete_docs:
                for doc_id in docs:
                    self._delete_document_nolock(doc_id)
                return len(docs)
            for doc_id in docs:
                self.documents[doc_id].folder_ids.discard(folder_id)
            return 0

    def move_document_to_folder(
        self, doc_id: str, new_folder_id: str
    ) -> bool:
        with self.lock:
            if doc_id not in self.documents:
                return False
            doc = self.documents[doc_id]
            for fid in list(doc.folder_ids):
                self.folder_to_doc_ids[fid].discard(doc_id)
            doc.folder_ids = {new_folder_id}
            self.folder_to_doc_ids[new_folder_id].add(doc_id)
            return True
