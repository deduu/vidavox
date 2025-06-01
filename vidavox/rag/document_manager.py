from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import threading

@dataclass
class Doc:
    doc_id: str
    text: str
    meta_data: Dict[str, any]

class DocumentManager:
    """
    Single-engine, multi-tenant document registry with shared documents.
    • All docs live in one dict -> O(1) lookup by id
    • user_to_doc_ids keeps a per-tenant index -> O(1) filtering
    • owner_id is injected into each doc.meta_data for persistence-layer filters
    • Documents without a specific user_id belong to all users (shared)
    """
    def __init__(self) -> None:
        self.documents: Dict[str, Doc] = {}
        self.user_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        self.shared_doc_ids: Set[str] = set()  # Tracks documents that belong to all users
        self.lock = threading.RLock()
        self._just_added: Dict[Optional[str], List[str]] = defaultdict(list)

    # ---------- INGEST ----------
    def add_document(
        self,
        doc_id: str,
        text: str,
        meta_data: Optional[Dict] = None,
        user_id: Optional[str] = None,
    ) -> None:
        meta = (meta_data or {}).copy()
        if user_id is not None:
            meta["owner_id"] = user_id
        
        with self.lock:
            self.documents[doc_id] = Doc(doc_id, text, meta)
            if user_id is not None:
                self.user_to_doc_ids[user_id].add(doc_id)
            else:
                self.shared_doc_ids.add(doc_id)

            self._just_added[user_id].append(doc_id)

    def add_documents(
        self,
        docs: List[Tuple[str, str, Dict]],
        user_id: Optional[str] = None,
    ) -> None:
        with self.lock:
            # Prepare all documents at once
            new_docs = {}
            doc_ids = set()
            newly_added_ids = []
            
            for doc_id, text, meta_data in docs:
                meta = (meta_data or {}).copy()
                if user_id is not None:
                    meta["owner_id"] = user_id
                new_docs[doc_id] = Doc(doc_id, text, meta)
                # doc_ids.add(doc_id)
                newly_added_ids.append(doc_id)
            
            # Batch update documents dictionary
            self.documents.update(new_docs)
            
            # Associate documents with user or mark as shared
            if user_id is not None:
                self.user_to_doc_ids[user_id].update(newly_added_ids)
            else:
                self.shared_doc_ids.update(newly_added_ids)
            
            self._just_added[user_id].extend(newly_added_ids)

    # ---------- READ ----------
    def get_document(self, user_id: str, doc_id: str) -> Optional[str]:
        """
        Return text if this user owns the doc or if it's shared.
        """
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
        """Fast O(#docs_user + #shared_docs) listing for query filters."""
        with self.lock:
            # Return both user-specific and shared documents
            user_docs = set(self.user_to_doc_ids.get(user_id, set()))
            return list(user_docs | self.shared_doc_ids)
    
    def get_all_documents(self) -> List[Doc]:
        """
        Return a list of every Doc in the registry (shared + per‐user).
        """
        with self.lock:
            # .values() is O(n) in number of docs
            return {doc_id: doc.text for doc_id, doc in self.documents.items()}
    
    def get_all_just_added_documents(
        self,
        user_id: Optional[str],
        clear_after: bool = True
    ) -> Dict[str, str]:
        """
        Return a map of doc_id -> text for everything that was
        added *since the last time you called this method* (for this user_id).
        If clear_after=True (default), empties the buffer.
        """
        with self.lock:
            just_ids = list(self._just_added.get(user_id, []))
            result = {doc_id: self.documents[doc_id].text for doc_id in just_ids}
            if clear_after:
                self._just_added[user_id].clear()
            return result

    


    def get_all_documents_by_user(self) -> Dict[Optional[str], List[Doc]]:
        """
        Return a mapping from user_id to that user’s Docs,
        plus key None for shared-only docs.
        """
        with self.lock:
            result: Dict[Optional[str], List[Doc]] = {}

            # per‐user docs
            for uid, doc_ids in self.user_to_doc_ids.items():
                result[uid] = [self.documents[d] for d in doc_ids]

            # shared docs (no owner)
            result[None] = [self.documents[d] for d in self.shared_doc_ids]

            return result

    # ---------- DELETE ----------
    def _delete_document_nolock(self, doc_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a document without locking."""
        # Check document exists
        if doc_id not in self.documents:
            return False
            
        # If user_id provided, check ownership
        if user_id is not None and doc_id not in self.user_to_doc_ids[user_id] and doc_id not in self.shared_doc_ids:
            return False
            
        # Remove from user's collection if it belongs to a user
        for uid, docs in self.user_to_doc_ids.items():
            if doc_id in docs:
                docs.remove(doc_id)
                
        # Remove from shared if it's shared
        self.shared_doc_ids.discard(doc_id)
        
        # Remove from main documents dictionary
        self.documents.pop(doc_id, None)
        return True
    
    def delete_document(self, doc_id: str, user_id: Optional[str] = None) -> bool:
        with self.lock:
            return self._delete_document_nolock(doc_id, user_id)

    def delete_documents(self, doc_ids: List[str], user_id: Optional[str] = None) -> int:
        """Batch delete multiple documents, returns count of deleted docs"""
        deleted_count = 0
        with self.lock:
            for doc_id in doc_ids:
                if self._delete_document_nolock(doc_id, user_id):
                    deleted_count += 1
            return deleted_count

    # ---------- UTIL ----------
    def clear_user(self, user_id: str) -> None:
        with self.lock:
            # Get user's document IDs
            doc_ids = self.user_to_doc_ids.pop(user_id, set())
            
            # Delete documents that belong only to this user
            # (not shared and not belonging to other users)
            for doc_id in doc_ids:
                is_used_elsewhere = any(doc_id in docs for uid, docs in self.user_to_doc_ids.items())
                if not is_used_elsewhere and doc_id not in self.shared_doc_ids:
                    self.documents.pop(doc_id, None)

    def doc_count(self, user_id: Optional[str] = None) -> int:
        with self.lock:
            if user_id is None:
                return len(self.documents)
            # Return count of user's documents plus shared documents
            return len(self.user_to_doc_ids.get(user_id, set())) + len(self.shared_doc_ids)

    def doc_ids(self, user_id: Optional[str] = None) -> List[str]:
        with self.lock:
            if user_id is None:
                return sorted(self.documents.keys())
            # Return both user-specific and shared documents
            return sorted(set(self.user_to_doc_ids.get(user_id, set())) | self.shared_doc_ids)
