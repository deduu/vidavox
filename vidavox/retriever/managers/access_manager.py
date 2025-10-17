from typing import List, Optional
from vidavox.retriever.managers.doc_manager import DocumentManager


class AccessControl:
    """Manages multi-tenant access control."""

    def __init__(self, doc_manager: DocumentManager):
        self.doc_manager = doc_manager

    def get_allowed_doc_ids(self, user_id: Optional[str]) -> Optional[List[str]]:
        """
        Get list of document IDs accessible to user.

        Returns None if user has admin access (no filtering).
        """
        if user_id is None:
            return None  # Admin access
        return self.doc_manager.get_user_docs(user_id)

    def can_access_document(self, doc_id: str, user_id: Optional[str]) -> bool:
        """Check if user can access a specific document."""
        if user_id is None:
            return True  # Admin access

        allowed = self.get_allowed_doc_ids(user_id)
        return allowed is not None and doc_id in allowed

    def filter_doc_ids(
        self,
        doc_ids: List[str],
        user_id: Optional[str],
    ) -> List[str]:
        """Filter document IDs based on user access."""
        if user_id is None:
            return doc_ids

        allowed = set(self.get_allowed_doc_ids(user_id) or [])
        return [doc_id for doc_id in doc_ids if doc_id in allowed]
