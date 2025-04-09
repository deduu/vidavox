from sqlalchemy import (
    Column, String, Text, DateTime, Integer, LargeBinary,
    JSON, ForeignKey, func, Index
)
from sqlalchemy import func
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from vidavox.document_store.models.base import Base

class FaissVector(Base):
    __tablename__ = 'faiss_vectors'
    doc_id = Column(String(255), ForeignKey('documents.doc_id', ondelete="CASCADE"), primary_key=True)
    vector = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=func.now())

    document = relationship("Document", back_populates="faiss_vector") 