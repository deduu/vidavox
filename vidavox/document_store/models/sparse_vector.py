
from sqlalchemy import (
    Column, String, Text, DateTime, Integer, LargeBinary,
    JSON, ForeignKey, func, Index
)
from sqlalchemy import func
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from vidavox.document_store.models.base import Base

class BM25Term(Base):
    __tablename__ = 'bm25_terms'
    doc_id = Column(String(255), ForeignKey('documents.doc_id', ondelete="CASCADE"), primary_key=True)
    term = Column(String(255), primary_key=True)
    frequency = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="bm25_terms")

# Create an index for BM25 term
Index('idx_bm25_terms_term', BM25Term.term)