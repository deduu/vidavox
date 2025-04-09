# vidavox/document_store/models/document.py
from sqlalchemy import (
    Column, String, Text, DateTime, Integer, LargeBinary,
     ForeignKey, func, Index
)
from sqlalchemy import func
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from zoneinfo import ZoneInfo
from sqlalchemy.dialects.postgresql import JSONB
from vidavox.document_store.models.base import Base

class Document(Base):
    __tablename__ = 'documents'
    doc_id = Column(String(255), primary_key=True)
    text = Column(Text, nullable=False)
    doc_metadata = Column(JSONB)  # Use a non-reserved name
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    faiss_vector = relationship("FaissVector", uselist=False, back_populates="document", cascade="all, delete-orphan")
    token_count = relationship("TokenCount", uselist=False, back_populates="document", cascade="all, delete-orphan")
    bm25_terms = relationship("BM25Term", back_populates="document", cascade="all, delete-orphan")


# Create a GIN index for metadata (PostgreSQL specific)
Index('idx_documents_metadata', Document.doc_metadata, postgresql_using='gin')


class TokenCount(Base):
    __tablename__ = 'token_counts'
    doc_id = Column(String(255), ForeignKey('documents.doc_id', ondelete="CASCADE"), primary_key=True)
    token_count = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="token_count")

class EngineMetadata(Base):
    __tablename__ = 'engine_metadata'
    key = Column(String(255), primary_key=True)
    value = Column(JSONB, nullable=False)