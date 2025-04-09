import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, delete

from vidavox.settings import DatabaseSettings
from vidavox.document_store.models import Base, Document, FaissVector, BM25Term, TokenCount, EngineMetadata
from zoneinfo import ZoneInfo

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStorePsql:
    """
    Handles persistence of RAG engine state in PostgreSQL using SQLAlchemy ORM (async version).
    """
    def __init__(self, db_settings: DatabaseSettings):
        self.engine = create_async_engine(
            db_settings.url,
            # echo=True,
            pool_size=db_settings.pool_size,
            max_overflow=db_settings.max_overflow,
            pool_timeout=db_settings.pool_timeout
        )
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        

    async def initialize(self):
        """Initialize the database by creating all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def store_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        async with self.async_session() as session:
            try:
                doc = await session.get(Document, doc_id)
                if not doc:
                    doc = Document(doc_id=doc_id, text=text, doc_metadata=metadata)
                else:
                    doc.text = text
                    doc.doc_metadata = metadata
                await session.merge(doc)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store document {doc_id}: {e}")
                return False

    async def store_documents_batch(self, docs: List[Tuple[str, str, Dict[str, Any]]]) -> bool:
        async with self.async_session() as session:
            try:
                for doc_id, text, metadata in docs:
                    doc = await session.get(Document, doc_id)
                    if not doc:
                        doc = Document(doc_id=doc_id, text=text, doc_metadata=metadata)
                    else:
                        doc.text = text
                        doc.doc_metadata = metadata
                    await session.merge(doc)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store documents batch: {e}")
                return False

    async def store_faiss_vector(self, doc_id: str, vector: np.ndarray) -> bool:
        async with self.async_session() as session:
            try:
                vector_bytes = pickle.dumps(vector)
                fv = await session.get(FaissVector, doc_id)
                if not fv:
                    fv = FaissVector(doc_id=doc_id, vector=vector_bytes)
                else:
                    fv.vector = vector_bytes
                    fv.created_at = func.now()
                await session.merge(fv)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store FAISS vector for document {doc_id}: {e}")
                return False

    async def store_faiss_vectors_batch(self, vectors: List[Tuple[str, np.ndarray]]) -> bool:
        async with self.async_session() as session:
            try:
                for doc_id, vector in vectors:
                    vector_bytes = pickle.dumps(vector)
                    fv = await session.get(FaissVector, doc_id)
                    if not fv:
                        fv = FaissVector(doc_id=doc_id, vector=vector_bytes)
                    else:
                        fv.vector = vector_bytes
                        fv.created_at = func.now()
                    await session.merge(fv)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store FAISS vectors batch: {e}")
                return False

    async def store_bm25_terms(self, doc_id: str, terms: Dict[str, int]) -> bool:
        async with self.async_session() as session:
            try:
                # Delete existing terms
                await session.execute(delete(BM25Term).where(BM25Term.doc_id == doc_id))
                # Add new terms
                for term, freq in terms.items():
                    bm25 = BM25Term(doc_id=doc_id, term=term, frequency=freq)
                    session.add(bm25)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store BM25 terms for document {doc_id}: {e}")
                return False

    async def store_token_count(self, doc_id: str, token_count: int) -> bool:
        async with self.async_session() as session:
            try:
                tc = await session.get(TokenCount, doc_id)
                if not tc:
                    tc = TokenCount(doc_id=doc_id, token_count=token_count)
                else:
                    tc.token_count = token_count
                await session.merge(tc)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store token count for document {doc_id}: {e}")
                return False

    async def store_token_counts_batch(self, counts: List[Tuple[str, int]]) -> bool:
        async with self.async_session() as session:
            try:
                for doc_id, token_count in counts:
                    tc = await session.get(TokenCount, doc_id)
                    if not tc:
                        tc = TokenCount(doc_id=doc_id, token_count=token_count)
                    else:
                        tc.token_count = token_count
                    await session.merge(tc)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store token counts batch: {e}")
                return False

    async def store_engine_metadata(self, key: str, value: Dict[str, Any]) -> bool:
        async with self.async_session() as session:
            try:
                meta = await session.get(EngineMetadata, key)
                if not meta:
                    meta = EngineMetadata(key=key, value=value)
                else:
                    meta.value = value
                await session.merge(meta)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store engine metadata {key}: {e}")
                return False

    async def get_document(self, doc_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        async with self.async_session() as session:
            try:
                doc = await session.get(Document, doc_id)
                return (doc.text, doc.doc_metadata) if doc else None
            except Exception as e:
                logger.error(f"Failed to retrieve document {doc_id}: {e}")
                return None

    async def get_all_documents(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        async with self.async_session() as session:
            try:
                result = await session.execute(select(Document))
                docs = result.scalars().all()
                return [(doc.doc_id, doc.text, doc.doc_metadata) for doc in docs]
            except Exception as e:
                logger.error(f"Failed to retrieve all documents: {e}")
                return []

    async def get_documents_by_prefix(self, prefix: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        async with self.async_session() as session:
            try:
                result = await session.execute(select(Document).filter(Document.doc_id.like(f"{prefix}%")))
                docs = result.scalars().all()
                return [(doc.doc_id, doc.text, doc.doc_metadata) for doc in docs]
            except Exception as e:
                logger.error(f"Failed to retrieve documents with prefix {prefix}: {e}")
                return []

    async def get_documents_by_metadata(self, key: str, value: Any) -> List[Tuple[str, str, Dict[str, Any]]]:
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(Document).filter(Document.metadata[key].astext == str(value))
                )
                docs = result.scalars().all()
                return [(doc.doc_id, doc.text, doc.metadata) for doc in docs]
            except Exception as e:
                logger.error(f"Failed to retrieve documents with metadata {key}={value}: {e}")
                return []

    async def get_faiss_vector(self, doc_id: str) -> Optional[np.ndarray]:
        async with self.async_session() as session:
            try:
                fv = await session.get(FaissVector, doc_id)
                if fv and fv.vector:
                    return pickle.loads(fv.vector)
                return None
            except Exception as e:
                logger.error(f"Failed to retrieve FAISS vector for document {doc_id}: {e}")
                return None

    async def get_all_faiss_vectors(self) -> Dict[str, np.ndarray]:
        async with self.async_session() as session:
            try:
                result = await session.execute(select(FaissVector))
                faiss_list = result.scalars().all()
                vectors = {}
                for fv in faiss_list:
                    if fv.vector:
                        vectors[fv.doc_id] = pickle.loads(fv.vector)
                return vectors
            except Exception as e:
                logger.error(f"Failed to retrieve all FAISS vectors: {e}")
                return {}

    async def get_bm25_terms(self, doc_id: str) -> Dict[str, int]:
        async with self.async_session() as session:
            try:
                result = await session.execute(select(BM25Term).filter(BM25Term.doc_id == doc_id))
                terms = result.scalars().all()
                return {term.term: term.frequency for term in terms}
            except Exception as e:
                logger.error(f"Failed to retrieve BM25 terms for document {doc_id}: {e}")
                return {}

    async def get_all_bm25_terms(self) -> Dict[str, Dict[str, int]]:
        async with self.async_session() as session:
            try:
                result = await session.execute(select(BM25Term))
                bm25_list = result.scalars().all()
                terms = {}
                for bm25 in bm25_list:
                    terms.setdefault(bm25.doc_id, {})[bm25.term] = bm25.frequency
                return terms
            except Exception as e:
                logger.error(f"Failed to retrieve all BM25 terms: {e}")
                return {}

    async def get_token_count(self, doc_id: str) -> Optional[int]:
        async with self.async_session() as session:
            try:
                tc = await session.get(TokenCount, doc_id)
                return tc.token_count if tc else None
            except Exception as e:
                logger.error(f"Failed to retrieve token count for document {doc_id}: {e}")
                return None

    async def get_all_token_counts(self) -> Dict[str, int]:
        async with self.async_session() as session:
            try:
                result = await session.execute(select(TokenCount))
                counts = result.scalars().all()
                return {tc.doc_id: tc.token_count for tc in counts}
            except Exception as e:
                logger.error(f"Failed to retrieve all token counts: {e}")
                return {}

    async def get_engine_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        async with self.async_session() as session:
            try:
                meta = await session.get(EngineMetadata, key)
                return meta.value if meta else None
            except Exception as e:
                logger.error(f"Failed to retrieve engine metadata for key {key}: {e}")
                return None

    async def delete_document(self, doc_id: str) -> bool:
        async with self.async_session() as session:
            try:
                doc = await session.get(Document, doc_id)
                if doc:
                    await session.delete(doc)
                    await session.commit()
                    return True
                return False
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete document {doc_id}: {e}")
                return False

    async def delete_documents_by_prefix(self, prefix: str) -> int:
        async with self.async_session() as session:
            try:
                result = await session.execute(select(Document).filter(Document.doc_id.like(f"{prefix}%")))
                docs = result.scalars().all()
                count = len(docs)
                for doc in docs:
                    await session.delete(doc)
                await session.commit()
                return count
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete documents with prefix {prefix}: {e}")
                return 0

    async def delete_all_documents(self) -> bool:
        async with self.async_session() as session:
            try:
                result = await session.execute(delete(Document))
                await session.commit()
                return result.rowcount > 0
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete all documents: {e}")
                return False

    async def export_to_file(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            documents = await self.get_all_documents()
            faiss_vectors = await self.get_all_faiss_vectors()
            bm25_terms = await self.get_all_bm25_terms()
            token_counts = await self.get_all_token_counts()
            backup_data = {
                "documents": documents,
                "faiss_vectors": faiss_vectors,
                "bm25_terms": bm25_terms,
                "token_counts": token_counts
            }
            backup_file = f"{path}_backup.pkl"
            with open(backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            logger.info(f"Database exported successfully to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            return False

    async def import_from_file(self, path: str) -> bool:
        try:
            backup_file = f"{path}_backup.pkl"
            if not os.path.exists(backup_file):
                logger.error(f"Backup file {backup_file} not found")
                return False
            with open(backup_file, 'rb') as f:
                backup_data = pickle.load(f)
            await self.delete_all_documents()
            if backup_data.get("documents"):
                for doc_id, text, metadata in backup_data["documents"]:
                    await self.store_document(doc_id, text, metadata)
                    if backup_data.get("faiss_vectors") and doc_id in backup_data["faiss_vectors"]:
                        await self.store_faiss_vector(doc_id, backup_data["faiss_vectors"][doc_id])
                    if backup_data.get("bm25_terms") and doc_id in backup_data["bm25_terms"]:
                        await self.store_bm25_terms(doc_id, backup_data["bm25_terms"][doc_id])
                    if backup_data.get("token_counts") and doc_id in backup_data["token_counts"]:
                        await self.store_token_count(doc_id, backup_data["token_counts"][doc_id])
            logger.info(f"Database imported successfully from {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to import database: {e}")
            return False
