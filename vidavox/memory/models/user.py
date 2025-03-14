# models/user.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from zoneinfo import ZoneInfo
from .base import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=True)
    phone_number = Column(String(15), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("Asia/Jakarta")))
    disclaimer_sent = Column(Boolean, default=False)
    
    # Relationship to Message
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
