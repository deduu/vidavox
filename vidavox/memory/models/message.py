# models/message.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(50))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Use a string reference for deferred resolution.
    user = relationship("User", back_populates="messages")
