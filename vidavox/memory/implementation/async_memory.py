# implementations/async_memory.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from vidavox.settings import DatabaseSettings, MemorySettings
from vidavox.memory.memory import ConversationMemoryInterface
from vidavox.utils.token_counter import SimpleTokenCounter, TikTokenCounter
from vidavox.memory.models.base import Base
from vidavox.memory.models.message import Message
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.future import select

class AsyncPostgresConversationMemory(ConversationMemoryInterface):
    def __init__(self, db_settings: DatabaseSettings, memory_settings: MemorySettings):
        self.engine = create_async_engine(
            db_settings.url,
            pool_size=db_settings.pool_size,
            max_overflow=db_settings.max_overflow,
            pool_timeout=db_settings.pool_timeout
        )

        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self.token_limit = memory_settings.token_limit
        
        
        if memory_settings.token_counter == "tiktoken":
            self.token_counter = TikTokenCounter(memory_settings.model_name)
        else:
            self.token_counter = SimpleTokenCounter()

    async def initialize(self):
        """Initialize the database by creating all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    # In your async_memory.py
    async def add_message(self, user_id: int, role: str, message: str, timestamp: Optional[datetime] = None) -> None:
        async with self.async_session() as session:
            if timestamp is None:
                timestamp = datetime.utcnow()
            msg = Message(user_id=user_id, role=role, message=message, timestamp=timestamp)
            session.add(msg)
            await session.commit()
            await self.trim_memory_if_needed(session)


    async def get_all_history(self) -> List[Dict]:
        async with self.async_session() as session:
            result = await session.execute(
                select(Message).order_by(Message.timestamp)
            )
            messages = result.scalars().all()
            return [{"role": msg.role, "content": msg.message} for msg in messages]
    
    async def get_history(
    self, 
    user_id: Optional[int] = None, 
    token_limit: Optional[int] = None, 
    last_n: Optional[int] = None
) -> List[Dict]:
        async with self.async_session() as session:
            query = select(Message).order_by(Message.timestamp)
            if user_id is not None:
                query = query.filter(Message.user_id == user_id)
            result = await session.execute(query)
            messages = result.scalars().all()

        # Accumulate messages in reverse (latest first)
        selected = []
        total_tokens = 0
        for msg in reversed(messages):
            tokens = self.token_counter.count_tokens(msg.message)
            # If token_limit is specified and no message has been added yet,
            # force-add the last message even if it exceeds token_limit.
            if token_limit is not None and len(selected) == 0 and tokens > token_limit:
                selected.append(msg)
                total_tokens = tokens
                continue
            # Otherwise, check if adding this message would exceed the token limit.
            if token_limit is not None and total_tokens + tokens > token_limit:
                break
            selected.append(msg)
            total_tokens += tokens
            # Stop if we've reached the maximum number of messages.
            if last_n is not None and len(selected) >= last_n:
                break

        # Reverse again to return messages in chronological order.
        selected.reverse()
        return [{"role": msg.role, "content": msg.message} for msg in selected]




    async def clear_memory(self) -> None:
        async with self.async_session() as session:
            await session.execute(select(Message).delete())
            await session.commit()

    async def get_total_tokens(self) -> int:
        async with self.async_session() as session:
            result = await session.execute(select(Message))
            messages = result.scalars().all()
            return sum(self.token_counter.count_tokens(msg.message) for msg in messages)

    async def trim_memory_if_needed(self, session: AsyncSession) -> None:
        result = await session.execute(select(Message).order_by(Message.timestamp))
        messages = result.scalars().all()
        total_tokens = sum(self.token_counter.count_tokens(msg.message) for msg in messages)
        
        while total_tokens > self.token_limit and messages:
            oldest = messages.pop(0)
            total_tokens -= self.token_counter.count_tokens(oldest.message)
            await session.delete(oldest)
        
        await session.commit()