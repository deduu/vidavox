from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime


class ConversationMemoryInterface(ABC):
    @abstractmethod
    def add_message(self, role: str, message: str, timestamp: Optional[datetime] = None) -> None:
        pass

    @abstractmethod
    def get_history(self) -> List[Dict]:
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        pass

    @abstractmethod
    def get_total_tokens(self) -> int:
        pass


from vidavox.settings import DatabaseSettings, MemorySettings, settings
from vidavox.memory.implementation.async_memory import AsyncPostgresConversationMemory
from datetime import datetime
from sqlalchemy.future import select

class AgentMemory:
    def __init__(
        self,
        db_url: str = None,
        token_limit: int = 500,
        token_counter: str = "simple",  # or "tiktoken"
        model_name: str = None  # required if token_counter == "tiktoken"
    ):
        # Use provided URL or default from settings
        if db_url is None:
            db_url = settings.POSTGRES_DB_URL
        self.db_settings = DatabaseSettings(url=db_url)
        self.memory_settings = MemorySettings(
            token_limit=token_limit,
            token_counter=token_counter,
            model_name=model_name
        )
        # Instantiate your async memory
        self.memory = AsyncPostgresConversationMemory(self.db_settings, self.memory_settings)

    async def initialize(self):
        """Initializes the database tables and returns the memory instance."""
        await self.memory.initialize()
        return self.memory

    async def add_user(self, username: str, hashed_password: str):
        """
        Adds a new user to the database.
        Returns the created user or existing user if found.
        """
        from vidavox.memory.models.user import User  # Import here to avoid circular dependencies
        async with self.memory.async_session() as session:
            result = await session.execute(select(User).filter_by(username=username))
            existing_user = result.scalars().first()
            if existing_user:
                return existing_user

            new_user = User(
                username=username,
                hashed_password=hashed_password,
                created_at=datetime.utcnow()
            )
            session.add(new_user)
            await session.commit()
            return new_user

