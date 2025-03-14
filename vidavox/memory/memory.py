from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo


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

    async def add_user(self, phone_number: str, username: str=None) -> Tuple[Dict[str, Any], bool]:
        """
        Creates a user if doesn't exist. Returns (user_dict, is_new).
        """
        from vidavox.memory.models.user import User

        async with self.memory.async_session() as session:
            # Check if user already exists
            result = await session.execute(
                select(User).where(User.phone_number == phone_number)
            )
            user = result.scalars().first()

            if user:
                # Return existing user as dict + is_new=False
                return {
                    "id": user.id,
                    "phone_number": user.phone_number,
                    "username": user.username,
                    "disclaimer_sent": user.disclaimer_sent
                }, False

            # Otherwise, create new
            new_user = User(phone_number=phone_number, username=username)
            session.add(new_user)
            await session.commit()

            # Reload or return known fields
            await session.refresh(new_user)  # refresh from DB if needed
            return {
                "id": new_user.id,
                "phone_number": new_user.phone_number,
                "username": new_user.username,
                "disclaimer_sent": new_user.disclaimer_sent
            }, True

    
    async def get_user_data(self, username: str) -> dict:
        """
        Retrieves user data from the database and ensures it's returned as a dictionary.
        """
        from vidavox.memory.models.user import User  # Avoid circular dependency

        async with self.memory.async_session() as session:
            result = await session.execute(select(User).filter_by(username=username))
            user = result.scalars().first()

            if user:
                # Return a sanitized dict (filter out SQLAlchemy overhead fields)
                user_dict = {
                    "id": user.id,
                    "phone_number": user.phone_number,
                    "username": user.username,
                    "disclaimer_sent": user.disclaimer_sent,
                    # Add other fields as needed
                }
                return user_dict
            else:
                return {}  # ✅ Ensure it always returns a dictionary

    
    async def update_user_data(self, phone_number: str, data: Dict[str, Any]) -> None:
        """
        Updates user data in the DB by phone_number with the given dict.
        Raises ValueError if the user does not exist.
        """
        from vidavox.memory.models.user import User

        async with self.memory.async_session() as session:
            result = await session.execute(
                select(User).where(User.phone_number == phone_number)
            )
            user = result.scalars().first()

            if not user:
                raise ValueError(f"User with phone_number '{phone_number}' not found.")

            # Update each field
            for key, value in data.items():
                setattr(user, key, value)

            # Because user is already in session, just commit
            await session.commit()
