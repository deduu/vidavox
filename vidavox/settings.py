

import os
from dotenv import load_dotenv

# load_dotenv()

env_file_path = os.path.join(os.getcwd(), ".env")  
load_dotenv(env_file_path)

from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseSettings:
    url: Optional[str] = "postgresql+asyncpg://postgres:admin@localhost:5432/coretax_wa"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

@dataclass
class MemorySettings:
    token_limit: int = 4096
    token_counter: str = "simple"  # "simple" or "tiktoken"
    model_name: Optional[str] = None  #

    
class Settings:
    POSTGRES_DB_URL: str = os.getenv("POSTGRES_DB_URL")
    SQLITE_DB_URL: str = os.getenv("SQLITE_DB_URL")
    # Add other settings as needed

settings = Settings()




