# backend/app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/db"
    REDIS_URL: str = "redis://localhost:6379"
    OPENAI_API_KEY: str = ""
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = ""
    MODEL_PATH: str = "models/yolov8n.pt"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()