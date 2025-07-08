from typing import Literal
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    ENV: Literal["LOCAL", "DEV", "CI", "QAS", "PROD"] = "DEV"

    OPENAI_API_KEY: str = ""
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.7
    SIMILARITY_TOP_K: int = 10
    SALER_AGENT_PROMPT_PATH: str = "prompts/saler.txt"
    ORCHESTRATOR_PROMPT_PATH: str = "prompts/orchestrator.txt"

    QDRANT_LINK: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "products_promotions"


settings = Settings()