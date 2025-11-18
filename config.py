"""Central configuration for Parenting-Child-Psychology-RAG-Pipeline using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    # Secrets / env variables
    #SECRET_KEY: str
    OPENROUTER_API_KEY: str  # This will be loaded from your .env
    #ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    MONGO_URL: str = "mongodb://localhost:27017/tutore_dev"

    # Directories / files
    ROOT_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = Path(__file__).parent / "data"
    CSV_PATH: Path = Path(__file__).parent / "parenting_articles.csv"
    CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_db")))
    CHROMA_COLLECTION: str = "parenting_articles"

    # Embedding model (HF SentenceTransformer)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_DOCS: int | None = None

    # Text split parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    # LLM model
    LLM_MODEL: str = "openrouter/auto"

    # Retrieval
    DEFAULT_TOP_K: int = 3

    # Prompt template
    PROMPT_TEMPLATE: str = (
        "You are a parenting assistant expert. Use ONLY the provided sources to answer the question.\n"
        "If uncertain or information not found, say so clearly.\n\n"
        "Question: {question}\n\nSources:\n{sources}\n\nAnswer:"
    )

    # Timeouts
    HTTP_TIMEOUT: int = 60

    # Pydantic config to load .env
    model_config = SettingsConfigDict(env_file=".env")


# Instantiate settings
settings = Settings()
