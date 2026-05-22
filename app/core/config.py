"""
Centralized settings. Reads from environment variables (loaded from .env by
docker-compose). Keep secrets out of source.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-5-20250929"

    # Postgres
    postgres_user: str = "ragapp"
    postgres_password: str = "changeme"
    postgres_db: str = "ragdb"
    postgres_host: str = "db"
    postgres_port: int = 5432

    # App
    log_level: str = "INFO"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    top_k_retrieval: int = 5

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
