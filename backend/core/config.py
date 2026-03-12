from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-02-01"
    azure_embed_deployment: str = "text-embedding-3-small"
    azure_embed_dimensions: int = 1536

    azure_chat_deployment: str = "gpt-4o"

    # Database — individual fields, assembled into DSN at runtime
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "horo"
    db_user: str = "postgres"
    db_password: str
    db_pool_min: int = 2
    db_pool_max: int = 10
    db_ssl: str = "disable"       # disable | require | verify-full

    @property
    def database_dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # RAG
    chunk_size: int = 400
    chunk_overlap: int = 80
    sim_threshold: float = 0.72
    top_k: int = 6

    # App
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()