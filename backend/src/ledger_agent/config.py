"""Application settings loaded from environment variables and the root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/ledger_agent/config.py -> parents[3] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SEC EDGAR
    sec_user_agent: str = ""

    # LLM providers
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Paths and logging
    data_dir: Path = REPO_ROOT / "data"
    log_level: str = "INFO"

    @field_validator("data_dir", mode="before")
    @classmethod
    def _empty_data_dir_uses_default(cls, value: object) -> object:
        # An empty DATA_DIR in .env falls back to <repo>/data
        if value in (None, ""):
            return REPO_ROOT / "data"
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
