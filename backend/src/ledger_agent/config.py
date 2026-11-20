"""Paramètres de l'application, lus depuis les variables d'environnement et le .env racine."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/ledger_agent/config.py -> parents[3] est la racine du dépôt en installation éditable ;
# LEDGER_AGENT_ROOT permet de la fixer ailleurs (image Docker, paquet installé)
REPO_ROOT = Path(os.environ.get("LEDGER_AGENT_ROOT") or Path(__file__).resolve().parents[3])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SEC EDGAR
    sec_user_agent: str = ""

    # Fournisseurs de LLM
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # Observabilité
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Chemins et journalisation
    data_dir: Path = REPO_ROOT / "data"
    log_level: str = "INFO"

    # Réseau : certaines routes IPv6 vers les fournisseurs de LLM sont cassées ;
    # FORCE_IPV4=true fait passer les appels LLM par IPv4
    force_ipv4: bool = False

    # API : configuration d'agent servie (un nom de evaluation/configs) et origines autorisées
    api_config: str = "agent"
    cors_origins: str = "http://localhost:3000"

    @field_validator("data_dir", mode="before")
    @classmethod
    def _empty_data_dir_uses_default(cls, value: object) -> object:
        # Un DATA_DIR vide dans le .env revient au dossier <dépôt>/data
        if value in (None, ""):
            return REPO_ROOT / "data"
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
