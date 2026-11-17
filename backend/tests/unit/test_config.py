from pathlib import Path

import pytest

from ledger_agent.config import REPO_ROOT, Settings


def test_env_variables_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "test-agent test@example.com")
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.sec_user_agent == "test-agent test@example.com"
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "secret-value"


def test_secrets_are_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert "secret-value" not in repr(settings)


def test_empty_data_dir_falls_back_to_repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", "")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.data_dir == REPO_ROOT / "data"


def test_custom_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.data_dir == tmp_path
