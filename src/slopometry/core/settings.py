"""Configuration settings for slopometry."""

import os
import sys
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_data_dir() -> Path:
    """Get platform-specific default data directory."""
    app_name = "slopometry"

    if sys.platform == "win32":
        # Windows: %LOCALAPPDATA%\slopometry
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = Path.home() / "AppData" / "Local"
        return Path(base) / app_name
    elif sys.platform == "darwin":
        # macOS: ~/Library/Application Support/slopometry
        return Path.home() / "Library" / "Application Support" / app_name
    else:
        # Linux/Unix: ~/.local/share/slopometry
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / app_name
        return Path.home() / ".local" / "share" / app_name


class Settings(BaseSettings):
    """Application settings with support for .env files."""

    model_config = SettingsConfigDict(
        env_file=[
            Path.home() / ".config/slopometry/.env",
            ".env",
        ],
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="SLOPOMETRY_",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        """Initialize settings and ensure global config directory exists."""
        self._ensure_global_config_dir()
        super().__init__(**kwargs)

    @staticmethod
    def _ensure_global_config_dir() -> None:
        """Ensure the global config directory exists."""
        global_config_dir = Path.home() / ".config/slopometry"
        global_config_dir.mkdir(parents=True, exist_ok=True)

    database_path: Path | None = None

    python_executable: str | None = None

    session_id_prefix: str = ""

    backup_existing_settings: bool = True

    event_display_limit: int = 50
    recent_sessions_limit: int = 10

    debug_mode: bool = False

    enable_complexity_analysis: bool = True
    enable_complexity_feedback: bool = False

    llm_proxy_url: str = ""
    llm_proxy_api_key: str = ""
    interactive_rating_enabled: bool = False

    hf_token: str = ""
    hf_default_repo: str = ""

    # User story generation settings
    user_story_agents: list[str] = ["o3", "claude-opus-4", "gemini-2.5-pro"]

    @field_validator("database_path", mode="before")
    @classmethod
    def validate_database_path(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def resolved_database_path(self) -> Path:
        """Get the resolved database path, using default if not set."""
        if self.database_path is not None:
            return self.database_path.resolve()

        data_dir = get_default_data_dir()
        return data_dir / "slopometry.db"

    @property
    def hook_command(self) -> str:
        """Get the hook command to execute."""
        if self.python_executable:
            return f"{self.python_executable} -m slopometry.hook_handler"

        return "slopometry hook-handler"


settings = Settings()
