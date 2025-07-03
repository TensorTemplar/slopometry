"""Configuration settings for slopometry."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with support for .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="SLOPOMETRY_",
    )

    # Database settings
    database_path: Path = Path(".claude/slopometry.db")

    # Hook handler settings
    python_executable: str | None = None

    # Session settings
    session_id_prefix: str = ""

    # Claude settings backup
    backup_existing_settings: bool = True

    # Display settings
    event_display_limit: int = 50
    recent_sessions_limit: int = 10

    # Debug settings
    debug_mode: bool = False
    
    # Complexity feedback settings
    enable_stop_feedback: bool = False

    @property
    def hook_command(self) -> str:
        """Get the hook command to execute."""
        # Use uv run to ensure we use the correct environment
        if self.python_executable:
            return f"{self.python_executable} -m slopometry.hook_handler"
        return "uv run python -m slopometry.hook_handler"


# Global settings instance
settings = Settings()
