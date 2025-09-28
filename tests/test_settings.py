"""Tests for configuration settings and override priority."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from slopometry.core.settings import Settings


class TestSettingsOverridePriority:
    """Test configuration override priority: local > global > defaults."""

    def _create_test_settings(self, global_config_path: Path, local_config_path: Path):
        """Create a Settings class with custom config paths for testing."""
        from pydantic_settings import BaseSettings, SettingsConfigDict

        class TestSettings(BaseSettings):
            model_config = SettingsConfigDict(
                env_file=[str(global_config_path), str(local_config_path)],
                env_file_encoding="utf-8",
                case_sensitive=False,
                env_prefix="SLOPOMETRY_",
                extra="ignore",
            )

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
            user_story_agents: list[str] = ["o3", "claude-opus-4", "gemini-2.5-pro"]

        return TestSettings

    def test_configuration_override_priority(self):
        """Test that local config overrides global config which overrides defaults."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create global config with some values
            global_config = temp_path / "global.env"
            global_config.write_text(
                "SLOPOMETRY_EVENT_DISPLAY_LIMIT=100\nSLOPOMETRY_DEBUG_MODE=true\nSLOPOMETRY_SESSION_ID_PREFIX=global_\n"
            )

            # Create local config that overrides some global values
            local_config = temp_path / "local.env"
            local_config.write_text("SLOPOMETRY_EVENT_DISPLAY_LIMIT=200\nSLOPOMETRY_RECENT_SESSIONS_LIMIT=25\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                TestSettings = self._create_test_settings(global_config, local_config)
                settings = TestSettings()

                assert settings.event_display_limit == 200

                assert settings.debug_mode is True
                assert settings.session_id_prefix == "global_"

                assert settings.recent_sessions_limit == 25

                assert settings.backup_existing_settings is True

    def test_environment_variables_override_all(self):
        """Test that environment variables override both local and global config."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            global_config = temp_path / "global.env"
            global_config.write_text("SLOPOMETRY_EVENT_DISPLAY_LIMIT=100\n")

            local_config = temp_path / "local.env"
            local_config.write_text("SLOPOMETRY_EVENT_DISPLAY_LIMIT=200\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {"SLOPOMETRY_EVENT_DISPLAY_LIMIT": "300"}, clear=False):
                for var in env_vars_to_clear:
                    if var != "SLOPOMETRY_EVENT_DISPLAY_LIMIT":
                        os.environ.pop(var, None)

                TestSettings = self._create_test_settings(global_config, local_config)
                settings = TestSettings()

                assert settings.event_display_limit == 300

    def test_global_config_directory_creation(self):
        """Test that global config directory is created during initialization."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_home = temp_path / "home"
            mock_home.mkdir()

            global_config_dir = mock_home / ".config/slopometry"
            assert not global_config_dir.exists()

            with patch("pathlib.Path.home", return_value=mock_home):
                Settings()

                assert global_config_dir.exists()
                assert global_config_dir.is_dir()

    def test_defaults_when_no_config_files(self):
        """Test that default values are used when no config files exist."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            global_config = temp_path / "nonexistent_global.env"
            local_config = temp_path / "nonexistent_local.env"

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                TestSettings = self._create_test_settings(global_config, local_config)
                settings = TestSettings()

                assert settings.event_display_limit == 50
                assert settings.recent_sessions_limit == 10
                assert settings.debug_mode is False
                assert settings.backup_existing_settings is True
                assert settings.session_id_prefix == ""

    def test_config_priority_explicit_order(self):
        """Test explicit priority order: env vars > local .env > global .env > defaults."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Default value (from class definition): event_display_limit = 50

            # 2. Global config
            global_config = temp_path / "global.env"
            global_config.write_text("SLOPOMETRY_EVENT_DISPLAY_LIMIT=75\n")

            # 3. Local config (should override global)
            local_config = temp_path / "local.env"
            local_config.write_text("SLOPOMETRY_EVENT_DISPLAY_LIMIT=125\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                TestSettings1 = self._create_test_settings(global_config, local_config)
                settings1 = TestSettings1()
                assert settings1.event_display_limit == 125

                with patch.dict(os.environ, {"SLOPOMETRY_EVENT_DISPLAY_LIMIT": "999"}, clear=False):
                    TestSettings2 = self._create_test_settings(global_config, local_config)
                    settings2 = TestSettings2()
                    assert settings2.event_display_limit == 999

                local_config.unlink()
                nonexistent_local = temp_path / "nonexistent_local.env"
                TestSettings3 = self._create_test_settings(global_config, nonexistent_local)
                settings3 = TestSettings3()
                assert settings3.event_display_limit == 75

                global_config.unlink()
                nonexistent_global = temp_path / "nonexistent_global.env"
                TestSettings4 = self._create_test_settings(nonexistent_global, nonexistent_local)
                settings4 = TestSettings4()
                assert settings4.event_display_limit == 50
