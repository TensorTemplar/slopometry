"""Tests for configuration settings and override priority."""

import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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
            offline_mode: bool = True
            user_story_agent: str = "gemini"

        return TestSettings

    def test_settings_init__prioritizes_local_over_global_config(self):
        """Test that local config overrides global config which overrides defaults."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            global_config = temp_path / "global.env"
            global_config.write_text(
                "SLOPOMETRY_EVENT_DISPLAY_LIMIT=100\nSLOPOMETRY_DEBUG_MODE=true\nSLOPOMETRY_SESSION_ID_PREFIX=global_\n"
            )

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

    def test_settings_init__prioritizes_env_vars_over_all_config(self):
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

    def test_config_dir_creation__creates_xdg_path_on_linux(self):
        """Test that global config directory is created (Linux/XDG)."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_home = temp_path / "home"
            mock_home.mkdir()

            global_config_dir = mock_home / ".config/slopometry"
            assert not global_config_dir.exists()

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": "", "XDG_DATA_HOME": ""}, clear=False):
                with patch("pathlib.Path.home", return_value=mock_home):
                    with patch("sys.platform", "linux"):
                        Settings()

                    assert global_config_dir.exists()
                    assert global_config_dir.is_dir()

    def test_config_dir_creation__creates_app_support_path_on_mac(self):
        """Test that global config directory is created (macOS)."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_home = temp_path / "home"
            mock_home.mkdir()

            global_config_dir = mock_home / "Library/Application Support/slopometry"
            assert not global_config_dir.exists()

            with patch("pathlib.Path.home", return_value=mock_home):
                with patch("sys.platform", "darwin"):
                    Settings()

                assert global_config_dir.exists()
                assert global_config_dir.is_dir()

    def test_config_dir_creation__creates_localappdata_path_on_windows(self):
        """Test that global config directory is created (Windows)."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_localappdata = temp_path / "AppData/Local"
            mock_localappdata.mkdir(parents=True)

            global_config_dir = mock_localappdata / "slopometry"
            assert not global_config_dir.exists()

            with patch.dict(os.environ, {"LOCALAPPDATA": str(mock_localappdata)}, clear=False):
                with patch("sys.platform", "win32"):
                    Settings()

                assert global_config_dir.exists()
                assert global_config_dir.is_dir()

    def test_settings_init__uses_defaults_when_no_config_files_exist(self):
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

    def test_settings_init__respects_explicit_priority_order(self):
        """Test explicit priority order: env vars > local .env > global .env > defaults."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            global_config = temp_path / "global.env"
            global_config.write_text("SLOPOMETRY_EVENT_DISPLAY_LIMIT=75\n")

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


class TestUnknownSettingsWarning:
    """Test warnings for unknown SLOPOMETRY_ prefixed settings."""

    def test_warn_unknown_prefixed_settings__warns_on_typo_in_env_var(self):
        """Settings init warns when unknown SLOPOMETRY_ env vars are detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dotenv_file = temp_path / ".env"
            dotenv_file.touch()

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(
                os.environ, {"SLOPOMETRY_ENABLE_STOP_FEEDBACK": "true", "SLOPOMETRY_DEBUG_MODE": "true"}, clear=False
            ):
                for var in env_vars_to_clear:
                    if var not in ("SLOPOMETRY_ENABLE_STOP_FEEDBACK", "SLOPOMETRY_DEBUG_MODE"):
                        os.environ.pop(var, None)

                with patch.object(Settings, "model_config", {**Settings.model_config, "env_file": [str(dotenv_file)]}):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        Settings()

                        slopometry_warnings = [warning for warning in w if "SLOPOMETRY_" in str(warning.message)]
                        assert len(slopometry_warnings) == 1
                        assert "SLOPOMETRY_ENABLE_STOP_FEEDBACK" in str(slopometry_warnings[0].message)
                        assert "SLOPOMETRY_DEBUG_MODE" not in str(slopometry_warnings[0].message)

    def test_warn_unknown_prefixed_settings__no_warning_for_valid_settings(self):
        """Settings init does not warn when all SLOPOMETRY_ env vars are valid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dotenv_file = temp_path / ".env"
            dotenv_file.write_text("SLOPOMETRY_DEBUG_MODE=true\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                with patch.object(Settings, "model_config", {**Settings.model_config, "env_file": [str(dotenv_file)]}):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        Settings()

                        slopometry_warnings = [warning for warning in w if "SLOPOMETRY_" in str(warning.message)]
                        assert len(slopometry_warnings) == 0

    def test_warn_unknown_prefixed_settings__warns_on_typo_in_dotenv_file(self):
        """Settings init warns when unknown SLOPOMETRY_ settings are in .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dotenv_file = temp_path / ".env"
            dotenv_file.write_text("SLOPOMETRY_FAKE_SETTING=value\nSLOPOMETRY_DEBUG_MODE=true\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                with patch.object(Settings, "model_config", {**Settings.model_config, "env_file": [str(dotenv_file)]}):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        Settings()

                        slopometry_warnings = [warning for warning in w if "SLOPOMETRY_" in str(warning.message)]
                        assert len(slopometry_warnings) == 1
                        assert "SLOPOMETRY_FAKE_SETTING" in str(slopometry_warnings[0].message)


class TestBaselineStrategyValidator:
    """Tests for baseline_strategy field validator."""

    def test_validate_baseline_strategy__accepts_auto(self):
        """Test that 'auto' is accepted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dotenv_file = temp_path / ".env"
            dotenv_file.write_text("SLOPOMETRY_BASELINE_STRATEGY=auto\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                with patch.object(Settings, "model_config", {**Settings.model_config, "env_file": [str(dotenv_file)]}):
                    s = Settings()
                    assert s.baseline_strategy == "auto"

    def test_validate_baseline_strategy__accepts_merge_anchored(self):
        """Test that 'merge_anchored' is accepted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dotenv_file = temp_path / ".env"
            dotenv_file.write_text("SLOPOMETRY_BASELINE_STRATEGY=merge_anchored\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                with patch.object(Settings, "model_config", {**Settings.model_config, "env_file": [str(dotenv_file)]}):
                    s = Settings()
                    assert s.baseline_strategy == "merge_anchored"

    def test_validate_baseline_strategy__accepts_time_sampled(self):
        """Test that 'time_sampled' is accepted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dotenv_file = temp_path / ".env"
            dotenv_file.write_text("SLOPOMETRY_BASELINE_STRATEGY=time_sampled\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                with patch.object(Settings, "model_config", {**Settings.model_config, "env_file": [str(dotenv_file)]}):
                    s = Settings()
                    assert s.baseline_strategy == "time_sampled"

    def test_validate_baseline_strategy__rejects_invalid_value(self):
        """Test that invalid values are rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dotenv_file = temp_path / ".env"
            dotenv_file.write_text("SLOPOMETRY_BASELINE_STRATEGY=invalid_strategy\n")

            env_vars_to_clear = [k for k in os.environ.keys() if k.startswith("SLOPOMETRY_")]
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)

                with patch.object(Settings, "model_config", {**Settings.model_config, "env_file": [str(dotenv_file)]}):
                    with pytest.raises(ValidationError, match="baseline_strategy"):
                        Settings()
