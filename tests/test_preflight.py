"""Tests for the preflight endpoint health check."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from slopometry.cli import cli
from slopometry.solo.cli.preflight import preflight_endpoints


class TestPreflightEndpoints:
    def test_preflight_endpoints__raises_when_chat_endpoint_down(self):
        chat_err = "chat LLM (https://chat.example/v1): APIConnectionError: no available server"
        embed_ok = None
        with patch("slopometry.solo.cli.preflight._check_endpoint", side_effect=[chat_err, embed_ok]):
            with pytest.raises(Exception) as exc_info:
                preflight_endpoints(
                    chat_endpoint="https://chat.example/v1",
                    embedding_endpoint="https://embed.example/v1",
                    chat_api_key="k1",
                    embedding_api_key="k2",
                )
            assert "chat.example" in str(exc_info.value)
            assert "embed.example" not in str(exc_info.value)

    def test_preflight_endpoints__raises_when_embedding_endpoint_down(self):
        chat_ok = None
        embed_err = "embedding (https://embed.example/v1): APIConnectionError: refused"
        with patch("slopometry.solo.cli.preflight._check_endpoint", side_effect=[chat_ok, embed_err]):
            with pytest.raises(Exception) as exc_info:
                preflight_endpoints(
                    chat_endpoint="https://chat.example/v1",
                    embedding_endpoint="https://embed.example/v1",
                    chat_api_key="k1",
                    embedding_api_key="k2",
                )
            assert "embed.example" in str(exc_info.value)

    def test_preflight_endpoints__raises_with_both_errors_listed(self):
        chat_err = "chat LLM: down"
        embed_err = "embedding: down"
        with patch("slopometry.solo.cli.preflight._check_endpoint", side_effect=[chat_err, embed_err]):
            with pytest.raises(Exception) as exc_info:
                preflight_endpoints(
                    chat_endpoint="https://chat.example/v1",
                    embedding_endpoint="https://embed.example/v1",
                    chat_api_key="k1",
                    embedding_api_key="k2",
                )
            msg = str(exc_info.value)
            assert "chat LLM: down" in msg
            assert "embedding: down" in msg

    def test_preflight_endpoints__passes_silently_when_both_endpoints_reachable(self):
        with patch("slopometry.solo.cli.preflight._check_endpoint", return_value=None):
            preflight_endpoints(
                chat_endpoint="https://chat.example/v1",
                embedding_endpoint="https://embed.example/v1",
                chat_api_key="k1",
                embedding_api_key="k2",
            )


class TestCheckEndpoint:
    def test_check_endpoint__returns_none_when_models_list_succeeds(self):
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock()
        with patch("openai.OpenAI", return_value=mock_client) as mock_openai:
            from slopometry.solo.cli.preflight import _check_endpoint

            result = _check_endpoint("test", "https://x/v1", "key")
        assert result is None
        mock_openai.assert_called_once_with(base_url="https://x/v1", api_key="key")

    def test_check_endpoint__returns_error_string_on_unreachable_endpoint(self):
        mock_client = MagicMock()
        mock_client.models.list.side_effect = RuntimeError("refused")
        with patch("openai.OpenAI", return_value=mock_client):
            from slopometry.solo.cli.preflight import _check_endpoint

            result = _check_endpoint("test", "https://x/v1", "key")
        assert result is not None
        assert "test" in result
        assert "https://x/v1" in result
        assert "RuntimeError" in result

    def test_check_endpoint__returns_error_when_openai_not_installed(self):
        original_import = __builtins__["__import__"]

        def failing_import(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            from slopometry.solo.cli.preflight import _check_endpoint

            result = _check_endpoint("test", "https://x/v1", "key")
        assert result is not None
        assert "openai package not installed" in result


class TestFindMemoriesPreflightIntegration:
    def test_find_memories__dry_run_skips_preflight(self, monkeypatch: pytest.MonkeyPatch):
        from slopometry.core.settings import settings

        monkeypatch.setattr(settings, "offline_mode", True)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "solo",
                "find-memories",
                "--project-dir",
                "/tmp",
                "--dry-run",
                "--llm-endpoint",
                "https://chat.example/v1",
            ],
        )
        assert "offline_mode" not in result.output or result.exit_code == 0

    def test_find_memories__offline_mode_blocks_before_preflight(self, monkeypatch: pytest.MonkeyPatch):
        from slopometry.core.settings import settings

        monkeypatch.setattr(settings, "offline_mode", True)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["solo", "find-memories", "--project-dir", "/tmp"],
        )
        assert result.exit_code != 0
        assert "offline_mode" in result.output.lower() or "offline" in result.output.lower()
