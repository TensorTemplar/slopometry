"""Tests for hf_uploader functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from slopometry.summoner.services.hf_uploader import upload_to_huggingface
from slopometry.summoner.services.llm_wrapper import OfflineModeError


def test_upload_to_huggingface__raises_offline_mode_error_when_offline():
    """Test that upload fails explicitly when offline_mode is enabled."""
    with patch("slopometry.summoner.services.hf_uploader.settings") as mock_settings:
        mock_settings.offline_mode = True

        with pytest.raises(OfflineModeError) as exc_info:
            upload_to_huggingface(Path("/fake/path.parquet"), "user/repo")

        assert "offline_mode=True" in str(exc_info.value)
        assert "SLOPOMETRY_OFFLINE_MODE=false" in str(exc_info.value)
