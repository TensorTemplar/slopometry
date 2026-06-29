"""Tests for MemoryExtractor (Claude Code + OpenCode transcript parsing)."""

import json
from pathlib import Path

import pytest

from slopometry.solo.services.memory_extractor import MemoryExtractor, TranscriptTruncationConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestExtractFromOpencodeSession:
    def test_extract_memories_from_opencode_session__reconstructs_user_text_only(self, tmp_path: Path):
        storage = tmp_path / "opencode_storage"
        _write_json(
            storage / "message" / "ses_a" / "msg_a1.json",
            {"id": "msg_a1", "role": "user", "sessionID": "ses_a", "time": {"created": 1000}},
        )
        _write_json(
            storage / "part" / "msg_a1" / "p1.json",
            {"id": "p1", "type": "text", "text": "hello world", "messageID": "msg_a1", "sessionID": "ses_a"},
        )
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out = extractor.extract_memories_from_opencode_session("ses_a", storage)
        assert "USER:" in out
        assert "hello world" in out

    def test_extract_memories_from_opencode_session__reconstructs_assistant_text(self, tmp_path: Path):
        storage = tmp_path / "opencode_storage"
        _write_json(
            storage / "message" / "ses_b" / "msg_b1.json",
            {"id": "msg_b1", "role": "assistant", "sessionID": "ses_b", "time": {"created": 2000}},
        )
        _write_json(
            storage / "part" / "msg_b1" / "p1.json",
            {"id": "p1", "type": "text", "text": "I will check that", "messageID": "msg_b1", "sessionID": "ses_b"},
        )
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out = extractor.extract_memories_from_opencode_session("ses_b", storage)
        assert "ASSISTANT:" in out
        assert "I will check that" in out

    def test_extract_memories_from_opencode_session__emits_tool_marker_with_input_output(self, tmp_path: Path):
        storage = tmp_path / "opencode_storage"
        _write_json(
            storage / "message" / "ses_c" / "msg_c1.json",
            {"id": "msg_c1", "role": "assistant", "sessionID": "ses_c", "time": {"created": 3000}},
        )
        _write_json(
            storage / "part" / "msg_c1" / "p1.json",
            {
                "id": "p1",
                "type": "tool",
                "tool": "bash",
                "callID": "call_xyz",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls -la"},
                    "output": "total 12\ndrwxr-xr-x",
                },
                "messageID": "msg_c1",
                "sessionID": "ses_c",
            },
        )
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out = extractor.extract_memories_from_opencode_session("ses_c", storage)
        assert "TOOL: bash" in out
        assert "ls -la" in out

    def test_extract_memories_from_opencode_session__skips_step_start_and_reasoning_parts(self, tmp_path: Path):
        storage = tmp_path / "opencode_storage"
        _write_json(
            storage / "message" / "ses_d" / "msg_d1.json",
            {"id": "msg_d1", "role": "assistant", "sessionID": "ses_d", "time": {"created": 4000}},
        )
        _write_json(
            storage / "part" / "msg_d1" / "p_step.json",
            {"id": "p_step", "type": "step-start", "messageID": "msg_d1", "sessionID": "ses_d"},
        )
        _write_json(
            storage / "part" / "msg_d1" / "p_reason.json",
            {
                "id": "p_reason",
                "type": "reasoning",
                "text": "internal thoughts that should not surface",
                "messageID": "msg_d1",
                "sessionID": "ses_d",
            },
        )
        _write_json(
            storage / "part" / "msg_d1" / "p_text.json",
            {
                "id": "p_text",
                "type": "text",
                "text": "user-visible reply",
                "messageID": "msg_d1",
                "sessionID": "ses_d",
            },
        )
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out = extractor.extract_memories_from_opencode_session("ses_d", storage)
        assert "user-visible reply" in out
        assert "internal thoughts" not in out

    def test_extract_memories_from_opencode_session__returns_empty_string_when_message_dir_missing(self, tmp_path: Path):
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out = extractor.extract_memories_from_opencode_session("ses_none", tmp_path / "opencode_storage")
        assert out == ""

    def test_extract_memories_from_opencode_session__orders_messages_chronologically(self, tmp_path: Path):
        storage = tmp_path / "opencode_storage"
        _write_json(
            storage / "message" / "ses_e" / "msg_e1.json",
            {"id": "msg_e1", "role": "user", "sessionID": "ses_e", "time": {"created": 2000}},
        )
        _write_json(
            storage / "part" / "msg_e1" / "p1.json",
            {"id": "p1", "type": "text", "text": "second message", "messageID": "msg_e1", "sessionID": "ses_e"},
        )
        _write_json(
            storage / "message" / "ses_e" / "msg_e0.json",
            {"id": "msg_e0", "role": "user", "sessionID": "ses_e", "time": {"created": 1000}},
        )
        _write_json(
            storage / "part" / "msg_e0" / "p1.json",
            {"id": "p1", "type": "text", "text": "first message", "messageID": "msg_e0", "sessionID": "ses_e"},
        )
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out = extractor.extract_memories_from_opencode_session("ses_e", storage)
        first_idx = out.index("first message")
        second_idx = out.index("second message")
        assert first_idx < second_idx

    def test_extract_memories_from_opencode_session__skips_unknown_role_messages(self, tmp_path: Path):
        storage = tmp_path / "opencode_storage"
        _write_json(
            storage / "message" / "ses_f" / "msg_f1.json",
            {"id": "msg_f1", "role": "system", "sessionID": "ses_f", "time": {"created": 1000}},
        )
        _write_json(
            storage / "part" / "msg_f1" / "p1.json",
            {"id": "p1", "type": "text", "text": "should not appear", "messageID": "msg_f1", "sessionID": "ses_f"},
        )
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out = extractor.extract_memories_from_opencode_session("ses_f", storage)
        assert "should not appear" not in out

    def test_extract_memories_from_opencode_session__respects_truncation_config_for_tool_parts(self, tmp_path: Path):
        storage = tmp_path / "opencode_storage"
        _write_json(
            storage / "message" / "ses_g" / "msg_g1.json",
            {"id": "msg_g1", "role": "assistant", "sessionID": "ses_g", "time": {"created": 1}},
        )
        long_input = "x" * 500
        long_output = "y" * 500
        _write_json(
            storage / "part" / "msg_g1" / "p1.json",
            {
                "id": "p1",
                "type": "tool",
                "tool": "bash",
                "state": {"input": {"command": long_input}, "output": long_output},
                "messageID": "msg_g1",
                "sessionID": "ses_g",
            },
        )
        extractor = MemoryExtractor("https://llm.example/v1", "model-x", "test-key")
        out_default = extractor.extract_memories_from_opencode_session("ses_g", storage)
        assert len(out_default) < 500 + 500 + 100
        out_short = extractor.extract_memories_from_opencode_session(
            "ses_g",
            storage,
            truncation=TranscriptTruncationConfig(tool_input_chars=10, tool_output_chars=10),
        )
        assert len(out_short) < len(out_default)


class TestTranscriptTruncationConfig:
    def test_transcript_truncation_config__defaults_match_pre_refactor_behavior(self):
        c = TranscriptTruncationConfig()
        assert c.tool_input_chars == 120
        assert c.tool_output_chars == 120
        assert c.tool_result_chars == 200

    def test_transcript_truncation_config__rejects_extra_fields(self):
        with pytest.raises(Exception):
            TranscriptTruncationConfig(unknown_field=42)
