"""HookEventAdapter protocol and registry for harness-specific wire formats."""

from slopometry.core.protocol.adapters.base import ADAPTERS, HookEventAdapter, register_adapter
from slopometry.core.protocol.adapters.claude_code import ClaudeCodeAdapter
from slopometry.core.protocol.adapters.opencode import OpenCodeAdapter

register_adapter(ClaudeCodeAdapter())
register_adapter(OpenCodeAdapter())

__all__ = ["ADAPTERS", "HookEventAdapter", "register_adapter"]
