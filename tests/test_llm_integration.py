"""Integration tests for the LLM agent.

These tests make real API calls and require running LLM services.
Skip by default - run with: SLOPOMETRY_RUN_INTEGRATION_TESTS=1 pytest tests/test_llm_integration.py -v
"""

import os

import pytest

from slopometry.core.settings import settings

_INTEGRATION_TESTS_ENABLED = os.environ.get("SLOPOMETRY_RUN_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes")

skip_without_integration_flag = pytest.mark.skipif(
    not _INTEGRATION_TESTS_ENABLED,
    reason="Integration tests skipped: set SLOPOMETRY_RUN_INTEGRATION_TESTS=1 to run",
)


@pytest.fixture
def agent():
    """Fixture providing the MiniMax-M3 agent."""
    from slopometry.summoner.services.llm_wrapper import get_agent

    return get_agent()


@skip_without_integration_flag
def test_minimax_m3__returns_response_when_given_simple_prompt(agent):
    """Test that MiniMax-M3 returns a response for a simple prompt."""
    prompt = "What is 2 + 2? Reply with just the number."

    result = agent.run_sync(prompt)

    assert result is not None
    assert result.output is not None
    assert "4" in result.output


@skip_without_integration_flag
def test_minimax_m3__handles_code_analysis_prompt(agent):
    """Test that MiniMax-M3 can analyze a simple code diff."""
    prompt = """Analyze this Python code change and describe what it does in one sentence:

```diff
- def greet():
-     print("Hello")
+ def greet(name: str):
+     print(f"Hello, {name}!")
```"""

    result = agent.run_sync(prompt)

    assert result is not None
    assert result.output is not None
    assert len(result.output) > 10


@skip_without_integration_flag
def test_minimax_m3__agent_name_matches_settings(agent):
    """Test that the agent name matches the configured llm_model_name."""
    assert agent.name == settings.llm_model_name
