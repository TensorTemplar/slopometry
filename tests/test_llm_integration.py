"""Integration tests for LLM agents.

These tests make real API calls and require running LLM services.
Skip by default - run with: SLOPOMETRY_RUN_INTEGRATION_TESTS=1 pytest tests/test_llm_integration.py -v
"""

import os

import pytest

_INTEGRATION_TESTS_ENABLED = os.environ.get("SLOPOMETRY_RUN_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes")

skip_without_integration_flag = pytest.mark.skipif(
    not _INTEGRATION_TESTS_ENABLED,
    reason="Integration tests skipped: set SLOPOMETRY_RUN_INTEGRATION_TESTS=1 to run",
)


@pytest.fixture
def agents():
    """Fixture providing the agents registry."""
    from slopometry.summoner.services.llm_wrapper import _get_agents

    return _get_agents()


@skip_without_integration_flag
def test_gpt_oss_120b__returns_response_when_given_simple_prompt(agents):
    """Test that gpt_oss_120b returns a response for a simple prompt."""
    agent = agents["gpt_oss_120b"]
    prompt = "What is 2 + 2? Reply with just the number."

    result = agent.run_sync(prompt)

    assert result is not None
    assert result.output is not None
    assert len(result.output) > 0
    assert "4" in result.output


@skip_without_integration_flag
def test_gpt_oss_120b__handles_code_analysis_prompt(agents):
    """Test that gpt_oss_120b can analyze a simple code diff."""
    agent = agents["gpt_oss_120b"]
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
def test_gemini__returns_response_when_given_simple_prompt(agents):
    """Test that gemini agent returns a response."""
    agent = agents["gemini"]
    prompt = "What is the capital of France? Reply with just the city name."

    result = agent.run_sync(prompt)

    assert result is not None
    assert result.output is not None
    assert "Paris" in result.output


@skip_without_integration_flag
def test_get_user_story_agent__returns_configured_agent():
    """Test that get_user_story_agent returns the agent configured in settings."""
    from slopometry.summoner.services.llm_wrapper import get_user_story_agent

    agent = get_user_story_agent()

    assert agent is not None
    assert agent.name == settings.user_story_agent
