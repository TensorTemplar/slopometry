import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from slopometry.core.models import FeatureBoundary, MergeCommit
from slopometry.core.settings import settings


class OfflineModeError(Exception):
    """Raised when attempting to use LLM features while offline_mode is enabled."""

    def __init__(self):
        super().__init__(
            "LLM features are disabled (offline_mode=True). "
            "Set SLOPOMETRY_OFFLINE_MODE=false to enable external requests."
        )


def _create_providers() -> tuple[OpenAIProvider, OpenAIProvider, AnthropicProvider]:
    """Create LLM providers. Only called when offline_mode is disabled."""
    llm_gateway = OpenAIProvider(base_url=settings.llm_proxy_url, api_key=settings.llm_proxy_api_key)
    responses_api_gateway = OpenAIProvider(base_url=settings.llm_responses_url, api_key=settings.llm_proxy_api_key)
    anthropic_gateway = AnthropicProvider(
        base_url=settings.anthropic_url, api_key=settings.anthropic_api_key.get_secret_value()
    )
    return llm_gateway, responses_api_gateway, anthropic_gateway


def _create_agents() -> dict[str, Agent]:
    """Create all available agents. Only called when offline_mode is disabled."""
    llm_gateway, responses_api_gateway, anthropic_gateway = _create_providers()

    return {
        "gpt_oss_120b": Agent(
            name="gpt_oss_120b",
            model=OpenAIResponsesModel("openai/gpt-oss-120b", provider=responses_api_gateway),
            retries=2,
            end_strategy="exhaustive",
            model_settings=OpenAIResponsesModelSettings(
                max_tokens=64000,
                seed=1337,
                openai_reasoning_effort="medium",
                temperature=1.0,
            ),
        ),
        "gemini": Agent(
            name="gemini",
            model=OpenAIChatModel(model_name="gemini-3-pro-preview", provider=llm_gateway),
        ),
        "minimax": Agent(
            name="minimax",
            model=AnthropicModel("minimax:MiniMax-M1.1", provider=anthropic_gateway),
            retries=2,
            end_strategy="exhaustive",
        ),
    }


_agents: dict[str, Agent] | None = None


def _get_agents() -> dict[str, Agent]:
    """Get or create the agents registry. Raises OfflineModeError if offline_mode is enabled."""
    global _agents
    if settings.offline_mode:
        raise OfflineModeError()
    if _agents is None:
        _agents = _create_agents()
    return _agents


def get_user_story_agent() -> Agent:
    """Get the configured agent for user story generation."""
    agents = _get_agents()
    agent_name = settings.user_story_agent
    if agent_name not in agents:
        raise ValueError(f"Unknown user_story_agent: {agent_name}. Available: {list(agents.keys())}")
    return agents[agent_name]


def get_user_story_prompt(diff: str) -> str:
    """Generate a prompt for creating user stories from a git diff.

    Args:
        diff: The git diff content to analyze

    Returns:
        Formatted prompt string with the diff included
    """
    return f"""
<instructions>
You are in the role of a principal software engineer.
Your task is to look at a git diff that will be provided in separate xml tags, between the current state and
 some past state of the codebase and create detailed user stories from what was implemented, according to this diff, 
 for re-implementation from scratch.

Your target audience is mid-level SWE so chose a balanced level of detail when creating the user stories, 
without biasing the implementation too much in any direction nor requesting specific implementation approaches.

When creating user stories, focus on functional parts of the diff and ignore non-functional parts, 
like changes in lock files, project requirements, readme files and so on.

Your output should be formatted markdown.
</instructions>

<diff_to_inspect>
{diff}
</diff_to_inspect>
"""


def resolve_commit_reference(commit_ref: str) -> str:
    """Resolve a commit reference to its absolute hash.

    Args:
        commit_ref: Git commit reference (e.g., 'HEAD~3', 'abc123', 'main')

    Returns:
        The absolute commit hash
    """
    try:
        result = subprocess.run(["git", "rev-parse", commit_ref], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.debug(f"Could not resolve commit ref '{commit_ref}', returning original: {e}")
        return commit_ref


def calculate_stride_size(base_commit: str, head_commit: str) -> int:
    """Calculate the number of commits between base and head.

    Args:
        base_commit: Base commit reference
        head_commit: Head commit reference

    Returns:
        Number of commits between base and head (stride size)
    """
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", f"{base_commit}..{head_commit}"], capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.debug(f"Could not calculate stride between {base_commit}..{head_commit}, using default 1: {e}")
        return 1


def get_commit_diff(base_commit: str, head_commit: str) -> str:
    """Get the diff between two commits.

    Args:
        base_commit: Base commit hash or reference
        head_commit: Head commit hash or reference

    Returns:
        The git diff output as a string
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{base_commit}..{head_commit}"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error getting diff: {e.stderr}"


def find_merge_commits(branch: str = "HEAD", limit: int = 50) -> list[MergeCommit]:
    """Find merge commits in the git history.

    Args:
        branch: Branch to analyze (default: HEAD)
        limit: Maximum number of commits to examine

    Returns:
        List of merge commit info with hash, message, and parent commits
    """
    try:
        result = subprocess.run(
            ["git", "log", branch, "--merges", f"-{limit}", "--format=%H|%P|%s"],
            capture_output=True,
            text=True,
            check=True,
        )

        merge_commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) >= 3:
                commit_hash = parts[0]
                parents = parts[1].split()
                message = parts[2]

                if len(parents) >= 2:
                    merge_commits.append(
                        MergeCommit(
                            hash=commit_hash,
                            parents=parents,
                            message=message,
                            feature_branch=parents[1],  # Second parent is typically the feature branch
                        )
                    )

        return merge_commits
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to find merge commits: {e}")
        return []


def get_feature_boundaries(limit: int = 20) -> list[FeatureBoundary]:
    """Identify feature boundaries by finding merge commits and their base commits.

    Args:
        limit: Maximum number of merge commits to analyze

    Returns:
        List of feature info with base commit, head commit, and description
    """

    merge_commits = find_merge_commits(limit=limit)
    features = []
    current_repo_path = Path.cwd()

    for merge in merge_commits:
        try:
            result = subprocess.run(
                ["git", "merge-base", merge.parents[0], merge.parents[1]], capture_output=True, text=True, check=True
            )
            merge_base = result.stdout.strip()

            result = subprocess.run(
                ["git", "log", "-1", "--format=%s", merge.feature_branch], capture_output=True, text=True, check=True
            )
            feature_tip_message = result.stdout.strip()

            features.append(
                FeatureBoundary(
                    base_commit=merge_base,
                    head_commit=merge.feature_branch,
                    merge_commit=merge.hash,
                    merge_message=merge.message,
                    feature_message=feature_tip_message,
                    repository_path=current_repo_path,
                )
            )
        except subprocess.CalledProcessError:
            continue

    return features
