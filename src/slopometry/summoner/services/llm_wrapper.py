import logging
import subprocess
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from slopometry.core.models.experiment import FeatureBoundary
from slopometry.core.settings import settings

logger = logging.getLogger(__name__)


class OfflineModeError(Exception):
    """Raised when attempting to use LLM features while offline_mode is enabled."""

    def __init__(self):
        super().__init__(
            "LLM features are disabled (offline_mode=True). "
            "Set SLOPOMETRY_OFFLINE_MODE=false to enable external requests."
        )


def get_agent() -> Agent:
    """Return the single MiniMax-M3 agent used for all LLM-based tasks.

    The endpoint is the public vLLM-hosted MiniMax-M3 (MXFP4) deployment
    exposed at https://llm2.droidcraft.org/minimax-m3-mxfp4-vllm/v1.
    Raises OfflineModeError when offline_mode is enabled.
    """
    if settings.offline_mode:
        raise OfflineModeError()

    provider = OpenAIProvider(
        base_url=settings.llm_proxy_url,
        api_key=settings.llm_proxy_api_key,
    )
    return Agent(
        name=settings.llm_model_name,
        model=OpenAIChatModel(model_name=settings.llm_model_name, provider=provider),
    )


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


def get_feature_boundaries(limit: int = 20) -> list[FeatureBoundary]:
    """Identify feature boundaries by finding merge commits and their base commits.

    Args:
        limit: Maximum number of merge commits to analyze

    Returns:
        List of feature info with base commit, head commit, and description
    """
    try:
        merge_log = subprocess.run(
            ["git", "log", "HEAD", "--merges", f"-{limit}", "--format=%H|%P|%s"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to find merge commits: {e}")
        return []

    current_repo_path = Path.cwd()
    features: list[FeatureBoundary] = []

    for line in merge_log.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        commit_hash, parents_raw, message = parts
        parents = parents_raw.split()
        if len(parents) < 2:
            continue
        feature_branch = parents[1]

        try:
            merge_base = subprocess.run(
                ["git", "merge-base", parents[0], feature_branch],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            feature_tip_message = subprocess.run(
                ["git", "log", "-1", "--format=%s", feature_branch],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            continue

        features.append(
            FeatureBoundary(
                base_commit=merge_base,
                head_commit=feature_branch,
                merge_commit=commit_hash,
                merge_message=message,
                feature_message=feature_tip_message,
                repository_path=current_repo_path,
            )
        )

    return features
