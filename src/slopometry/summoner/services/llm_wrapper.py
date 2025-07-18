import subprocess
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from slopometry.core.models import FeatureBoundary, MergeCommit
from slopometry.core.settings import settings

llm_gateway = OpenAIProvider(base_url=settings.llm_proxy_url, api_key=settings.llm_proxy_api_key)

# In llm gateway land, everything is an openai model
gemma_fp8 = OpenAIModel(model_name="gemma3:27b-it-q8_0", provider=llm_gateway)
cluade = Agent(model=OpenAIModel(model_name="claude-opus-4", provider=llm_gateway))
gemini = Agent(model=OpenAIModel(model_name="gemini-2.5-pro", provider=llm_gateway))
user_story_agent = Agent(model=OpenAIModel(model_name="o3", provider=llm_gateway))


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
    except subprocess.CalledProcessError:
        return commit_ref  # Return original if resolution fails


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
    except (subprocess.CalledProcessError, ValueError):
        return 1  # Default to 1 if calculation fails


def get_commit_diff(commit_ref: str) -> str:
    """Get the diff between a specific commit and the current state.

    Args:
        commit_ref: Git commit hash or reference (e.g., 'HEAD~3', 'abc123')

    Returns:
        The git diff output as a string
    """
    try:
        result = subprocess.run(["git", "diff", f"{commit_ref}..HEAD"], capture_output=True, text=True, check=True)
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
        # Get merge commits with their parents
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

                if len(parents) >= 2:  # True merge commit has 2+ parents
                    merge_commits.append(
                        MergeCommit(
                            hash=commit_hash,
                            parents=parents,
                            message=message,
                            feature_branch=parents[1],  # Second parent is typically the feature branch
                        )
                    )

        return merge_commits
    except subprocess.CalledProcessError:
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
            # Find the common ancestor of the two parents (merge base)
            result = subprocess.run(
                ["git", "merge-base", merge.parents[0], merge.parents[1]], capture_output=True, text=True, check=True
            )
            merge_base = result.stdout.strip()

            # Get the feature branch tip commit message for better context
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
