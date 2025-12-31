"""LLM integration service for summoner features."""

import os
from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.models import UserStoryEntry
from slopometry.core.settings import settings
from slopometry.summoner.services.llm_wrapper import (
    calculate_stride_size,
    get_commit_diff,
    get_feature_boundaries,
    get_user_story_agent,
    get_user_story_prompt,
    resolve_commit_reference,
)


class LLMService:
    """Handles AI/LLM integration for user story generation and analysis."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def generate_user_stories_from_commits(
        self, repo_path: Path, base_commit: str, head_commit: str
    ) -> tuple[int, list[str]]:
        """Generate user stories from commit diffs using configured AI agent.

        Returns:
            Tuple of (successful_generations, error_messages)
        """
        original_dir = os.getcwd()
        error_messages: list[str] = []

        try:
            os.chdir(repo_path)

            resolved_base = resolve_commit_reference(base_commit)
            resolved_head = resolve_commit_reference(head_commit)

            stride_size = calculate_stride_size(resolved_base, resolved_head)

            diff = get_commit_diff(resolved_base, resolved_head)

            if not diff:
                return 0, ["No changes found between commits"]

            prompt = get_user_story_prompt(diff)

            agent = get_user_story_agent()
            result = agent.run_sync(prompt)

            user_story_entry = UserStoryEntry(
                base_commit=resolved_base,
                head_commit=resolved_head,
                diff_content=diff,
                stride_size=stride_size,
                user_stories=result.output,
                rating=3,  # Default neutral rating
                guidelines_for_improving="",
                model_used=settings.user_story_agent,
                prompt_template=prompt,
                repository_path=str(repo_path),
            )

            self.db.save_user_story_entry(user_story_entry)
            return 1, error_messages

        except Exception as e:
            return 0, [f"Failed to generate user stories: {e}"]
        finally:
            os.chdir(original_dir)

    def get_feature_boundaries(self, repo_path: Path, limit: int = 20) -> list:
        """Get detected feature boundaries from merge commits."""

        cached_features = self.db.get_feature_boundaries(repo_path)
        if cached_features:
            return cached_features

        original_dir = os.getcwd()
        try:
            os.chdir(repo_path)
            features = get_feature_boundaries(limit=limit)
            if features:
                self.db.save_feature_boundaries(features, repo_path)
            return features
        except Exception:
            return []
        finally:
            os.chdir(original_dir)

    def prepare_features_data_for_display(self, features: list) -> list[dict]:
        """Prepare feature boundaries data for display formatting."""
        features_data = []
        for feature in features:
            base_short = feature.base_commit[:8]
            head_short = feature.head_commit[:8]

            best_entry_id = self.db.get_best_user_story_entry_for_feature(feature)
            best_entry_short = best_entry_id[:8] if best_entry_id else "N/A"

            features_data.append(
                {
                    "feature_id": feature.short_id,
                    "feature_message": feature.feature_message,
                    "commits_display": f"{base_short} → {head_short}",
                    "merge_message": feature.merge_message,
                    "best_entry_id": best_entry_short,
                }
            )
        return features_data

    def get_commit_info_for_display(self, base_commit: str, head_commit: str) -> dict:
        """Get commit information for display purposes."""
        try:
            resolved_base = resolve_commit_reference(base_commit)
            resolved_head = resolve_commit_reference(head_commit)
            stride_size = calculate_stride_size(resolved_base, resolved_head)

            return {
                "base_display": f"{base_commit} → {resolved_base[:8]}...",
                "head_display": f"{head_commit} → {resolved_head[:8]}...",
                "stride_size": stride_size,
                "resolved_base": resolved_base,
                "resolved_head": resolved_head,
            }
        except Exception:
            return {
                "base_display": base_commit,
                "head_display": head_commit,
                "stride_size": 1,
                "resolved_base": base_commit,
                "resolved_head": head_commit,
            }

    def get_configured_agent(self) -> str:
        """Get the configured user story agent name."""
        return settings.user_story_agent
