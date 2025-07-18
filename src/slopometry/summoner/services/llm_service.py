"""LLM integration service for summoner features."""

import os
from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.models import DiffUserStoryDataset
from slopometry.core.settings import settings
from slopometry.summoner.services.llm_wrapper import (
    calculate_stride_size,
    cluade,
    gemini,
    get_commit_diff,
    get_feature_boundaries,
    get_user_story_prompt,
    resolve_commit_reference,
    user_story_agent,
)


class LLMService:
    """Handles AI/LLM integration for user story generation and analysis."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()
        self.agent_map = {
            "o3": user_story_agent,
            "claude-opus-4": cluade,
            "gemini-2.5-pro": gemini,
        }

    def generate_user_stories_from_commits(
        self, repo_path: Path, base_commit: str, head_commit: str
    ) -> tuple[int, list[str]]:
        """Generate user stories from commit diffs using configured AI agents.

        Returns:
            Tuple of (successful_generations, error_messages)
        """
        original_dir = os.getcwd()
        error_messages = []

        try:
            os.chdir(repo_path)

            resolved_base = resolve_commit_reference(base_commit)
            resolved_head = resolve_commit_reference(head_commit)

            stride_size = calculate_stride_size(resolved_base, resolved_head)

            diff = get_commit_diff(base_commit)

            if not diff:
                return 0, ["No changes found between commits"]

            prompt = get_user_story_prompt(diff)

            successful_generations = 0

            for model_name in settings.user_story_agents:
                if model_name not in self.agent_map:
                    error_messages.append(f"Unknown agent '{model_name}', skipping")
                    continue

                agent = self.agent_map[model_name]

                try:
                    result = agent.run_sync(prompt)

                    dataset_entry = DiffUserStoryDataset(
                        base_commit=resolved_base,
                        head_commit=resolved_head,
                        diff_content=diff,
                        stride_size=stride_size,
                        user_stories=result.data,
                        rating=3,  # Default neutral rating for bulk generation
                        guidelines_for_improving="",
                        model_used=model_name,
                        prompt_template=prompt,
                        repository_path=str(repo_path),
                    )

                    self.db.save_dataset_entry(dataset_entry)
                    successful_generations += 1

                except Exception as e:
                    error_messages.append(f"Failed to generate with {model_name}: {e}")
                    continue

            return successful_generations, error_messages

        except Exception as e:
            return 0, [f"Failed to generate user stories: {e}"]
        finally:
            os.chdir(original_dir)

    def get_feature_boundaries(self, repo_path: Path, limit: int = 20) -> list:
        """Get detected feature boundaries from merge commits."""
        original_dir = os.getcwd()
        try:
            os.chdir(repo_path)
            return get_feature_boundaries(limit=limit)
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
            features_data.append(
                {
                    "feature_message": feature.feature_message,
                    "commits_display": f"{base_short} → {head_short}",
                    "merge_message": feature.merge_message,
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

    def get_available_agents(self) -> list[str]:
        """Get list of available AI agents."""
        return list(self.agent_map.keys())

    def get_configured_agents(self) -> list[str]:
        """Get list of configured AI agents from settings."""
        return settings.user_story_agents
