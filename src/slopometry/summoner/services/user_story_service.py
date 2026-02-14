"""User story management service for summoner features."""

import re
from pathlib import Path

import click

from slopometry.core.database import EventDatabase
from slopometry.core.models.user_story import UserStoryDisplayData, UserStoryEntry, UserStoryStatistics
from slopometry.core.settings import settings
from slopometry.display.console import console


class UserStoryService:
    """Handles user story management and export for summoner users."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def get_user_story_statistics(self) -> UserStoryStatistics:
        """Get comprehensive user story statistics."""
        try:
            stats_dict = self.db.get_user_story_stats()
            return UserStoryStatistics(**stats_dict)
        except Exception:
            return UserStoryStatistics(
                total_entries=0,
                avg_rating=0,
                unique_models=0,
                unique_repos=0,
                rating_distribution={},
            )

    def get_user_story_entries(self, limit: int = 10) -> list[UserStoryEntry]:
        """Get recent user story entries."""
        try:
            return self.db.get_user_story_entries(limit=limit)
        except Exception:
            return []

    def prepare_entries_data_for_display(self, entries: list[UserStoryEntry]) -> list[UserStoryDisplayData]:
        """Prepare user story entries for display formatting."""
        entries_data = []
        for entry in entries:
            entries_data.append(
                UserStoryDisplayData(
                    entry_id=entry.short_id,
                    date=entry.created_at.strftime("%Y-%m-%d %H:%M"),
                    commits=f"{entry.base_commit[:8]}→{entry.head_commit[:8]}",
                    rating=f"{entry.rating}/5",
                    model=entry.model_used,
                    repository=Path(entry.repository_path).name,
                )
            )
        return entries_data

    def export_user_stories(self, output_path: Path) -> int:
        """Export user stories to Parquet format."""
        try:
            return self.db.export_user_stories(output_path)
        except ImportError as e:
            raise ImportError(f"Missing required dependencies for export: {e}")

    def upload_to_huggingface(self, output_path: Path, hf_repo: str | None = None) -> str:
        """Upload user stories to Hugging Face."""
        if not hf_repo:
            if settings.hf_default_repo:
                hf_repo = settings.hf_default_repo
            else:
                project_name = Path.cwd().name.lower().replace("_", "-").replace(" ", "-")
                project_name = re.sub(r"[^a-z0-9-]", "", project_name)
                hf_repo = f"slopometry-{project_name}-userstories"

        try:
            from slopometry.summoner.services.hf_uploader import upload_to_huggingface

            upload_to_huggingface(output_path, hf_repo)
            return hf_repo
        except ImportError:
            raise ImportError(
                "Hugging Face datasets library not installed. Install with: pip install datasets huggingface-hub pandas pyarrow"
            )

    def filter_entries_for_rating(
        self, limit: int, filter_model: str | None = None, unrated_only: bool = False
    ) -> list[UserStoryEntry]:
        """Filter user story entries for rating workflow."""
        entries = self.get_user_story_entries(limit=limit * 2)

        if not entries:
            return []

        filtered_entries = []
        for entry in entries:
            if filter_model and entry.model_used != filter_model:
                continue
            if unrated_only and entry.rating != 3:
                continue
            filtered_entries.append(entry)

        return filtered_entries[:limit]

    def collect_user_rating_and_feedback(self) -> tuple[int, str]:
        """Collect rating and feedback from user for user story entry."""
        console.print("\n[bold yellow]User Story Collection[/bold yellow]")
        console.print("Please rate the generated user stories and provide feedback for improvement.")

        while True:
            try:
                rating = click.prompt(
                    "\nRate the quality of generated user stories (1-5, where 5 is excellent)", type=int
                )
                if 1 <= rating <= 5:
                    break
                else:
                    console.print("[red]Please enter a rating between 1 and 5[/red]")
            except (ValueError, click.Abort):
                console.print("[red]Please enter a valid number between 1 and 5[/red]")

        guidelines = click.prompt(
            "\nProvide guidelines for improving user story generation (optional)", default="", show_default=False
        )

        return rating, guidelines

    def rate_user_story_entry(self, entry: UserStoryEntry, new_rating: int, guidelines: str) -> None:
        """Update rating and guidelines for a user story entry."""
        entry.rating = new_rating
        entry.guidelines_for_improving = guidelines
        self.db.save_user_story_entry(entry)

    def save_user_story_entry(self, entry: UserStoryEntry) -> None:
        """Save a user story entry to the database."""
        self.db.save_user_story_entry(entry)
