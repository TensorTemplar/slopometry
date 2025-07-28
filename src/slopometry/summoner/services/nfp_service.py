"""NFP (Next Feature Prediction) service for summoner features."""

from slopometry.core.database import EventDatabase
from slopometry.core.models import NextFeaturePrediction


class NFPService:
    """Handles NFP objective management for summoner users."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def list_nfp_objectives(self, repo_filter: str | None = None) -> list[NextFeaturePrediction]:
        """List all NFP objectives, optionally filtered by repository."""
        try:
            return self.db.list_nfp_objectives(repo_filter)
        except Exception:
            return []

    def get_nfp_objective(self, nfp_id: str) -> NextFeaturePrediction | None:
        """Get detailed information for an NFP objective."""
        try:
            return self.db.get_nfp_objective(nfp_id)
        except Exception:
            return None

    def delete_nfp_objective(self, nfp_id: str) -> bool:
        """Delete an NFP objective and all its user stories."""
        try:
            return self.db.delete_nfp_objective(nfp_id)
        except Exception:
            return False

    def prepare_objectives_data_for_display(self, objectives: list[NextFeaturePrediction]) -> list[dict]:
        """Prepare NFP objectives data for display formatting."""
        objectives_data = []
        for nfp in objectives:
            objectives_data.append(
                {
                    "id": nfp.id,
                    "title": nfp.title,
                    "commits": f"{nfp.base_commit} → {nfp.target_commit}",
                    "story_count": nfp.story_count,
                    "complexity": nfp.total_estimated_complexity,
                    "created_date": nfp.created_at.strftime("%Y-%m-%d"),
                }
            )
        return objectives_data

    def get_objective_summary(self, nfp: NextFeaturePrediction | None) -> dict:
        """Get summary information for an NFP objective."""
        if not nfp:
            return {}

        return {
            "id": nfp.id,
            "title": nfp.title,
            "description": nfp.description,
            "base_commit": nfp.base_commit,
            "target_commit": nfp.target_commit,
            "created_at": nfp.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": nfp.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "story_count": nfp.story_count,
            "total_estimated_complexity": nfp.total_estimated_complexity,
            "high_priority_stories_count": len(nfp.get_high_priority_stories()),
        }

    def get_stories_by_priority(self, nfp: NextFeaturePrediction | None, priority: int) -> list:
        """Get user stories for a specific priority level."""
        if not nfp:
            return []

        try:
            return nfp.get_stories_by_priority(priority)
        except Exception:
            return []

    def get_priority_name(self, priority: int) -> str:
        """Get human-readable priority name."""
        priority_names = {1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Nice to Have"}
        return priority_names.get(priority, "Unknown")
