"""Language detection for repository analysis."""

import logging
import subprocess
from pathlib import Path

from slopometry.core.models.hook import ProjectLanguage

logger = logging.getLogger(__name__)

# Map file extensions to supported ProjectLanguage
EXTENSION_MAP: dict[str, ProjectLanguage] = {
    ".py": ProjectLanguage.PYTHON,
    ".rs": ProjectLanguage.RUST,
}

# Extensions we recognize but don't support yet (for explicit warnings)
KNOWN_UNSUPPORTED_EXTENSIONS: dict[str, str] = {
    ".go": "Go",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
}


class LanguageDetector:
    """Detect programming languages present in a git repository."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def detect_languages(self) -> tuple[set[ProjectLanguage], set[str]]:
        """Detect languages by scanning git-tracked files.

        Returns:
            Tuple of (supported_languages, unsupported_language_names)
            - supported_languages: Set of ProjectLanguage enums found
            - unsupported_language_names: Set of language names found but not supported
        """
        tracked_files = self._get_tracked_files()

        supported: set[ProjectLanguage] = set()
        unsupported: set[str] = set()

        for file_path in tracked_files:
            ext = Path(file_path).suffix.lower()

            if ext in EXTENSION_MAP:
                supported.add(EXTENSION_MAP[ext])
            elif ext in KNOWN_UNSUPPORTED_EXTENSIONS:
                unsupported.add(KNOWN_UNSUPPORTED_EXTENSIONS[ext])

        return supported, unsupported

    def _get_tracked_files(self) -> list[str]:
        """Get list of git-tracked files in the repository."""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return []

            return [line for line in result.stdout.strip().split("\n") if line]

        except subprocess.TimeoutExpired:
            logger.warning("Language detection timed out for %s", self.repo_path)
            return []
        except FileNotFoundError:
            logger.debug("git not found, cannot detect languages in %s", self.repo_path)
            return []
