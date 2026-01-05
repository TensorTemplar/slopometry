"""Language guard for complexity analysis features."""

from pathlib import Path

from slopometry.core.language_detector import LanguageDetector
from slopometry.core.models import LanguageGuardResult, ProjectLanguage


def check_language_support(
    repo_path: Path,
    required_language: ProjectLanguage,
) -> LanguageGuardResult:
    """Check if repository has required language for analysis.

    Returns LanguageGuardResult with:
    - allowed=True if required_language is detected in repo
    - Warning info about detected but unsupported languages
    """
    detector = LanguageDetector(repo_path)
    detected_supported, detected_unsupported = detector.detect_languages()

    return LanguageGuardResult(
        allowed=required_language in detected_supported,
        required_language=required_language,
        detected_supported=detected_supported,
        detected_unsupported=detected_unsupported,
    )
