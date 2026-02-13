"""Language-related models for complexity analysis features."""

from enum import Enum

from pydantic import BaseModel, Field


class ProjectLanguage(str, Enum):
    """Supported languages for complexity analysis."""

    PYTHON = "python"
    RUST = "rust"


class LanguageGuardResult(BaseModel):
    """Result of language guard check for complexity analysis features."""

    allowed: bool = Field(description="Whether the required language is available for analysis")
    required_language: ProjectLanguage = Field(description="The language required by the feature")
    detected_supported: set[ProjectLanguage] = Field(
        default_factory=set, description="Languages detected in repo that are supported"
    )
    detected_unsupported: set[str] = Field(
        default_factory=set, description="Language names detected but not supported (e.g., 'Rust', 'Go')"
    )

    def format_warning(self) -> str | None:
        """Return warning message if unsupported languages found, else None."""
        if not self.detected_unsupported:
            return None
        return f"Found {', '.join(sorted(self.detected_unsupported))} files but analysis not yet supported"
