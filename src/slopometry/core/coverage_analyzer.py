"""Test coverage analysis by detecting existing coverage files."""

import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

from coverage import Coverage
from pydantic import BaseModel, Field


class CoverageResult(BaseModel):
    """Result of coverage analysis from existing files."""

    total_coverage_percent: float = Field(default=0.0, description="Total test coverage percentage (0-100)")
    num_statements: int = Field(default=0, description="Total number of statements")
    covered_statements: int = Field(default=0, description="Number of covered statements")
    missing_statements: int = Field(default=0, description="Number of missing statements")
    files_analyzed: int = Field(default=0, description="Number of Python files analyzed")
    coverage_available: bool = Field(default=True, description="Whether coverage data was available")
    source_file: str | None = Field(
        default=None, description="Which file the coverage was read from (e.g., 'coverage.xml')"
    )
    error_message: str | None = Field(default=None, description="Error message if coverage failed")


class CoverageAnalyzer:
    """Analyzes test coverage by detecting existing coverage files.

    Looks for coverage files in order of preference:
    1. coverage.xml - Cobertura XML format (easy to parse)
    2. .coverage - SQLite database (requires coverage library)
    """

    def __init__(self, repo_path: Path):
        """Initialize the coverage analyzer.

        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = repo_path.resolve()

    def analyze_coverage(self) -> CoverageResult:
        """Detect and parse existing coverage files.

        Returns:
            CoverageResult with coverage data from existing files
        """
        coverage_xml = self.repo_path / "coverage.xml"
        if coverage_xml.exists():
            return self._parse_coverage_xml(coverage_xml)

        coverage_db = self.repo_path / ".coverage"
        if coverage_db.exists():
            return self._parse_coverage_db(coverage_db)

        return CoverageResult(
            coverage_available=False,
            error_message="No coverage files found. Run 'uv run pytest' first.",
        )

    def _parse_coverage_xml(self, xml_path: Path) -> CoverageResult:
        """Parse coverage.xml (Cobertura format).

        Args:
            xml_path: Path to coverage.xml file

        Returns:
            CoverageResult with parsed data
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            line_rate = float(root.get("line-rate", 0))
            lines_valid = int(root.get("lines-valid", 0))
            lines_covered = int(root.get("lines-covered", 0))

            files_count = len(root.findall(".//class"))

            return CoverageResult(
                total_coverage_percent=line_rate * 100,
                num_statements=lines_valid,
                covered_statements=lines_covered,
                missing_statements=lines_valid - lines_covered,
                files_analyzed=files_count,
                coverage_available=True,
                source_file="coverage.xml",
            )

        except ET.ParseError as e:
            return CoverageResult(
                coverage_available=False,
                error_message=f"Failed to parse coverage.xml: {e}",
            )
        except Exception as e:
            return CoverageResult(
                coverage_available=False,
                error_message=f"Error reading coverage.xml: {e}",
            )

    def _parse_coverage_db(self, db_path: Path) -> CoverageResult:
        """Parse .coverage SQLite database using coverage library.

        Args:
            db_path: Path to .coverage file

        Returns:
            CoverageResult with parsed data
        """
        try:
            cov = Coverage(data_file=str(db_path))
            cov.load()

            output = StringIO()
            try:
                total_percent = cov.report(file=output, show_missing=False)
            except Exception:
                total_percent = 0.0

            data = cov.get_data()
            files_count = len(data.measured_files())

            return CoverageResult(
                total_coverage_percent=total_percent,
                files_analyzed=files_count,
                coverage_available=True,
                source_file=".coverage",
            )

        except Exception as e:
            return CoverageResult(
                coverage_available=False,
                error_message=f"Error reading .coverage: {e}",
            )
