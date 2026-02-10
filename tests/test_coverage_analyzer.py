"""Tests for CoverageAnalyzer - file-based coverage detection."""

import shutil
from pathlib import Path

from slopometry.core.coverage_analyzer import CoverageAnalyzer, CoverageResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCoverageResult:
    """Unit tests for CoverageResult model."""

    def test_coverage_result__default_values(self) -> None:
        """Test CoverageResult has sensible defaults."""
        result = CoverageResult()

        assert result.total_coverage_percent is None, "Should be N/A until calculated"
        assert result.coverage_available is True
        assert result.error_message is None
        assert result.source_file is None
        assert result.num_statements == 0

    def test_coverage_result__with_error(self) -> None:
        """Test CoverageResult can represent an error state."""
        result = CoverageResult(
            coverage_available=False,
            error_message="No coverage files found",
        )

        assert result.coverage_available is False
        assert result.error_message == "No coverage files found"

    def test_coverage_result__with_source(self) -> None:
        """Test CoverageResult includes source file info."""
        result = CoverageResult(
            total_coverage_percent=85.5,
            coverage_available=True,
            source_file="coverage.xml",
        )

        assert result.source_file == "coverage.xml"
        assert result.total_coverage_percent == 85.5


class TestCoverageAnalyzerXML:
    """Tests for parsing coverage.xml files using real fixture."""

    def test_analyze_coverage__parses_real_xml_fixture(self, tmp_path: Path) -> None:
        """Test parsing real coverage.xml from this repository."""
        fixture_xml = FIXTURES_DIR / "coverage.xml"
        assert fixture_xml.exists(), f"coverage.xml fixture not found at {fixture_xml}"

        shutil.copy(fixture_xml, tmp_path / "coverage.xml")

        analyzer = CoverageAnalyzer(tmp_path)
        result = analyzer.analyze_coverage()

        assert result.coverage_available is True
        assert result.source_file == "coverage.xml"
        # Real fixture: line-rate="0.4487", lines-valid="4230", lines-covered="1898"
        assert result.total_coverage_percent == 44.87
        assert result.num_statements == 4230
        assert result.covered_statements == 1898
        assert result.missing_statements == 4230 - 1898
        assert result.files_analyzed > 0

    def test_analyze_coverage__parses_flat_xml(self, tmp_path: Path) -> None:
        """Test parsing flattened coverage.xml (no packages)."""
        fixture_xml = FIXTURES_DIR / "coverage_flat.xml"
        assert fixture_xml.exists(), f"coverage_flat.xml fixture not found at {fixture_xml}"

        shutil.copy(fixture_xml, tmp_path / "coverage.xml")

        analyzer = CoverageAnalyzer(tmp_path)
        result = analyzer.analyze_coverage()

        assert result.coverage_available is True
        assert result.source_file == "coverage.xml"
        assert result.total_coverage_percent == 80.0
        assert result.num_statements == 10
        assert result.covered_statements == 8
        assert result.files_analyzed == 2

    def test_analyze_coverage__handles_malformed_xml(self, tmp_path: Path) -> None:
        """Test graceful handling of malformed XML."""
        coverage_xml = tmp_path / "coverage.xml"
        coverage_xml.write_text("not valid xml <unclosed")

        analyzer = CoverageAnalyzer(tmp_path)
        result = analyzer.analyze_coverage()

        assert result.coverage_available is False
        assert result.error_message is not None
        assert "parse" in result.error_message.lower()


class TestCoverageAnalyzerDB:
    """Tests for parsing .coverage SQLite database using real fixture."""

    def test_analyze_coverage__parses_real_db_fixture(self, tmp_path: Path) -> None:
        """Test parsing .coverage database by generating one dynamically."""
        # Create a python file to cover
        source_file = tmp_path / "foo.py"
        source_file.write_text("def foo():\n    return 1\n\nfoo()\n")

        # Generate coverage data using subprocess to avoid conflicts with pytest-cov
        import subprocess
        import sys

        # We need to run coverage in a way that generates the .coverage file in tmp_path
        subprocess.run(
            [sys.executable, "-m", "coverage", "run", "--data-file=.coverage", "foo.py"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        analyzer = CoverageAnalyzer(tmp_path)
        result = analyzer.analyze_coverage()

        assert result.coverage_available is True
        assert result.source_file == ".coverage"
        assert result.total_coverage_percent == 100.0
        assert result.files_analyzed == 1

    def test_analyze_coverage__handles_corrupt_db(self, tmp_path: Path) -> None:
        """Test graceful handling when .coverage file is corrupt."""
        coverage_db = tmp_path / ".coverage"
        coverage_db.write_text("not a valid sqlite database")

        analyzer = CoverageAnalyzer(tmp_path)
        result = analyzer.analyze_coverage()

        assert result.coverage_available is False
        assert result.error_message is not None


class TestCoverageAnalyzerNoFiles:
    """Tests for when no coverage files exist."""

    def test_analyze_coverage__no_files_found(self, tmp_path: Path) -> None:
        """Test that appropriate message is returned when no coverage files exist."""
        analyzer = CoverageAnalyzer(tmp_path)
        result = analyzer.analyze_coverage()

        assert result.coverage_available is False
        assert result.error_message is not None
        assert "No coverage files found" in result.error_message
        assert "pytest" in result.error_message.lower()


class TestCoverageAnalyzerFilePriority:
    """Tests for file priority (XML preferred over DB)."""

    def test_analyze_coverage__prefers_xml_over_db(self, tmp_path: Path) -> None:
        """Test that coverage.xml is preferred over .coverage when both exist."""
        # Copy both real fixtures
        fixture_xml = FIXTURES_DIR / "coverage.xml"
        fixture_db = FIXTURES_DIR / ".coverage"

        assert fixture_xml.exists(), f"coverage.xml fixture not found at {fixture_xml}"
        assert fixture_db.exists(), f".coverage fixture not found at {fixture_db}"

        shutil.copy(fixture_xml, tmp_path / "coverage.xml")
        shutil.copy(fixture_db, tmp_path / ".coverage")

        analyzer = CoverageAnalyzer(tmp_path)
        result = analyzer.analyze_coverage()

        # Should use XML, not DB
        assert result.source_file == "coverage.xml"
        assert result.coverage_available is True
