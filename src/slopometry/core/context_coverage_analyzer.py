"""Context coverage analyzer for tracking what files Claude read before editing."""

import ast
import json
from pathlib import Path

from slopometry.core.models import ContextCoverage, FileCoverageStatus


class ContextCoverageAnalyzer:
    """Analyzes session transcripts to determine if sufficient context was read before edits."""

    def __init__(self, working_directory: Path):
        """Initialize the analyzer.

        Args:
            working_directory: Root directory of the project being analyzed
        """
        self.working_directory = working_directory
        self._import_graph: dict[str, set[str]] = {}
        self._reverse_import_graph: dict[str, set[str]] = {}

    def analyze_transcript(self, transcript_path: Path) -> ContextCoverage:
        """Analyze a session transcript to compute context coverage.

        Args:
            transcript_path: Path to the Claude Code transcript JSONL file

        Returns:
            ContextCoverage with coverage metrics for all edited files
        """
        files_read, files_edited, read_timestamps, edit_timestamps = self._extract_file_events(transcript_path)

        self._build_import_graph()

        file_coverage = []
        all_blind_spots: set[str] = set()

        for edited_file in files_edited:
            coverage = self._compute_file_coverage(edited_file, files_read, read_timestamps, edit_timestamps)
            file_coverage.append(coverage)

            unread_imports = set(coverage.imports) - set(coverage.imports_read)
            unread_dependents = set(coverage.dependents) - set(coverage.dependents_read)
            unread_tests = set(coverage.test_files) - set(coverage.test_files_read)
            all_blind_spots.update(unread_imports)
            all_blind_spots.update(unread_dependents)
            all_blind_spots.update(unread_tests)

        return ContextCoverage(
            files_edited=list(files_edited),
            files_read=list(files_read),
            file_coverage=file_coverage,
            blind_spots=sorted(all_blind_spots),
        )

    def _extract_file_events(self, transcript_path: Path) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
        """Extract Read and Edit file paths from transcript with their sequence numbers.

        Args:
            transcript_path: Path to transcript JSONL

        Returns:
            Tuple of (files_read, files_edited, read_timestamps, edit_timestamps)
            Timestamps are sequence numbers for ordering.
        """
        files_read: set[str] = set()
        files_edited: set[str] = set()
        read_timestamps: dict[str, int] = {}
        edit_timestamps: dict[str, int] = {}
        sequence = 0

        try:
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    sequence += 1
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    tool_name = self._get_tool_name(event)
                    if not tool_name:
                        continue

                    file_path = self._extract_file_path(event, tool_name)
                    if not file_path:
                        continue

                    relative_path = self._to_relative_path(file_path)
                    if not relative_path:
                        continue

                    if not relative_path.endswith(".py"):
                        continue

                    if tool_name == "Read":
                        files_read.add(relative_path)
                        if relative_path not in read_timestamps:
                            read_timestamps[relative_path] = sequence
                    elif tool_name in ("Edit", "Write", "MultiEdit"):
                        files_edited.add(relative_path)
                        if relative_path not in edit_timestamps:
                            edit_timestamps[relative_path] = sequence

        except (OSError, json.JSONDecodeError):
            pass

        return files_read, files_edited, read_timestamps, edit_timestamps

    def _get_tool_name(self, event: dict) -> str | None:
        """Extract tool name from transcript event."""
        if "tool_name" in event:
            return event["tool_name"]
        if "message" in event and "content" in event["message"]:
            for content in event["message"]["content"]:
                if isinstance(content, dict) and content.get("type") == "tool_use":
                    return content.get("name")
        return None

    def _extract_file_path(self, event: dict, tool_name: str) -> str | None:
        """Extract file path from event based on tool type."""
        tool_input = event.get("tool_input", {})
        if not tool_input:
            if "message" in event and "content" in event["message"]:
                for content in event["message"]["content"]:
                    if (
                        isinstance(content, dict)
                        and content.get("type") == "tool_use"
                        and content.get("name") == tool_name
                    ):
                        tool_input = content.get("input", {})
                        break

        if tool_name == "Read":
            return tool_input.get("file_path")
        elif tool_name in ("Edit", "Write", "MultiEdit"):
            return tool_input.get("file_path")
        return None

    def _to_relative_path(self, file_path: str) -> str | None:
        """Convert absolute path to relative path from working directory."""
        try:
            abs_path = Path(file_path).resolve()
            wd_resolved = self.working_directory.resolve()

            if abs_path.is_relative_to(wd_resolved):
                return str(abs_path.relative_to(wd_resolved))
        except (ValueError, OSError):
            pass
        return None

    def _build_import_graph(self) -> None:
        """Build import graph for all Python files in working directory."""
        from slopometry.core.git_tracker import GitTracker

        tracker = GitTracker(self.working_directory)
        python_files = tracker.get_tracked_python_files()

        for file_path in python_files:
            try:
                relative_path = str(file_path.relative_to(self.working_directory))
                imports = self._extract_imports(file_path)
                self._import_graph[relative_path] = imports

                for imported in imports:
                    if imported not in self._reverse_import_graph:
                        self._reverse_import_graph[imported] = set()
                    self._reverse_import_graph[imported].add(relative_path)

            except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
                continue

    def _extract_imports(self, file_path: Path) -> set[str]:
        """Extract local project imports from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Set of relative paths to imported local modules
        """
        imports: set[str] = set()
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except Exception:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_import(alias.name)
                    if resolved:
                        imports.add(resolved)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    resolved = self._resolve_import(node.module)
                    if resolved:
                        imports.add(resolved)

        return imports

    def _resolve_import(self, module_name: str) -> str | None:
        """Resolve a module name to a relative file path if it exists in the project.

        Args:
            module_name: Dotted module name (e.g., 'slopometry.core.models')

        Returns:
            Relative path to the module file, or None if not found
        """
        parts = module_name.split(".")

        base_paths = [Path("."), Path("src")]
        for base in base_paths:
            possible_paths = [
                base / Path(*parts).with_suffix(".py"),
                base / Path(*parts) / "__init__.py",
            ]

            for rel_path in possible_paths:
                full_path = self.working_directory / rel_path
                if full_path.exists():
                    return str(rel_path)

        if len(parts) > 1:
            for i in range(len(parts) - 1, 0, -1):
                sub_parts = parts[:i]
                for base in base_paths:
                    for rel_path in [
                        base / Path(*sub_parts).with_suffix(".py"),
                        base / Path(*sub_parts) / "__init__.py",
                    ]:
                        full_path = self.working_directory / rel_path
                        if full_path.exists():
                            return str(rel_path)

        return None

    def _find_test_files(self, source_file: str) -> list[str]:
        """Find test files related to a source file.

        Args:
            source_file: Relative path to source file

        Returns:
            List of relative paths to related test files
        """
        source_path = Path(source_file)
        source_name = source_path.stem

        patterns = [
            f"tests/test_{source_name}.py",
            f"test/test_{source_name}.py",
            f"tests/{source_name}_test.py",
            f"test_{source_name}.py",
        ]

        test_files = []
        for pattern in patterns:
            test_path = self.working_directory / pattern
            if test_path.exists():
                test_files.append(pattern)

        tests_dir = self.working_directory / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob(f"*{source_name}*.py"):
                rel_path = str(test_file.relative_to(self.working_directory))
                if rel_path not in test_files:
                    test_files.append(rel_path)

        return test_files

    def _compute_file_coverage(
        self,
        edited_file: str,
        files_read: set[str],
        read_timestamps: dict[str, int],
        edit_timestamps: dict[str, int],
    ) -> FileCoverageStatus:
        """Compute coverage status for a single edited file.

        Args:
            edited_file: Relative path to edited file
            files_read: Set of all files that were read
            read_timestamps: Sequence numbers when files were first read
            edit_timestamps: Sequence numbers when files were first edited

        Returns:
            FileCoverageStatus for the edited file
        """
        edit_time = edit_timestamps.get(edited_file, float("inf"))
        was_read_before = edited_file in read_timestamps and read_timestamps[edited_file] < edit_time

        imports = sorted(self._import_graph.get(edited_file, set()))
        imports_read = [f for f in imports if f in files_read]

        dependents = sorted(self._reverse_import_graph.get(edited_file, set()))
        dependents_read = [f for f in dependents if f in files_read]

        test_files = self._find_test_files(edited_file)
        test_files_read = [f for f in test_files if f in files_read]

        return FileCoverageStatus(
            file_path=edited_file,
            was_read_before_edit=was_read_before,
            imports=imports,
            imports_read=imports_read,
            dependents=dependents,
            dependents_read=dependents_read,
            test_files=test_files,
            test_files_read=test_files_read,
        )


def analyze_context_coverage(transcript_path: Path, working_directory: Path) -> ContextCoverage:
    """Convenience function to analyze context coverage from a transcript.

    Args:
        transcript_path: Path to Claude Code transcript JSONL
        working_directory: Root directory of the project

    Returns:
        ContextCoverage with metrics
    """
    analyzer = ContextCoverageAnalyzer(working_directory)
    return analyzer.analyze_transcript(transcript_path)
