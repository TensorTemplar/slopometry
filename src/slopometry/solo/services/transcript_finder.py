"""Transcript discovery for memory extraction."""

from dataclasses import dataclass
from pathlib import Path

from slopometry.core.models.protocol.events import AbstractEventSource
from slopometry.core.settings import get_claude_projects_dirs, get_opencode_storage_root


@dataclass(frozen=True)
class DiscoveredTranscript:
    """A session's transcript location, with provenance for downstream routing."""

    session_id: str
    transcript_path: Path
    project_dir: Path
    source: AbstractEventSource


class TranscriptFinder:
    """Finds Claude Code and OpenCode transcripts for memory extraction."""

    def find_claude_project_dirs(self) -> list[Path]:
        """Find Claude Code project directories based on platform."""
        return get_claude_projects_dirs()

    def _decode_claude_project_dir(self, dirname: str) -> Path | None:
        """Decode Claude project directory name back to working directory.

        Claude encodes path separators as hyphens, so e.g.
        -mnt-terradump-code-slopometry -> /mnt/terradump/code/slopometry.
        Hyphens in directory names are double-escaped as ``--``.
        """
        if not dirname.startswith("-"):
            return None
        raw = dirname[1:]
        parts = raw.split("-")
        decoded_parts: list[str] = []
        i = 0
        while i < len(parts):
            if not parts[i] and i + 1 < len(parts):
                decoded_parts.append("-")
                i += 2
            else:
                decoded_parts.append(parts[i])
                i += 1
        decoded = "/" + "/".join(decoded_parts)
        try:
            return Path(decoded).resolve()
        except Exception:  # slopometry: allow-silent - corrupt symlink or permissions in project dir; skip gracefully
            return None

    def find_slopometry_transcripts(self, project_dir: Path) -> list[DiscoveredTranscript]:
        """Find transcripts saved by slopometry in a project.

        Returns:
            List of DiscoveredTranscript with source=claude_code (slopometry's
            Claude Code harness produces these)
        """
        slop_dir = project_dir / ".slopometry"
        if not slop_dir.exists():
            return []

        results: list[DiscoveredTranscript] = []
        for session_dir in slop_dir.iterdir():
            if not session_dir.is_dir():
                continue
            transcript_path = session_dir / "transcript.jsonl"
            if transcript_path.exists():
                results.append(
                    DiscoveredTranscript(
                        session_id=session_dir.name,
                        transcript_path=transcript_path,
                        project_dir=project_dir,
                        source=AbstractEventSource.CLAUDE_CODE,
                    )
                )

        return results

    def find_opencode_storage_root(self) -> Path:
        """Find OpenCode's storage root directory.

        Layout: ``<root>/project/<id>.json``,
        ``<root>/session/<project_id>/<session_id>.json``,
        ``<root>/message/<session_id>/<message_id>.json``,
        ``<root>/part/<message_id>/<part_id>.json``.
        """
        return get_opencode_storage_root()

    def find_opencode_sessions(self, project_dir: Path) -> list[DiscoveredTranscript]:
        """Find OpenCode sessions whose working directory matches project_dir.

        A session matches if its ``project.worktree`` equals ``project_dir``
        OR its ``session.directory`` equals ``project_dir`` or is a descendant
        of it. This covers the case where a session was opened from a
        subdirectory of a registered worktree.

        Returns:
            List of DiscoveredTranscript with source=opencode and
            transcript_path pointing at the session file (the extractor walks
            message/part directories from there).
        """
        import json

        storage_root = self.find_opencode_storage_root()
        if not storage_root.is_dir():
            return []

        project_dir_resolved = project_dir.resolve()
        results: list[DiscoveredTranscript] = []

        project_dir_path = storage_root / "project"
        if not project_dir_path.is_dir():
            return results

        def _directory_matches_project(directory_str: str | None) -> bool:
            """True if ``directory_str`` equals ``project_dir_resolved`` or either is inside the other."""
            if not directory_str:
                return False
            try:
                directory_resolved = Path(directory_str).resolve()
            except OSError:
                return False
            if directory_resolved == project_dir_resolved:
                return True
            try:
                directory_resolved.relative_to(project_dir_resolved)
                return True
            except ValueError:
                pass
            try:
                project_dir_resolved.relative_to(directory_resolved)
                return True
            except ValueError:
                return False

        for project_file in project_dir_path.glob("*.json"):
            try:
                project_meta = json.loads(project_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            worktree = project_meta.get("worktree")
            if not _directory_matches_project(worktree):
                continue

            project_id = project_meta.get("id")
            if not project_id:
                continue

            session_dir = storage_root / "session" / project_id
            if not session_dir.is_dir():
                continue

            for session_file in session_dir.glob("ses_*.json"):
                try:
                    session_meta = json.loads(session_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                session_id = session_meta.get("id")
                if not session_id:
                    continue
                if not _directory_matches_project(session_meta.get("directory")):
                    continue
                results.append(
                    DiscoveredTranscript(
                        session_id=session_id,
                        transcript_path=session_file,
                        project_dir=project_dir,
                        source=AbstractEventSource.OPENCODE,
                    )
                )

        return results

    def discover_transcripts(self, project_dir: Path) -> list[DiscoveredTranscript]:
        """Discover all transcripts for a project, across all harnesses.

        Returns:
            List of DiscoveredTranscript, one per session per harness, deduplicated
            when the same session appears across multiple discovery sources.
        """
        results: list[DiscoveredTranscript] = []
        seen: set[tuple[str, str]] = set()
        project_dir = project_dir.resolve()

        for claude_projects_dir in self.find_claude_project_dirs():
            if not claude_projects_dir.exists():
                continue

            for project_subdir in claude_projects_dir.iterdir():
                if not project_subdir.is_dir():
                    continue

                working_dir = self._decode_claude_project_dir(project_subdir.name)
                if working_dir != project_dir:
                    continue

                for transcript_path in project_subdir.glob("*.jsonl"):
                    session_id = transcript_path.stem
                    key = (session_id, AbstractEventSource.CLAUDE_CODE.value)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        DiscoveredTranscript(
                            session_id=session_id,
                            transcript_path=transcript_path,
                            project_dir=project_dir,
                            source=AbstractEventSource.CLAUDE_CODE,
                        )
                    )

        for t in self.find_slopometry_transcripts(project_dir):
            key = (t.session_id, t.source.value)
            if key in seen:
                continue
            seen.add(key)
            results.append(t)

        for t in self.find_opencode_sessions(project_dir):
            key = (t.session_id, t.source.value)
            if key in seen:
                continue
            seen.add(key)
            results.append(t)

        return results
