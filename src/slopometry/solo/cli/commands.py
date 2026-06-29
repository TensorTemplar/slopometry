"""CLI commands for solo-leveler features."""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import click

from slopometry.core.models.memory import FreshnessAction
from slopometry.display.console import console, styled_pager
from slopometry.solo.services.memory_freshness import audit_staleness

if TYPE_CHECKING:
    from slopometry.core.models import ImpactAssessment, RepoBaseline, SessionStatistics
    from slopometry.core.models.session import BehavioralPatternTrends

logger = logging.getLogger(__name__)


def complete_session_id(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[str]:
    """Complete session IDs, filtered by current repository if in a git repo."""
    from slopometry.core.git_tracker import GitTracker
    from slopometry.solo.services.session_service import SessionService

    try:
        session_service = SessionService()
        cwd = Path.cwd()
        git_tracker = GitTracker(cwd)
        git_state = git_tracker.get_git_state()

        if git_state.is_git_repo:
            sessions = session_service.list_sessions_by_repository(cwd)
        else:
            sessions = session_service.list_sessions()

        return [session for session in sessions if session.startswith(incomplete)]
    except Exception:
        return []


@click.group()
def solo() -> None:
    """Solo-leveler commands for basic session tracking."""
    pass


def _warn_if_not_in_path() -> None:
    """Print a warning if slopometry is not in PATH."""
    import shutil

    if not shutil.which("slopometry"):
        console.print("\n[yellow]Warning: 'slopometry' is not in your PATH.[/yellow]")
        console.print("[yellow]Run 'uv tool update-shell' and restart your terminal to fix this.[/yellow]")


@click.command()
@click.option(
    "--global/--local",
    "global_",
    default=False,
    help="Install hooks globally (~/.claude) or locally (./.claude)",
)
@click.option(
    "--target",
    type=click.Choice(["claude-code", "opencode"]),
    default="claude-code",
    help="Target tool to install hooks for (default: claude-code)",
)
def install(global_: bool, target: str) -> None:
    """Install slopometry hooks to automatically track all sessions and tool usage."""
    from slopometry.core.settings import get_default_config_dir, get_default_data_dir
    from slopometry.solo.services.hook_service import HookService

    hook_service = HookService()

    if target == "opencode":
        success, message = hook_service.install_opencode()
        if success:
            for line in message.split("\n"):
                console.print(f"[green]{line}[/green]")
            console.print("[cyan]All OpenCode sessions will now be automatically tracked[/cyan]")
            console.print(f"[dim]Data: {get_default_data_dir()}[/dim]")
            _warn_if_not_in_path()
        else:
            console.print(f"[red]{message}[/red]")
        return

    success, message = hook_service.install_hooks(global_)

    if success:
        for line in message.split("\n"):
            console.print(f"[green]{line}[/green]")
        console.print("[cyan]All Claude Code sessions will now be automatically tracked[/cyan]")
        console.print(f"[dim]Config: {get_default_config_dir()}[/dim]")
        console.print(f"[dim]Data:   {get_default_data_dir()}[/dim]")
        _warn_if_not_in_path()
    else:
        console.print(f"[red]{message}[/red]")


@click.command()
@click.option(
    "--global/--local",
    "global_",
    default=False,
    help="Remove hooks globally (~/.claude) or locally (./.claude)",
)
@click.option(
    "--target",
    type=click.Choice(["claude-code", "opencode"]),
    default="claude-code",
    help="Target tool to remove hooks from (default: claude-code)",
)
def uninstall(global_: bool, target: str) -> None:
    """Remove slopometry hooks to completely stop automatic session tracking."""
    from slopometry.solo.services.hook_service import HookService

    hook_service = HookService()

    if target == "opencode":
        success, message = hook_service.uninstall_opencode()
    else:
        success, message = hook_service.uninstall_hooks(global_)

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")


@solo.command("ls")
@click.option("--limit", default=None, type=int, help="Number of recent sessions to show")
@click.option("--all", "show_all", is_flag=True, help="Show sessions from all projects (default: current project only)")
@click.option("--pager/--no-pager", default=True, help="Use pager for long output (like less)")
def list_sessions(limit: int, show_all: bool, pager: bool) -> None:
    """List recent sessions (filtered to current project by default)."""
    from slopometry.display.formatters import create_sessions_table
    from slopometry.solo.services.session_service import SessionService

    session_service = SessionService()

    working_directory = None
    if not show_all:
        working_directory = str(Path.cwd().resolve())

    sessions_data = session_service.get_sessions_for_display(limit=limit, working_directory=working_directory)

    if not sessions_data:
        if working_directory:
            console.print(f"[yellow]No sessions found for {working_directory}[/yellow]")
            console.print("[dim]Use --all to show sessions from all projects[/dim]")
        else:
            console.print("[yellow]No sessions found[/yellow]")
            console.print("[dim]Run 'slopometry install' to start tracking sessions[/dim]")
        return

    table = create_sessions_table(sessions_data)
    if pager:
        with console.pager(pager=styled_pager, styles=True):
            console.print(table)
    else:
        console.print(table)


@solo.command()
@click.argument("session_id", shell_complete=complete_session_id)
@click.option("--smell-details", is_flag=True, help="Show files affected by each code smell")
@click.option("--file-details", is_flag=True, help="Show full file lists in delta sections")
@click.option("--pager/--no-pager", default=True, help="Use pager for long output (like less)")
def show(session_id: str, smell_details: bool, file_details: bool, pager: bool) -> None:
    """Show detailed statistics for a session."""
    import time

    from slopometry.display.formatters import display_session_summary
    from slopometry.solo.services.session_service import SessionService

    start_time = time.perf_counter()

    session_service = SessionService()
    stats = session_service.get_session_statistics(session_id)

    if not stats:
        console.print(f"[red]No data found for session {session_id}[/red]")
        return

    from slopometry.core.database import EventDatabase

    baseline, assessment = _compute_session_baseline(stats)
    source = EventDatabase().get_session_source(session_id)

    _persist_behavioral_patterns(session_id, stats)
    behavioral_trends = _compute_behavioral_trends(session_id, stats)

    def _display() -> None:
        assert stats is not None
        display_session_summary(
            stats,
            session_id,
            baseline,
            assessment,
            show_smell_files=smell_details,
            show_file_details=file_details,
            source=source,
            behavioral_trends=behavioral_trends,
        )

        elapsed = time.perf_counter() - start_time
        if elapsed > 5:
            console.print(f"\n[dim]Analysis completed in {elapsed:.1f}s[/dim]")

    if pager:
        with console.pager(pager=styled_pager, styles=True):
            _display()
    else:
        _display()


@click.command()
@click.option("--smell-details", is_flag=True, help="Show files affected by each code smell")
@click.option("--file-details", is_flag=True, help="Show full file lists in delta sections")
@click.option("--pager/--no-pager", default=True, help="Use pager for long output (like less)")
def latest(smell_details: bool, file_details: bool, pager: bool) -> None:
    """Show detailed statistics for the most recent session."""
    import time

    from slopometry.display.formatters import display_session_summary
    from slopometry.solo.services.session_service import SessionService

    start_time = time.perf_counter()

    session_service = SessionService()
    most_recent = session_service.get_most_recent_session()

    if not most_recent:
        console.print("[red]No sessions found[/red]")
        return

    console.print(f"[bold]Showing most recent session: {most_recent}[/bold]\n")
    stats = session_service.get_session_statistics(most_recent)
    if stats:
        if stats.complexity_metrics and stats.working_directory:
            working_dir = Path(stats.working_directory)
            if working_dir.exists():
                from slopometry.core.project_guard import MultiProjectError, guard_single_project

                try:
                    guard_single_project(working_dir)
                except MultiProjectError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    return

                # Always fetch fresh coverage from coverage.xml (don't use stale cached value)
                try:
                    from slopometry.core.coverage_analyzer import CoverageAnalyzer

                    coverage_analyzer = CoverageAnalyzer(working_dir)
                    coverage_result = coverage_analyzer.analyze_coverage()

                    if coverage_result.coverage_available:
                        stats.complexity_metrics.test_coverage_percent = coverage_result.total_coverage_percent
                        stats.complexity_metrics.test_coverage_source = coverage_result.source_file

                        try:
                            from slopometry.core.code_quality_cache import CodeQualityCacheManager
                            from slopometry.core.database import EventDatabase

                            db = EventDatabase()
                            with db._get_db_connection() as conn:
                                cache_manager = CodeQualityCacheManager(conn)
                                cache_manager.update_cached_coverage(
                                    most_recent,
                                    coverage_result.total_coverage_percent,
                                    coverage_result.source_file or "",
                                )
                        except Exception as cache_err:
                            logger.debug(f"Failed to cache coverage result: {cache_err}")
                except Exception as e:
                    logger.debug(f"Coverage analysis failed (optional): {e}")

        baseline, assessment = _compute_session_baseline(stats)

        from slopometry.core.database import EventDatabase

        source = EventDatabase().get_session_source(most_recent)

        _persist_behavioral_patterns(most_recent, stats)
        behavioral_trends = _compute_behavioral_trends(most_recent, stats)

        def _display() -> None:
            assert stats is not None and most_recent is not None
            display_session_summary(
                stats,
                most_recent,
                baseline,
                assessment,
                show_smell_files=smell_details,
                show_file_details=file_details,
                source=source,
                behavioral_trends=behavioral_trends,
            )

            elapsed = time.perf_counter() - start_time
            if elapsed > 5:
                console.print(f"\n[dim]Analysis completed in {elapsed:.1f}s[/dim]")

        if pager:
            with console.pager(pager=styled_pager, styles=True):
                _display()
        else:
            _display()


def _compute_behavioral_trends(session_id: str, stats: "SessionStatistics") -> "BehavioralPatternTrends | None":
    """Compute rolling average trends from historical behavioral pattern data."""
    from slopometry.core.models.session import BehavioralPatternTrends

    if not stats.behavioral_patterns or not stats.working_directory:
        return None
    try:
        from slopometry.core.database import EventDatabase
        from slopometry.core.models.session import BehavioralPatternTrend

        db = EventDatabase()
        repository_path = str(Path(stats.working_directory).resolve())
        history = db.get_behavioral_pattern_history(repository_path, limit=10, exclude_session_id=session_id)

        if not history:
            return None

        num_sessions = len(history)
        avg_od_rate = sum(h["ownership_dodging_rate"] for h in history) / num_sessions
        avg_sw_rate = sum(h["simple_workaround_rate"] for h in history) / num_sessions

        return BehavioralPatternTrends(
            ownership_dodging=BehavioralPatternTrend(avg_rate=avg_od_rate, num_sessions=num_sessions),
            simple_workaround=BehavioralPatternTrend(avg_rate=avg_sw_rate, num_sessions=num_sessions),
        )
    except Exception as e:
        logger.debug(f"Failed to compute behavioral trends: {e}")
        return None


def _persist_behavioral_patterns(session_id: str, stats: "SessionStatistics") -> None:
    """Save behavioral pattern rates to the database for trend tracking."""
    if not stats.behavioral_patterns or not stats.working_directory:
        return
    try:
        from slopometry.core.database import EventDatabase

        db = EventDatabase()
        repository_path = str(Path(stats.working_directory).resolve())
        db.save_behavioral_patterns(session_id, repository_path, stats.behavioral_patterns)
    except Exception as e:
        logger.debug(f"Failed to persist behavioral patterns: {e}")


def _compute_session_baseline(
    stats: "SessionStatistics",
) -> tuple["RepoBaseline", "ImpactAssessment"] | tuple[None, None]:
    """Compute baseline and assessment for a session's complexity delta."""
    if not stats.complexity_delta:
        return None, None

    from slopometry.core.database import EventDatabase
    from slopometry.core.git_tracker import GitTracker
    from slopometry.summoner.services.baseline_service import BaselineService
    from slopometry.summoner.services.impact_calculator import ImpactCalculator

    working_dir = Path(stats.working_directory) if stats.working_directory else None
    if not working_dir or not working_dir.exists():
        return None, None

    db = EventDatabase()
    git_tracker = GitTracker(working_dir)
    head_sha = git_tracker._get_current_commit_sha()
    cached_baseline = db.get_cached_baseline(str(working_dir.resolve()), head_sha) if head_sha else None

    if not cached_baseline:
        console.print("[dim]Computing repository baseline (first run may take a while)...[/dim]")

    baseline_service = BaselineService(db=db)
    baseline = baseline_service.get_or_compute_baseline(working_dir)

    if not baseline:
        return None, None

    impact_calculator = ImpactCalculator()
    assessment = impact_calculator.calculate_impact(stats.complexity_delta, baseline)

    return baseline, assessment


@solo.command()
@click.argument("session_id", required=False, shell_complete=complete_session_id)
@click.option("--all", "all_sessions", is_flag=True, help="Delete all sessions")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def cleanup(session_id: str | None, all_sessions: bool, yes: bool) -> None:
    """Clean up session data.

    If SESSION_ID is provided, delete that specific session.
    If --all is provided, delete all sessions.
    Otherwise, show usage help.
    """
    from slopometry.solo.services.session_service import SessionService

    session_service = SessionService()

    if session_id and all_sessions:
        console.print("[red]Error: Cannot specify both session ID and --all[/red]")
        return

    if not session_id and not all_sessions:
        console.print("[yellow]Usage:[/yellow]")
        console.print("  slopometry cleanup SESSION_ID    # Delete specific session")
        console.print("  slopometry cleanup --all         # Delete all sessions")
        return

    if session_id:
        basic_info = session_service.get_session_basic_info(session_id)
        if not basic_info:
            console.print(f"[red]Session {session_id} not found[/red]")
            return

        start_time, total_events = basic_info
        console.print(f"\n[bold]Session to delete: {session_id}[/bold]")
        console.print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Total events: {total_events}")

        if not yes:
            confirm = click.confirm("\nAre you sure you want to delete this session?", default=False)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                return

        events_deleted, files_deleted = session_service.cleanup_session(session_id)
        console.print(f"[green]Deleted {events_deleted} events and {files_deleted} files[/green]")

    else:  # all_sessions
        sessions = session_service.list_sessions()
        if not sessions:
            console.print("[yellow]No sessions to delete[/yellow]")
            return

        console.print(f"\n[bold red]WARNING: This will delete ALL {len(sessions)} sessions![/bold red]")
        console.print("This action cannot be undone.")

        if not yes:
            confirm = click.confirm("\nAre you sure you want to delete all sessions?", default=False)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                return

        sessions_deleted, events_deleted, files_deleted = session_service.cleanup_all_sessions()
        console.print(
            f"[green]Deleted {sessions_deleted} sessions, {events_deleted} events, and {files_deleted} files[/green]"
        )


@click.command()
def status() -> None:
    """Show installation status and hook configuration."""
    from slopometry.core.settings import settings
    from slopometry.solo.services.hook_service import HookService
    from slopometry.solo.services.session_service import SessionService

    hook_service = HookService()
    status_info = hook_service.get_installation_status()

    console.print("[bold]Slopometry Installation Status[/bold]\n")

    console.print(f"[cyan]Data directory:[/cyan] {settings.resolved_database_path.parent}")
    console.print(f"[cyan]Database:[/cyan] {settings.resolved_database_path}\n")

    global_icon = "[green]✓[/green]" if status_info["global"] else "[red]✗[/red]"
    console.print(f"{global_icon} Global hooks: {status_info['global_path']}")

    local_icon = "[green]✓[/green]" if status_info["local"] else "[red]✗[/red]"
    console.print(f"{local_icon} Local hooks: {status_info['local_path']}")

    if not status_info["global"] and not status_info["local"]:
        console.print("\n[yellow]No slopometry hooks found. Run 'slopometry install' to start tracking.[/yellow]")
    else:
        console.print("\n[green]Hooks are installed. Claude Code sessions are being tracked automatically.[/green]")

        session_service = SessionService()
        sessions = session_service.list_sessions(limit=3)
        if sessions:
            console.print("\n[bold]Recent Sessions:[/bold]")
            for session_id in sessions:
                stats = session_service.get_session_statistics(session_id)
                if stats:
                    console.print(f"  • {session_id} ({stats.total_events} events)")


@solo.command()
@click.option("--enable/--disable", default=None, help="Enable or disable stop event feedback")
def feedback(enable: bool | None) -> None:
    """Configure complexity feedback on stop events."""
    from slopometry.core.settings import settings

    if enable is None:
        current_status = "enabled" if settings.enable_complexity_feedback else "disabled"
        console.print(f"[bold]Complexity feedback is currently {current_status}[/bold]")
        console.print("")
        console.print("To change this setting:")
        console.print("  slopometry feedback --enable    # Enable feedback")
        console.print("  slopometry feedback --disable   # Disable feedback")
        console.print("")
        if not settings.enable_complexity_feedback:
            console.print("[yellow]Note: Feedback is disabled by default. Enable it to receive[/yellow]")
            console.print("[yellow]complexity analysis when Claude Code sessions end.[/yellow]")
        return

    env_file = Path(".env")
    env_var = "SLOPOMETRY_ENABLE_STOP_FEEDBACK"
    env_value = "true" if enable else "false"

    if enable:
        console.print("[green]Enabling[/green] complexity feedback on stop events")
    else:
        console.print("[yellow]Disabling[/yellow] complexity feedback on stop events")
    console.print("")
    console.print("To persist this setting, add to your .env file:")
    console.print(f"  {env_var}={env_value}")

    if env_file.exists():
        content = env_file.read_text()
        if env_var in content:
            lines = content.split("\n")
            new_lines = [
                f"{env_var}={env_value}" if line.startswith(f"{env_var}=") else line
                for line in lines
            ]
            env_file.write_text("\n".join(new_lines))
        else:
            with env_file.open("a") as f:
                f.write(f"\n{env_var}={env_value}\n")
    else:
        env_file.write_text(f"{env_var}={env_value}\n")

    console.print(f"[green]Added {env_var}={env_value} to .env file[/green]")
    console.print("")
    console.print("[bold]Note:[/bold] You may need to restart Claude Code for changes to take effect.")


@solo.command()
def migrations() -> None:
    """Show database migration status."""
    from slopometry.core.migrations import MigrationRunner
    from slopometry.solo.services.session_service import SessionService

    session_service = SessionService()
    migration_runner = MigrationRunner(session_service.db.db_path)
    status = migration_runner.get_migration_status()

    console.print("[bold]Database Migration Status[/bold]\n")

    if status["applied"]:
        console.print("[green]Applied Migrations:[/green]")
        for migration in status["applied"]:
            console.print(f"  ✓ {migration['version']}: {migration['description']}")
            console.print(f"    Applied: {migration['applied_at']}")
        console.print()

    if status["pending"]:
        console.print("[red]Pending Migrations:[/red]")
        for migration in status["pending"]:
            console.print(f"  • {migration['version']}: {migration['description']}")
        console.print()
    else:
        console.print("[green]All migrations are up to date![/green]")

    console.print(f"Total migrations: {status['total']}")
    console.print(f"Applied: {len(status['applied'])}")
    console.print(f"Pending: {len(status['pending'])}")


def _find_plan_names_from_transcript(transcript_path: Path) -> list[str]:
    """Extract plan filenames from transcript by searching for plans/*.md references."""
    import re

    plan_names: set[str] = set()
    pattern = re.compile(r"plans/([a-z0-9-]+\.md)")

    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                matches = pattern.findall(line)
                plan_names.update(matches)
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to parse plans from transcript: {e}[/yellow]")

    return list(plan_names)


@solo.command()
@click.argument("session_id", required=False, shell_complete=complete_session_id)
@click.option("--output-dir", "-o", default=".", help="Directory to save the transcript to (default: current)")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt when using latest session")
def save_transcript(session_id: str | None, output_dir: str, yes: bool) -> None:
    """Save the Claude Code transcript, plans, and todos for a session.

    If no SESSION_ID is provided, saves from the latest session.
    Creates .slopometry/<session-id>/ with transcript.jsonl, plans/, and todos/.
    """
    import shutil
    from pathlib import Path

    from slopometry.solo.services.session_service import SessionService

    session_service = SessionService()

    if not session_id:
        session_id = session_service.get_most_recent_session()
        if not session_id:
            console.print("[red]No sessions found[/red]")
            return

        stats = session_service.get_session_statistics(session_id)
        if not stats:
            console.print(f"[red]No data found for latest session {session_id}[/red]")
            return

        console.print(f"[bold]Latest session: {session_id}[/bold]")
        console.print(f"Start time: {stats.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Total events: {stats.total_events}")

        if not yes:
            confirm = click.confirm("\nSave transcript for this session?", default=True)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                return
    else:
        stats = session_service.get_session_statistics(session_id)
        if not stats:
            console.print(f"[red]No data found for session {session_id}[/red]")
            return

    import json

    from slopometry.core.database import EventDatabase
    from slopometry.core.models import AgentTool, SessionMetadata

    output_path_dir = Path(output_dir)
    session_dir = output_path_dir / ".slopometry" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    db = EventDatabase()
    source = db.get_session_source(session_id)
    is_opencode = source == "opencode"

    if is_opencode:
        # OpenCode: extract transcript from the Stop event metadata (fetched via SDK at session end)
        transcript_data = db.get_opencode_transcript(session_id)
        if transcript_data:
            transcript_output = session_dir / "transcript.json"
            transcript_output.write_text(json.dumps(transcript_data, indent=2))
            console.print(
                f"[green]✓[/green] Saved transcript ({len(transcript_data)} messages) to: {transcript_output}"
            )
        else:
            console.print("[yellow]No transcript found in stop event (session may still be running)[/yellow]")
    else:
        # Claude Code: copy the JSONL transcript file from disk
        if not stats.transcript_path:
            console.print(f"[red]No transcript path found for session {session_id}[/red]")
            console.print("[yellow]This may be an older session before transcript tracking was added[/yellow]")
            return

        transcript_path = Path(stats.transcript_path)
        if not transcript_path.exists():
            console.print(f"[red]Transcript file not found: {transcript_path}[/red]")
            return

        transcript_output = session_dir / "transcript.jsonl"
        try:
            shutil.copy2(transcript_path, transcript_output)
            console.print(f"[green]✓[/green] Saved transcript to: {transcript_output}")
        except Exception as e:
            console.print(f"[red]Failed to copy transcript: {e}[/red]")
            return

        plan_names = _find_plan_names_from_transcript(transcript_path)
        if plan_names:
            plans_dir = session_dir / "plans"
            plans_dir.mkdir(exist_ok=True)
            for plan_name in plan_names:
                plan_source = Path.home() / ".claude" / "plans" / plan_name
                if plan_source.exists():
                    shutil.copy2(plan_source, plans_dir / plan_name)
                    console.print(f"[green]✓[/green] Saved plan: {plan_name}")

    # Save final todos from session statistics
    if stats.plan_evolution and stats.plan_evolution.final_todos:
        todos_file = session_dir / "final_todos.json"
        todos_data = [
            {"content": todo.content, "status": todo.status, "activeForm": todo.activeForm}
            for todo in stats.plan_evolution.final_todos
        ]
        todos_file.write_text(json.dumps(todos_data, indent=2))
        console.print(f"[green]✓[/green] Saved {len(todos_data)} todos to: final_todos.json")

    # Save structured session metadata
    token_usage = stats.plan_evolution.token_usage if stats.plan_evolution else None

    if is_opencode:
        # OpenCode: extract model info and version from event metadata
        model_id = None
        opencode_version = None
        with db._get_db_connection() as conn:
            import sqlite3

            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT metadata FROM hook_events
                WHERE session_id = ? AND event_type = 'MessageUpdated' AND source = 'opencode'
                ORDER BY sequence_number ASC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row:
                try:
                    meta = json.loads(row["metadata"])
                    model_id = meta.get("model_id")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Failed to parse MessageUpdated metadata for session {session_id}: {e}")

            stop_row = conn.execute(
                """
                SELECT metadata FROM hook_events
                WHERE session_id = ? AND event_type = 'Stop' AND source = 'opencode'
                ORDER BY sequence_number DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if stop_row:
                try:
                    stop_meta = json.loads(stop_row["metadata"])
                    opencode_version = stop_meta.get("opencode_version")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Failed to parse Stop event metadata for session {session_id}: {e}")

        metadata = SessionMetadata(
            session_id=stats.session_id,
            agent_tool=AgentTool.OPENCODE,
            agent_version=opencode_version,
            model=model_id,
            start_time=stats.start_time,
            end_time=stats.end_time,
            total_events=stats.total_events,
            working_directory=stats.working_directory,
            git_branch=stats.initial_git_state.current_branch if stats.initial_git_state else None,
            token_usage=token_usage,
        )
    else:
        from slopometry.core.transcript_token_analyzer import extract_transcript_metadata

        if not stats.transcript_path:
            console.print(
                "[yellow]Warning: No transcript path for Claude Code session, skipping metadata extraction[/yellow]"
            )
            return

        transcript_meta = extract_transcript_metadata(Path(stats.transcript_path))

        metadata = SessionMetadata(
            session_id=stats.session_id,
            agent_tool=AgentTool.CLAUDE_CODE,
            agent_version=transcript_meta.agent_version,
            model=transcript_meta.model,
            start_time=stats.start_time,
            end_time=stats.end_time,
            total_events=stats.total_events,
            working_directory=stats.working_directory,
            git_branch=transcript_meta.git_branch,
            token_usage=token_usage,
        )

    metadata_file = session_dir / "session_metadata.json"
    metadata_file.write_text(metadata.model_dump_json(indent=2))
    console.print("[green]✓[/green] Saved session metadata to: session_metadata.json")


_ACTION_PRIORITY: dict[FreshnessAction, int] = {
    FreshnessAction.SUPERSEDE: 3,
    FreshnessAction.MERGE: 2,
    FreshnessAction.DEDUPE: 1,
    FreshnessAction.KEEP_BOTH: 0,
}


def _highest_priority_action(group: list) -> FreshnessAction:
    """Pick the most consequential action from a candidate's reconciliation group.

    When a candidate matches multiple existing memories, the LLM may return
    different actions for each pair. A single priority resolves conflicts:
    SUPERSEDE > MERGE > DEDUPE > KEEP_BOTH. Only the winning action's side
    effects are applied.
    """
    return max(group, key=lambda d: _ACTION_PRIORITY[d.action]).action


@solo.command(name="find-memories")
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Project directory (default: cwd)",
)
@click.option(
    "--llm-endpoint",
    type=str,
    default=None,
    help="LLM endpoint URL (default: from settings)",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="Model name (default: from settings)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-process already processed sessions",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without doing it",
)
def find_memories(
    project_dir: Path | None,
    llm_endpoint: str | None,
    llm_model: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Scan transcripts, extract memory candidates, and save to database.

    This command:
    1. Validates LLM + embedding endpoints are reachable (skipped with --dry-run)
    2. Discovers transcripts across all configured harnesses
    3. Filters to sessions not yet processed
    4. Parses and cleans conversation data
    5. Generates memory candidates via LLM
    6. Runs freshness validation against existing memories
    7. Saves memories to database

    Aborts explicitly (no partial completion) if either endpoint is
    unreachable. Use --dry-run to verify transcript discovery without
    touching the LLM.
    """
    from slopometry.core.settings import settings
    from slopometry.solo.services.memory_extractor import MemoryExtractor
    from slopometry.solo.services.memory_service import MemoryService
    from slopometry.solo.services.transcript_finder import TranscriptFinder

    if project_dir is None:
        project_dir = Path.cwd()

    endpoint = llm_endpoint or settings.memory_llm_endpoint
    model = llm_model or settings.memory_llm_model
    api_key = settings.memory_llm_api_key.get_secret_value()

    console.print("[bold]Slopometry Memory Extraction[/bold]")
    console.print(f"Project: {project_dir}")
    console.print(f"LLM: {endpoint} / {model}")
    console.print()

    if dry_run:
        console.print("[yellow]--dry-run mode, no changes will be made--[/yellow]\n")

    if settings.offline_mode and not llm_endpoint:
        raise click.ClickException(
            "Memory extraction requires external LLM calls, which are disabled (offline_mode=True). "
            "Set SLOPOMETRY_OFFLINE_MODE=false to enable."
        )

    from slopometry.solo.cli.preflight import preflight_endpoints

    if not dry_run:
        preflight_endpoints(
            chat_endpoint=endpoint,
            embedding_endpoint=settings.memory_embedding_endpoint,
            chat_api_key=api_key,
            embedding_api_key=settings.memory_embedding_api_key.get_secret_value(),
        )

    transcript_finder = TranscriptFinder()
    memory_service = MemoryService()
    memory_extractor = MemoryExtractor(endpoint, model, api_key)

    from slopometry.solo.services.embedding_service import EmbeddingService

    embedding_service = EmbeddingService(
        endpoint=settings.memory_embedding_endpoint,
        model=settings.memory_embedding_model,
        api_key=settings.memory_embedding_api_key.get_secret_value(),
    )

    console.print("[dim]Discovering transcripts...[/dim]")
    transcripts = transcript_finder.discover_transcripts(project_dir)

    if not transcripts:
        console.print("[yellow]No transcripts found for this project.[/yellow]")
        return

    source_counts: Counter[str] = Counter(t.source.value for t in transcripts)
    source_breakdown = ", ".join(f"{src}={n}" for src, n in sorted(source_counts.items()))
    console.print(f"[green]Found {len(transcripts)} transcript(s)[/green] [dim]({source_breakdown})[/dim]\n")

    sessions_to_process: list = []
    for t in transcripts:
        if not force and memory_service.is_session_processed(t.session_id, str(t.project_dir), source=t.source.value):
            console.print(f"[dim]Skipping {t.source.value} {t.session_id}: already processed[/dim]")
            continue
        sessions_to_process.append(t)

    if not sessions_to_process:
        console.print("[yellow]No new sessions to process.[/yellow]")
        return

    console.print(f"[cyan]Processing {len(sessions_to_process)} session(s)...[/cyan]\n")

    total_memories = 0
    for t in sessions_to_process:
        console.print(f"[bold]Session: {t.session_id}[/bold] [dim]({t.source.value})[/dim]")
        console.print(f"  Transcript: {t.transcript_path}")

        if dry_run:
            console.print("  [yellow]-- dry-run: would extract and save memories --[/yellow]")
            continue

        try:
            from slopometry.core.models.protocol.events import AbstractEventSource

            if t.source == AbstractEventSource.OPENCODE:
                from slopometry.solo.services.transcript_finder import TranscriptFinder

                storage_root = TranscriptFinder().find_opencode_storage_root()
                if not storage_root.is_dir():
                    console.print("  [yellow]OpenCode storage not found[/yellow]")
                    continue
                cleaned_transcript = memory_extractor.extract_memories_from_opencode_session(
                    t.session_id, storage_root
                )
            else:
                cleaned_transcript = memory_extractor.extract_memories_from_transcript(t.transcript_path)

            if not cleaned_transcript.strip():
                console.print("  [yellow]Warning: Empty transcript[/yellow]")
                continue

            console.print(f"  [dim]Extracted {len(cleaned_transcript)} chars of conversation[/dim]")

            candidates = memory_extractor.generate_memory_candidates(cleaned_transcript)

            proj_dir_str = str(t.project_dir)

            if not candidates:
                console.print("  [yellow]No memory candidates generated[/yellow]")
                memory_service.mark_session_processed(t.session_id, proj_dir_str, 0, source=t.source.value)
                continue

            console.print(f"  [green]Generated {len(candidates)} candidates[/green]")

            console.print("  [dim]Generating embeddings...[/dim]")
            for i, candidate in enumerate(candidates):
                try:
                    embedding = embedding_service.get_embedding(candidate.content)
                    if embedding:
                        candidate.embedding = embedding
                    else:
                        console.print(f"  [yellow]Warning: Failed to get embedding for candidate {i+1}, continuing without[/yellow]")
                except Exception as e:
                    console.print(f"  [red]Embedding error for candidate {i+1}: {e}[/red]")
                    raise

            from slopometry.solo.services.memory_freshness import validate_freshness

            existing_memories = memory_service.get_memories(project_dir=proj_dir_str, limit=settings.memory_query_limit)
            decisions: list = []
            if existing_memories:
                decisions, distribution = validate_freshness(
                    candidates,
                    existing_memories,
                    llm_endpoint=endpoint,
                    llm_model=model,
                    api_key=api_key,
                    floor_threshold=settings.freshness_threshold_floor,
                    ceiling_threshold=settings.freshness_threshold_ceiling,
                )
                console.print(
                    f"  [dim]Project similarity distribution: "
                    f"n={distribution.n_pairs} "
                    f"mean={distribution.mean:.2f} "
                    f"p50={distribution.p50:.2f} "
                    f"p75={distribution.p75:.2f} "
                    f"p90={distribution.p90:.2f} "
                    f"p95={distribution.p95:.2f}[/dim]"
                )
                console.print(
                    f"  [dim]Derived dedupe threshold: {distribution.derived_threshold:.2f}[/dim]"
                )
                if decisions:
                    console.print(f"  [yellow]Freshness: {len(decisions)} similar pair(s) reviewed:[/yellow]")
                    for decision in decisions:
                        action_color = decision.action.color
                        console.print(
                            f"    [{action_color}]{decision.action.value.upper()}[/{action_color}]"
                            f" (sim={decision.similarity:.2f}): "
                            f"[dim]new=[/dim]{decision.new_candidate.content[:80]!r} "
                            f"[dim]existing=[/dim]{decision.existing_memory.content[:80]!r}"
                        )
                        console.print(f"      [dim]REASON:[/dim] {decision.reason}")
                    decisions_by_candidate = defaultdict(list)
                    for decision in decisions:
                        decisions_by_candidate[id(decision.new_candidate)].append(decision)

                    deduped_candidate_ids: set[int] = set()
                    merge_links: list[tuple[str, int]] = []
                    for group in decisions_by_candidate.values():
                        candidate = group[0].new_candidate
                        primary_action = _highest_priority_action(group)
                        if primary_action == FreshnessAction.MERGE:
                            merge_decision = next(
                                d for d in group if d.action == FreshnessAction.MERGE and d.merged_content
                            )
                            candidate.content = merge_decision.merged_content
                            merge_links.append((merge_decision.existing_memory.id, id(candidate)))
                        elif primary_action == FreshnessAction.DEDUPE:
                            deduped_candidate_ids.add(id(candidate))
                        if candidate.metadata is None:
                            candidate.metadata = {}
                        candidate.metadata["freshness_action"] = primary_action
                        candidate.metadata["freshness_reason"] = group[0].reason
                        if primary_action != FreshnessAction.KEEP_BOTH:
                            candidate.metadata["freshness_pair_with"] = group[0].existing_memory.id

                    if deduped_candidate_ids:
                        candidates = [c for c in candidates if id(c) not in deduped_candidate_ids]
                        console.print(f"  [dim]Skipped {len(deduped_candidate_ids)} duplicate candidate(s)[/dim]")

            from slopometry.core.models.memory import MemoryCreateRequest

            request = MemoryCreateRequest(
                session_id=t.session_id,
                project_dir=proj_dir_str,
                candidates=candidates,
            )

            saved = memory_service.save_memories(request)
            candidate_id_map: dict[int, str] = {id(c): e.id for c, e in zip(candidates, saved)}

            for decision in decisions:
                if decision.action == FreshnessAction.SUPERSEDE:
                    new_id = candidate_id_map.get(id(decision.new_candidate))
                    if new_id is None:
                        continue
                    memory_service.update_memory(
                        decision.existing_memory.id, superseded_by=new_id
                    )
                    console.print(
                        f"      [dim]Superseded {decision.existing_memory.id} -> {new_id}[/dim]"
                    )

            for existing_id, cand_obj_id in merge_links:
                new_id = candidate_id_map.get(cand_obj_id)
                if new_id is None:
                    continue
                memory_service.update_memory(existing_id, superseded_by=new_id)
                console.print(
                    f"      [dim]Merged {existing_id} -> {new_id}[/dim]"
                )

            saved_ids = {m.id for m in saved}
            current_memories = memory_service.get_memories(project_dir=proj_dir_str, limit=settings.memory_query_limit)
            pre_existing_memories = [m for m in current_memories if m.id not in saved_ids]
            if pre_existing_memories:
                stale_pairs = audit_staleness(
                    pre_existing_memories,
                    cleaned_transcript,
                    llm_endpoint=endpoint,
                    llm_model=model,
                    api_key=api_key,
                    max_tokens=settings.memory_staleness_audit_max_tokens,
                    transcript_truncation_chars=settings.memory_transcript_truncation_chars,
                )
                for memory_entry, reason in stale_pairs:
                    memory_service.retire_memory(memory_entry.id, reason)
                    console.print(
                        f"  [yellow]Retired {memory_entry.id}: {reason}[/yellow]"
                    )

            memory_service.mark_session_processed(
                t.session_id, proj_dir_str, len(saved), source=t.source.value
            )
            total_memories += len(saved)
            console.print(f"  [green]Saved {len(saved)} memories[/green]")

        except Exception as e:
            console.print(f"  [red]Error processing session: {e}[/red]")
            continue

    console.print()
    if not dry_run:
        console.print(
            f"[bold green]Done! Extracted {total_memories} memories from {len(sessions_to_process)} sessions.[/bold green]"
        )
    else:
        console.print(
            f"[bold yellow]Dry run complete. Would process {len(sessions_to_process)} sessions.[/bold yellow]"
        )


@solo.command(name="prune-memories")
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Project directory (default: cwd)",
)
@click.option(
    "--llm-endpoint",
    type=str,
    default=None,
    help="LLM endpoint URL (default: from settings)",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="Model name (default: from settings)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be retired without making changes",
)
def prune_memories(
    project_dir: Path | None,
    llm_endpoint: str | None,
    llm_model: str | None,
    dry_run: bool,
) -> None:
    """Audit existing memories for staleness and retire stale ones.

    Sends all active (non-superseded, non-retired) memories for the project
    to the LLM alongside the most recent session transcripts and asks which
    memories describe fixed bugs, completed work, or outdated state.

    Stale memories are marked with a ``retired_reason`` and excluded from
    future queries. Use --dry-run to preview without changes (the LLM is
    still called to identify stale memories; only the DB write is skipped).
    """
    from slopometry.core.settings import settings
    from slopometry.solo.services.memory_service import MemoryService
    from slopometry.solo.services.transcript_finder import TranscriptFinder

    if project_dir is None:
        project_dir = Path.cwd()

    endpoint = llm_endpoint or settings.memory_llm_endpoint
    model = llm_model or settings.memory_llm_model
    api_key = settings.memory_llm_api_key.get_secret_value()

    console.print("[bold]Slopometry Memory Pruning[/bold]")
    console.print(f"Project: {project_dir}")
    console.print(f"LLM: {endpoint} / {model}")
    console.print()

    if settings.offline_mode and not llm_endpoint:
        raise click.ClickException(
            "Memory pruning requires external LLM calls, which are disabled (offline_mode=True). "
            "Set SLOPOMETRY_OFFLINE_MODE=false to enable."
        )

    from slopometry.solo.cli.preflight import preflight_endpoints

    preflight_endpoints(
        chat_endpoint=endpoint,
        embedding_endpoint=settings.memory_embedding_endpoint,
        chat_api_key=api_key,
        embedding_api_key=settings.memory_embedding_api_key.get_secret_value(),
    )

    memory_service = MemoryService()
    proj_dir_str = str(project_dir)

    existing = memory_service.get_memories(project_dir=proj_dir_str, limit=settings.memory_query_limit)
    if not existing:
        console.print("[yellow]No active memories to audit.[/yellow]")
        return

    console.print(f"[cyan]Auditing {len(existing)} active memories for staleness...[/cyan]")

    transcript_finder = TranscriptFinder()
    transcripts = transcript_finder.discover_transcripts(project_dir)

    from slopometry.core.models.protocol.events import AbstractEventSource
    from slopometry.solo.services.memory_extractor import MemoryExtractor

    memory_extractor = MemoryExtractor(endpoint, model, api_key)

    transcript_texts: list[str] = []
    for t in transcripts[: settings.memory_prune_transcript_window]:
        if t.source == AbstractEventSource.OPENCODE:
            storage_root = transcript_finder.find_opencode_storage_root()
            if not storage_root.is_dir():
                continue
            text = memory_extractor.extract_memories_from_opencode_session(t.session_id, storage_root)
        else:
            text = memory_extractor.extract_memories_from_transcript(t.transcript_path)
        if text.strip():
            transcript_texts.append(text)

    combined_transcript = "\n---\n".join(transcript_texts)
    if not combined_transcript.strip():
        console.print("[yellow]No transcripts found to audit against.[/yellow]")
        return

    console.print(f"[dim]Using {len(transcript_texts)} transcript(s) for context[/dim]")

    stale_pairs = audit_staleness(
        existing,
        combined_transcript,
        llm_endpoint=endpoint,
        llm_model=model,
        api_key=api_key,
        max_tokens=settings.memory_staleness_audit_max_tokens,
        transcript_truncation_chars=settings.memory_transcript_truncation_chars,
    )

    if not stale_pairs:
        console.print("[green]No stale memories found.[/green]")
        return

    console.print(f"\n[yellow]Found {len(stale_pairs)} stale memor(ies):[/yellow]")
    for memory_entry, reason in stale_pairs:
        console.print(f"  [yellow]RETIRE[/yellow] [{memory_entry.memory_type.value}] {memory_entry.content[:100]}")
        console.print(f"    [dim]REASON:[/dim] {reason}")

    if dry_run:
        console.print(f"\n[yellow]--dry-run: would retire {len(stale_pairs)} memor(ies)[/yellow]")
        return

    retired_count = 0
    for memory_entry, reason in stale_pairs:
        if memory_service.retire_memory(memory_entry.id, reason):
            retired_count += 1

    console.print(f"\n[bold green]Retired {retired_count} memor(ies).[/bold green]")


@solo.command(name="show-memories")
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Project directory (default: cwd)",
)
@click.option(
    "--type",
    "memory_type",
    type=click.Choice(["user", "feedback", "project", "reference"]),
    default=None,
    help="Filter by memory type",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Maximum number of results (default: 50)",
)
def show_memories(
    project_dir: Path | None,
    memory_type: str | None,
    limit: int,
) -> None:
    """List and manage memories for a project.

    When run without options, enters interactive mode with actions:
      (r)etain <id>  - Mark memory as retained
      (d)elete <id>  - Delete a memory
      (e)dit <id>    - Edit memory content
      (f)ilter <type> - Filter by type (user|feedback|project|reference)
      (q)uit         - Quit
    """
    from rich.table import Table

    from slopometry.core.models.memory import MemoryType
    from slopometry.core.settings import settings
    from slopometry.solo.services.embedding_service import EmbeddingService
    from slopometry.solo.services.memory_service import MemoryService

    if project_dir is None:
        project_dir = Path.cwd()

    memory_service = MemoryService()
    project_dir_str = str(project_dir.resolve())

    embedding_service = EmbeddingService(
        endpoint=settings.memory_embedding_endpoint,
        model=settings.memory_embedding_model,
        api_key=settings.memory_embedding_api_key.get_secret_value(),
    )

    current_type_filter = memory_type

    def display_memories(mtype: str | None) -> list:
        return memory_service.get_memories(
            project_dir=project_dir_str,
            memory_type=MemoryType(mtype) if mtype else None,
            limit=limit,
        )

    while True:
        console.print(f"\n[bold]Slopometry Memories: {project_dir}[/bold]\n")

        memories = display_memories(current_type_filter)

        if not memories:
            console.print("[yellow]No memories found.[/yellow]")
            try:
                user_input = console.input("\nPress Enter to continue or 'q' to quit... ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Cancelled[/yellow]")
                return
            if user_input.lower() == "q":
                console.print("[green]Goodbye![/green]")
                return
            continue

        table = Table(show_header=True, header_style="bold")
        table.add_column("ID", style="dim", width=8)
        table.add_column("Type", width=10)
        table.add_column("Uniqueness", width=10)
        table.add_column("Content", max_width=60)
        table.add_column("Session", style="dim")

        for idx, memory in enumerate(memories, 1):
            content_preview = memory.content[:55] + "..." if len(memory.content) > 55 else memory.content
            retained_marker = " [retained]" if memory.retained else ""

            uniqueness = "N/A"
            if memory.embedding:
                comparison_set = [m.embedding for j, m in enumerate(memories) if j != idx - 1 and m.embedding]
                uniqueness_score = embedding_service.compute_uniqueness_score(memory.embedding, comparison_set)
                uniqueness = f"{uniqueness_score:.2f}"

            table.add_row(
                str(idx),
                memory.memory_type.value,
                uniqueness,
                content_preview + retained_marker,
                memory.session_id[:8],
            )

        console.print(table)

        console.print("\n[bold]Actions:[/bold]")
        console.print("  (r)etain <id>  - Mark memory as retained")
        console.print("  (d)elete <id>  - Delete a memory")
        console.print("  (e)dit <id>    - Edit memory content")
        console.print("  (f)ilter <type> - Filter by type (user|feedback/project|reference)")
        console.print("  (p)urge        - Delete ALL memories (requires confirmation)")
        console.print("  (q)uit         - Quit")

        try:
            user_input = console.input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled[/yellow]")
            return

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if cmd == "q":
            console.print("[green]Goodbye![/green]")
            return

        if cmd == "f":
            if not arg:
                console.print("[yellow]Usage: f <type> (user|feedback|project|reference)[/yellow]")
                continue
            if arg not in ["user", "feedback", "project", "reference"]:
                console.print(f"[red]Invalid type: {arg}[/red]")
                continue
            current_type_filter = arg
            continue

        if cmd == "p":
            console.print("\n[bold red]WARNING: This will delete ALL memories![/bold red]")
            try:
                confirm = console.input("Type 'yes' to confirm: ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Cancelled[/yellow]")
                continue

            if confirm.lower() == "yes":
                count = memory_service.delete_all_memories()
                console.print(f"[green]Deleted {count} memories[/green]")
            else:
                console.print("[yellow]Purge cancelled[/yellow]")
            continue

        if not arg or not arg.isdigit():
            console.print("[yellow]Usage: <action> <id> (e.g., d 1, r 2)[/yellow]")
            continue

        idx = int(arg) - 1
        if idx < 0 or idx >= len(memories):
            console.print(f"[red]Invalid ID: {arg} (must be 1-{len(memories)})[/red]")
            continue

        memory = memories[idx]

        if cmd == "d":
            if memory_service.delete_memory(memory.id):
                console.print(f"[green]Deleted memory {arg}[/green]")
            else:
                console.print(f"[red]Failed to delete memory {arg}[/red]")
            continue

        elif cmd == "r":
            if memory_service.update_memory(memory.id, retained=True):
                console.print(f"[green]Marked memory {arg} as retained[/green]")
            else:
                console.print(f"[red]Failed to update memory {arg}[/red]")
            continue

        elif cmd == "e":
            console.print("[dim]Current content:[/dim]")
            console.print(f"  {memory.content}")
            console.print("[dim]Enter new content (or press Enter to cancel):[/dim]")
            try:
                new_content = console.input("  New content: ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Cancelled[/yellow]")
                continue

            if new_content:
                if memory_service.update_memory(memory.id, content=new_content):
                    console.print(f"[green]Updated memory {arg}[/green]")
                else:
                    console.print(f"[red]Failed to update memory {arg}[/red]")
            else:
                console.print("[yellow]No changes made[/yellow]")
            continue

        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print("Valid commands: r (retain), d (delete), e (edit), f (filter), p (purge), q (quit)")
