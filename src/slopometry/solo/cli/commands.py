"""CLI commands for solo-leveler features."""

import logging
from pathlib import Path

import click
from rich.console import Console

# Imports moved inside functions to optimize startup time

console = Console()
logger = logging.getLogger(__name__)


def complete_session_id(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[str]:
    """Complete session IDs from the database."""
    from slopometry.solo.services.session_service import SessionService

    try:
        session_service = SessionService()
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
def install(global_: bool) -> None:
    """Install slopometry hooks into Claude Code settings to automatically track all sessions and tool usage."""
    from slopometry.core.settings import get_default_config_dir, get_default_data_dir
    from slopometry.solo.services.hook_service import HookService

    hook_service = HookService()
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
def uninstall(global_: bool) -> None:
    """Remove slopometry hooks from Claude Code settings to completely stop automatic session tracking."""
    from slopometry.solo.services.hook_service import HookService

    hook_service = HookService()
    success, message = hook_service.uninstall_hooks(global_)

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")


@solo.command("ls")
@click.option("--limit", default=None, type=int, help="Number of recent sessions to show")
def list_sessions(limit: int) -> None:
    """List recent Claude Code sessions."""
    from slopometry.display.formatters import create_sessions_table
    from slopometry.solo.services.session_service import SessionService

    session_service = SessionService()
    sessions_data = session_service.get_sessions_for_display(limit=limit)

    if not sessions_data:
        console.print("[yellow]No sessions found[/yellow]")
        console.print("[dim]Run 'slopometry install' to start tracking Claude Code sessions[/dim]")
        return

    table = create_sessions_table(sessions_data)
    console.print(table)


@solo.command()
@click.argument("session_id", shell_complete=complete_session_id)
@click.option("--smell-details", is_flag=True, help="Show files affected by each code smell")
@click.option("--file-details", is_flag=True, help="Show full file lists in delta sections")
def show(session_id: str, smell_details: bool, file_details: bool) -> None:
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

    baseline, assessment = _compute_session_baseline(stats)
    display_session_summary(
        stats,
        session_id,
        baseline,
        assessment,
        show_smell_files=smell_details,
        show_file_details=file_details,
    )

    elapsed = time.perf_counter() - start_time
    if elapsed > 5:
        console.print(f"\n[dim]Analysis completed in {elapsed:.1f}s[/dim]")


@click.command()
@click.option("--smell-details", is_flag=True, help="Show files affected by each code smell")
@click.option("--file-details", is_flag=True, help="Show full file lists in delta sections")
def latest(smell_details: bool, file_details: bool) -> None:
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
                try:
                    from slopometry.core.coverage_analyzer import CoverageAnalyzer

                    coverage_analyzer = CoverageAnalyzer(working_dir)
                    coverage_result = coverage_analyzer.analyze_coverage()

                    if coverage_result.coverage_available:
                        stats.complexity_metrics.test_coverage_percent = coverage_result.total_coverage_percent
                        stats.complexity_metrics.test_coverage_source = coverage_result.source_file
                except Exception as e:
                    logger.debug(f"Coverage analysis failed (optional): {e}")

        baseline, assessment = _compute_session_baseline(stats)
        display_session_summary(
            stats,
            most_recent,
            baseline,
            assessment,
            show_smell_files=smell_details,
            show_file_details=file_details,
        )

        elapsed = time.perf_counter() - start_time
        if elapsed > 5:
            console.print(f"\n[dim]Analysis completed in {elapsed:.1f}s[/dim]")


def _compute_session_baseline(stats):
    """Compute baseline and assessment for a session's complexity delta.

    Returns:
        Tuple of (baseline, assessment) or (None, None) if unavailable
    """
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
        stats = session_service.get_session_statistics(session_id)
        if not stats:
            console.print(f"[red]Session {session_id} not found[/red]")
            return

        console.print(f"\n[bold]Session to delete: {session_id}[/bold]")
        console.print(f"Start time: {stats.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Total events: {stats.total_events}")

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

    if enable:
        console.print("[green]Enabling[/green] complexity feedback on stop events")
        console.print("")
        console.print("To persist this setting, add to your .env file:")
        console.print(f"  {env_var}=true")

        if env_file.exists():
            content = env_file.read_text()
            if env_var in content:
                lines = content.split("\n")
                new_lines: list[str] = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=true")
                    else:
                        new_lines.append(line)
                env_file.write_text("\n".join(new_lines))
            else:
                with env_file.open("a") as f:
                    f.write(f"\n{env_var}=true\n")
        else:
            env_file.write_text(f"{env_var}=true\n")

        console.print(f"[green]Added {env_var}=true to .env file[/green]")

    else:
        console.print("[yellow]Disabling[/yellow] complexity feedback on stop events")
        console.print("")
        console.print("To persist this setting, add to your .env file:")
        console.print(f"  {env_var}=false")

        if env_file.exists():
            content = env_file.read_text()
            if env_var in content:
                lines = content.split("\n")
                new_lines: list[str] = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=false")
                    else:
                        new_lines.append(line)
                env_file.write_text("\n".join(new_lines))
            else:
                with env_file.open("a") as f:
                    f.write(f"\n{env_var}=false\n")
        else:
            env_file.write_text(f"{env_var}=false\n")

        console.print(f"[green]Added {env_var}=false to .env file[/green]")

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


def _find_session_todos(session_id: str) -> list[Path]:
    """Find todo files matching session ID pattern in ~/.claude/todos/."""
    todos_dir = Path.home() / ".claude" / "todos"
    if not todos_dir.exists():
        return []

    return list(todos_dir.glob(f"{session_id}-*.json"))


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

    # If no session_id provided, use the latest session
    if not session_id:
        session_id = session_service.get_most_recent_session()
        if not session_id:
            console.print("[red]No sessions found[/red]")
            return

        stats = session_service.get_session_statistics(session_id)
        if not stats:
            console.print(f"[red]No data found for latest session {session_id}[/red]")
            return

        # Show session info and ask for confirmation
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

    if not stats.transcript_path:
        console.print(f"[red]No transcript path found for session {session_id}[/red]")
        console.print("[yellow]This may be an older session before transcript tracking was added[/yellow]")
        return

    transcript_path = Path(stats.transcript_path)
    if not transcript_path.exists():
        console.print(f"[red]Transcript file not found: {transcript_path}[/red]")
        return

    output_path_dir = Path(output_dir)
    session_dir = output_path_dir / ".slopometry" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

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

    todo_files = _find_session_todos(session_id)
    if todo_files:
        todos_dir = session_dir / "todos"
        todos_dir.mkdir(exist_ok=True)
        for todo_file in todo_files:
            shutil.copy2(todo_file, todos_dir / todo_file.name)
            console.print(f"[green]✓[/green] Saved todo: {todo_file.name}")
