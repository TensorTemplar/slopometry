"""CLI commands for solo-leveler features."""

from pathlib import Path

import click
from rich.console import Console

from slopometry.core.settings import settings
from slopometry.display.formatters import create_sessions_table, display_session_summary
from slopometry.solo.services.hook_service import HookService
from slopometry.solo.services.session_service import SessionService

console = Console()


def complete_session_id(ctx, param, incomplete):
    """Complete session IDs from the database."""
    try:
        session_service = SessionService()
        sessions = session_service.list_sessions()
        return [session for session in sessions if session.startswith(incomplete)]
    except Exception:
        return []


@click.group()
def solo():
    """Solo-leveler commands for basic session tracking."""
    pass


@click.command()
@click.option(
    "--global/--local",
    "global_",
    default=False,
    help="Install hooks globally (~/.claude) or locally (./.claude)",
)
def install(global_):
    """Install slopometry hooks into Claude Code settings to automatically track all sessions and tool usage."""
    hook_service = HookService()
    success, message = hook_service.install_hooks(global_)

    if success:
        console.print(f"[green]{message}[/green]")
        console.print("[cyan]All Claude Code sessions will now be automatically tracked[/cyan]")
    else:
        console.print(f"[red]{message}[/red]")


@click.command()
@click.option(
    "--global/--local",
    "global_",
    default=False,
    help="Remove hooks globally (~/.claude) or locally (./.claude)",
)
def uninstall(global_):
    """Remove slopometry hooks from Claude Code settings to completely stop automatic session tracking."""
    hook_service = HookService()
    success, message = hook_service.uninstall_hooks(global_)

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")


@solo.command("ls")
@click.option("--limit", default=None, help="Number of recent sessions to show")
def list_sessions(limit):
    """List recent Claude Code sessions."""
    session_service = SessionService()
    all_sessions = session_service.list_sessions()
    sessions = all_sessions[: int(limit)] if limit else all_sessions

    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        console.print("[dim]Run 'slopometry install' to start tracking Claude Code sessions[/dim]")
        return

    sessions_data = session_service.prepare_sessions_data_for_display(sessions)
    table = create_sessions_table(sessions_data)
    console.print(table)


@solo.command()
@click.argument("session_id", shell_complete=complete_session_id)
def show(session_id):
    """Show detailed statistics for a session."""
    session_service = SessionService()
    stats = session_service.get_session_statistics(session_id)

    if not stats:
        console.print(f"[red]No data found for session {session_id}[/red]")
        return

    display_session_summary(stats, session_id)


@click.command()
def latest():
    """Show detailed statistics for the most recent session."""
    session_service = SessionService()
    most_recent = session_service.get_most_recent_session()

    if not most_recent:
        console.print("[red]No sessions found[/red]")
        return

    console.print(f"[bold]Showing most recent session: {most_recent}[/bold]\\n")
    stats = session_service.get_session_statistics(most_recent)
    if stats:
        display_session_summary(stats, most_recent)


@solo.command()
@click.argument("session_id", required=False, shell_complete=complete_session_id)
@click.option("--all", "all_sessions", is_flag=True, help="Delete all sessions")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def cleanup(session_id, all_sessions, yes):
    """Clean up session data.

    If SESSION_ID is provided, delete that specific session.
    If --all is provided, delete all sessions.
    Otherwise, show usage help.
    """
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

        console.print(f"\\n[bold]Session to delete: {session_id}[/bold]")
        console.print(f"Start time: {stats.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Total events: {stats.total_events}")

        if not yes:
            confirm = click.confirm("\\nAre you sure you want to delete this session?", default=False)
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

        console.print(f"\\n[bold red]WARNING: This will delete ALL {len(sessions)} sessions![/bold red]")
        console.print("This action cannot be undone.")

        if not yes:
            confirm = click.confirm("\\nAre you sure you want to delete all sessions?", default=False)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                return

        sessions_deleted, events_deleted, files_deleted = session_service.cleanup_all_sessions()
        console.print(
            f"[green]Deleted {sessions_deleted} sessions, {events_deleted} events, and {files_deleted} files[/green]"
        )


@click.command()
def status():
    """Show installation status and hook configuration."""
    hook_service = HookService()
    status_info = hook_service.get_installation_status()

    console.print("[bold]Slopometry Installation Status[/bold]\\n")

    console.print(f"[cyan]Data directory:[/cyan] {settings.resolved_database_path.parent}")
    console.print(f"[cyan]Database:[/cyan] {settings.resolved_database_path}\\n")

    global_icon = "[green]✓[/green]" if status_info["global"] else "[red]✗[/red]"
    console.print(f"{global_icon} Global hooks: {status_info['global_path']}")

    local_icon = "[green]✓[/green]" if status_info["local"] else "[red]✗[/red]"
    console.print(f"{local_icon} Local hooks: {status_info['local_path']}")

    if not status_info["global"] and not status_info["local"]:
        console.print("\\n[yellow]No slopometry hooks found. Run 'slopometry install' to start tracking.[/yellow]")
    else:
        console.print("\\n[green]Hooks are installed. Claude Code sessions are being tracked automatically.[/green]")

        session_service = SessionService()
        sessions = session_service.list_sessions(limit=3)
        if sessions:
            console.print("\\n[bold]Recent Sessions:[/bold]")
            for session_id in sessions:
                stats = session_service.get_session_statistics(session_id)
                if stats:
                    console.print(f"  • {session_id} ({stats.total_events} events)")


@solo.command()
@click.option("--enable/--disable", default=None, help="Enable or disable stop event feedback")
def feedback(enable):
    """Configure complexity feedback on stop events."""
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
                lines = content.split("\\n")
                new_lines = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=true")
                    else:
                        new_lines.append(line)
                env_file.write_text("\\n".join(new_lines))
            else:
                with env_file.open("a") as f:
                    f.write(f"\\n{env_var}=true\\n")
        else:
            env_file.write_text(f"{env_var}=true\\n")

        console.print(f"[green]Added {env_var}=true to .env file[/green]")

    else:
        console.print("[yellow]Disabling[/yellow] complexity feedback on stop events")
        console.print("")
        console.print("To persist this setting, add to your .env file:")
        console.print(f"  {env_var}=false")

        if env_file.exists():
            content = env_file.read_text()
            if env_var in content:
                lines = content.split("\\n")
                new_lines = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=false")
                    else:
                        new_lines.append(line)
                env_file.write_text("\\n".join(new_lines))
            else:
                with env_file.open("a") as f:
                    f.write(f"\\n{env_var}=false\\n")
        else:
            env_file.write_text(f"{env_var}=false\\n")

        console.print(f"[green]Added {env_var}=false to .env file[/green]")

    console.print("")
    console.print("[bold]Note:[/bold] You may need to restart Claude Code for changes to take effect.")
