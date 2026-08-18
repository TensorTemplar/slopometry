"""Main CLI dispatcher for slopometry."""

import shutil
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

# REASON: analyzed repos may contain invalid escape sequences that emit SyntaxWarnings during AST parsing
warnings.filterwarnings("ignore", category=SyntaxWarning)

import click

from slopometry.display.console import console


def get_version() -> str:
    """Get package version."""
    return version("slopometry")


def check_slopometry_in_path() -> bool:
    """Check if slopometry is available in PATH."""
    return shutil.which("slopometry") is not None


def warn_if_not_in_path() -> None:
    """Print a warning if slopometry is not in PATH."""
    if not check_slopometry_in_path():
        console.print("\n[yellow]Warning: 'slopometry' is not in your PATH.[/yellow]")
        console.print("[yellow]Run 'uv tool update-shell' and restart your terminal to fix this.[/yellow]")


@click.group()
@click.version_option(version=get_version(), prog_name="slopometry")
def cli() -> None:
    """Slopometry - Claude Code session tracker.

    Solo-leveler features: Basic session tracking and analysis
    Summoner features: Advanced experimentation and AI integration
    """
    pass


@cli.command("hook-handler", hidden=True)
def hook_handler() -> None:
    """Internal command for processing hook events."""
    from slopometry.core.hook_handler import handle_hook

    sys.exit(handle_hook())


@cli.command("hook-pre-tool-use", hidden=True)
def hook_pre_tool_use() -> None:
    """Internal command for processing PreToolUse hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.protocol.adapters.claude import CLAUDE_HOOK_KIND_MAP

    sys.exit(handle_hook(event_type_override=CLAUDE_HOOK_KIND_MAP["PreToolUse"]))


@cli.command("hook-post-tool-use", hidden=True)
def hook_post_tool_use() -> None:
    """Internal command for processing PostToolUse hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.protocol.adapters.claude import CLAUDE_HOOK_KIND_MAP

    sys.exit(handle_hook(event_type_override=CLAUDE_HOOK_KIND_MAP["PostToolUse"]))


@cli.command("hook-notification", hidden=True)
def hook_notification() -> None:
    """Internal command for processing Notification hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.protocol.adapters.claude import CLAUDE_HOOK_KIND_MAP

    sys.exit(handle_hook(event_type_override=CLAUDE_HOOK_KIND_MAP["Notification"]))


@cli.command("hook-stop", hidden=True)
def hook_stop() -> None:
    """Internal command for processing Stop hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.protocol.adapters.claude import CLAUDE_HOOK_KIND_MAP

    sys.exit(handle_hook(event_type_override=CLAUDE_HOOK_KIND_MAP["Stop"]))


@cli.command("hook-subagent-stop", hidden=True)
def hook_subagent_stop() -> None:
    """Internal command for processing SubagentStop hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.protocol.adapters.claude import CLAUDE_HOOK_KIND_MAP

    sys.exit(handle_hook(event_type_override=CLAUDE_HOOK_KIND_MAP["SubagentStop"]))


@cli.command("hook-opencode", hidden=True)
@click.option(
    "--event-type",
    required=True,
    type=click.Choice(
        ["pre_tool_use", "post_tool_use", "stop", "subagent_stop", "subagent_start", "todo_updated", "message_updated"]
    ),
    help="Type of OpenCode event being forwarded.",
)
def hook_opencode(event_type: str) -> None:
    """Internal command for processing OpenCode plugin events.

    Called by the OpenCode TypeScript plugin with JSON on stdin.
    """
    from slopometry.core.opencode_handler import handle_opencode_hook

    sys.exit(handle_opencode_hook(event_type))


@cli.command("ingest")
@click.option("--source", required=True, help="Agent tool that produced the events, e.g. 'mmkr'.")
@click.option(
    "--file",
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSONL file of event envelopes; reads from stdin when omitted.",
)
@click.option(
    "--working-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Working directory recorded on ingested events; defaults to the current directory.",
)
def ingest(source: str, input_file: Path | None, working_directory: Path | None) -> None:
    """Ingest hook-protocol event envelopes (JSONL) from any agent tool.

    Each line must be an EventEnvelope JSON object; see
    slopometry.core.protocol.schema for the stable schema. Re-ingesting the
    same (source, event_id) pairs is an idempotent no-op.
    """
    import json

    from pydantic import ValidationError

    from slopometry.core.protocol.ingest import ingest_envelopes
    from slopometry.core.protocol.schema import EventEnvelope

    content = input_file.read_text() if input_file else sys.stdin.read()
    envelopes: list[EventEnvelope] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            envelope = EventEnvelope.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as e:
            console.print(f"[red]Invalid envelope at line {line_number}: {e}[/red]")
            sys.exit(2)
        if envelope.source != source:
            console.print(
                f"[red]Envelope source '{envelope.source}' at line {line_number} does not match --source '{source}'[/red]"
            )
            sys.exit(2)
        envelopes.append(envelope)

    report = ingest_envelopes(envelopes, working_directory=str(working_directory) if working_directory else None)
    console.print(
        f"[green]Ingested {report.inserted} events from source '{source}'"
        f" (skipped {report.skipped_duplicates} duplicates)[/green]"
    )


@cli.command("shell-completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def shell_completion(shell: str) -> None:
    """Generate shell completion script."""
    warn_if_not_in_path()

    if shell == "bash":
        console.print("[bold]Add this to your ~/.bashrc:[/bold]")
        console.print('eval "$(_SLOPOMETRY_COMPLETE=bash_source slopometry)"')
        console.print("\n[bold]Or install directly:[/bold]")
        console.print("_SLOPOMETRY_COMPLETE=bash_source slopometry > ~/.slopometry-complete.sh")
        console.print("echo 'source ~/.slopometry-complete.sh' >> ~/.bashrc")
    elif shell == "zsh":
        console.print("[bold]Manual Installation (Recommended for Oh My Zsh):[/bold]")
        console.print("mkdir -p ~/.oh-my-zsh/completions")
        console.print(
            "_SLOPOMETRY_COMPLETE=zsh_source slopometry | sed '/commands\\[slopometry\\]/d' > "
            "~/.oh-my-zsh/completions/_slopometry"
        )
        console.print("\n[dim]Note: You may need to run 'compinit' or restart your shell.[/dim]")

        console.print("\n[bold]Configuration (Cleanup):[/bold]")
        console.print("To hide internal functions like '_slopometry' from tab completion, add this to your ~/.zshrc:")
        console.print("zstyle ':completion:*:*:-command-:*:*' ignored-patterns '_slopometry*'")

        console.print("\n[bold]Alternative (Direct Eval):[/bold]")
        console.print('eval "$(_SLOPOMETRY_COMPLETE=zsh_source slopometry)"')
    elif shell == "fish":
        console.print("[bold]Add this to your fish config:[/bold]")
        console.print("_SLOPOMETRY_COMPLETE=fish_source slopometry | source")
        console.print("\n[bold]Or install directly:[/bold]")
        console.print("_SLOPOMETRY_COMPLETE=fish_source slopometry > ~/.config/fish/completions/slopometry.fish")

    console.print("\n[yellow]Note: Restart your shell or source your config file after installation.[/yellow]")


from slopometry.solo.cli.commands import install, latest, solo, status, uninstall
from slopometry.summoner.cli.commands import summoner

cli.add_command(install)
cli.add_command(uninstall)
cli.add_command(status)
cli.add_command(latest)

cli.add_command(solo)
cli.add_command(summoner)


if __name__ == "__main__":
    cli()
