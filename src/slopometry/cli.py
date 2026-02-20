"""Main CLI dispatcher for slopometry."""

import shutil
import sys
from importlib.metadata import version

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
    from slopometry.core.models.hook import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.PRE_TOOL_USE))


@cli.command("hook-post-tool-use", hidden=True)
def hook_post_tool_use() -> None:
    """Internal command for processing PostToolUse hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.models.hook import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.POST_TOOL_USE))


@cli.command("hook-notification", hidden=True)
def hook_notification() -> None:
    """Internal command for processing Notification hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.models.hook import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.NOTIFICATION))


@cli.command("hook-stop", hidden=True)
def hook_stop() -> None:
    """Internal command for processing Stop hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.models.hook import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.STOP))


@cli.command("hook-subagent-stop", hidden=True)
def hook_subagent_stop() -> None:
    """Internal command for processing SubagentStop hook events."""
    from slopometry.core.hook_handler import handle_hook
    from slopometry.core.models.hook import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.SUBAGENT_STOP))


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
