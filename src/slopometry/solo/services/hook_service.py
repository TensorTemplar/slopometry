"""Hook management service for Claude Code integration."""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from slopometry.core.settings import get_default_config_dir, get_default_data_dir, settings


class HookCommand(BaseModel, extra="allow"):
    """A single hook command to execute."""

    type: str = "command"
    command: str


class HookConfig(BaseModel, extra="allow"):
    """Configuration for a hook event handler."""

    matcher: str | None = None
    hooks: list[HookCommand]

    def is_slopometry_hook(self) -> bool:
        """Check if this config contains slopometry hooks."""
        return any("slopometry hook-" in hook.command for hook in self.hooks)


class ClaudeSettingsHooks(BaseModel, extra="allow"):
    """Hooks section of Claude Code settings.json."""

    PreToolUse: list[HookConfig] = Field(default_factory=list)
    PostToolUse: list[HookConfig] = Field(default_factory=list)
    Notification: list[HookConfig] = Field(default_factory=list)
    Stop: list[HookConfig] = Field(default_factory=list)
    SubagentStop: list[HookConfig] = Field(default_factory=list)

    def _all_hook_lists(self) -> list[tuple[str, list[HookConfig]]]:
        """Return all hook lists with their names for iteration."""
        return [
            ("PreToolUse", self.PreToolUse),
            ("PostToolUse", self.PostToolUse),
            ("Notification", self.Notification),
            ("Stop", self.Stop),
            ("SubagentStop", self.SubagentStop),
        ]

    def remove_slopometry_hooks(self) -> bool:
        """Remove all slopometry hooks. Returns True if any removed."""
        removed = False
        for name, configs in self._all_hook_lists():
            original_len = len(configs)
            filtered = [c for c in configs if not c.is_slopometry_hook()]
            if len(filtered) < original_len:
                removed = True
            match name:
                case "PreToolUse":
                    self.PreToolUse = filtered
                case "PostToolUse":
                    self.PostToolUse = filtered
                case "Notification":
                    self.Notification = filtered
                case "Stop":
                    self.Stop = filtered
                case "SubagentStop":
                    self.SubagentStop = filtered
        return removed

    def add_slopometry_hooks(self, hook_configs: dict[str, list[dict]]) -> None:
        """Add slopometry hooks, replacing any existing ones."""
        self.remove_slopometry_hooks()
        for hook_type, configs in hook_configs.items():
            parsed = [HookConfig.model_validate(c) for c in configs]
            match hook_type:
                case "PreToolUse":
                    self.PreToolUse.extend(parsed)
                case "PostToolUse":
                    self.PostToolUse.extend(parsed)
                case "Notification":
                    self.Notification.extend(parsed)
                case "Stop":
                    self.Stop.extend(parsed)
                case "SubagentStop":
                    self.SubagentStop.extend(parsed)

    def has_slopometry_hooks(self) -> bool:
        """Check if any slopometry hooks are installed."""
        for _, configs in self._all_hook_lists():
            for config in configs:
                if config.is_slopometry_hook():
                    return True
        return False


class ClaudePermissions(BaseModel, extra="allow"):
    """Permissions section of Claude Code settings.json."""

    allow: list[str] = Field(default_factory=list)


class ClaudeSettings(BaseModel, extra="allow"):
    """Complete Claude Code settings.json structure."""

    hooks: ClaudeSettingsHooks = Field(default_factory=ClaudeSettingsHooks)
    permissions: ClaudePermissions = Field(default_factory=ClaudePermissions)

    @classmethod
    def load(cls, path: Path) -> "ClaudeSettings":
        """Load settings from file."""
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        """Save settings to file."""
        path.parent.mkdir(exist_ok=True)
        path.write_text(self.model_dump_json(indent=2, exclude_defaults=True))


class HookService:
    """Handles Claude Code hook installation and management."""

    WHITELISTED_COMMANDS = [
        "Bash(slopometry solo:*)",
        "Bash(slopometry solo show:*)",
    ]

    def _update_gitignore(self, working_dir: Path) -> tuple[bool, str | None]:
        """Add .slopometry/ to .gitignore if in a git repository.

        Args:
            working_dir: The working directory where .gitignore should be created/updated

        Returns:
            Tuple of (was_updated, message). message is None if no update was made,
            otherwise contains a description of what was done.
        """
        git_dir = working_dir / ".git"
        if not git_dir.exists():
            return False, None

        gitignore_path = working_dir / ".gitignore"
        entry = settings.gitignore_entry

        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text()
                for line in content.splitlines():
                    if line.strip().rstrip("/") == entry.rstrip("/"):
                        return False, None
            except OSError:
                return False, None

        try:
            if gitignore_path.exists():
                content = gitignore_path.read_text()
                if content and not content.endswith("\n"):
                    content += "\n"
                content += f"\n{settings.gitignore_comment}\n{entry}\n"
                gitignore_path.write_text(content)
            else:
                gitignore_path.write_text(f"{settings.gitignore_comment}\n{entry}\n")
            return True, f"Added {entry} to .gitignore"
        except OSError as e:
            return False, f"Warning: Could not update .gitignore: {e}"

    def create_hook_configuration(self) -> dict:
        """Create slopometry hook configuration for Claude Code."""
        base_command = settings.hook_command.replace("hook-handler", "hook-{}")

        return {
            "PreToolUse": [
                {"matcher": ".*", "hooks": [{"type": "command", "command": base_command.format("pre-tool-use")}]}
            ],
            "PostToolUse": [
                {"matcher": ".*", "hooks": [{"type": "command", "command": base_command.format("post-tool-use")}]}
            ],
            "Notification": [{"hooks": [{"type": "command", "command": base_command.format("notification")}]}],
            "Stop": [{"hooks": [{"type": "command", "command": base_command.format("stop")}]}],
            "SubagentStop": [{"hooks": [{"type": "command", "command": base_command.format("subagent-stop")}]}],
        }

    def _ensure_global_directories(self) -> None:
        """Create global config and data directories if they don't exist."""
        config_dir = get_default_config_dir()
        data_dir = get_default_data_dir()

        config_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    def install_hooks(self, global_: bool = False) -> tuple[bool, str]:
        """Install slopometry hooks into Claude Code settings.

        Also ensures global config and data directories exist.

        Returns:
            Tuple of (success, message)
        """
        self._ensure_global_directories()

        settings_dir = Path.home() / ".claude" if global_ else Path.cwd() / ".claude"
        settings_file = settings_dir / "settings.json"

        try:
            claude_settings = ClaudeSettings.load(settings_file)
        except (json.JSONDecodeError, ValueError):
            return False, f"Invalid JSON in {settings_file}"

        if claude_settings.hooks.has_slopometry_hooks() and settings.backup_existing_settings:
            backup_file = settings_dir / f"settings.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_file.write_text(settings_file.read_text())

        claude_settings.hooks.add_slopometry_hooks(self.create_hook_configuration())
        claude_settings.permissions.allow = list(
            set(claude_settings.permissions.allow) | set(self.WHITELISTED_COMMANDS)
        )

        try:
            claude_settings.save(settings_file)
        except Exception as e:
            return False, f"Failed to write settings: {e}"

        gitignore_message = None
        if not global_:
            _, gitignore_message = self._update_gitignore(Path.cwd())

        scope = "globally" if global_ else "locally"
        message = f"Slopometry hooks installed {scope} in {settings_file}"
        if gitignore_message:
            message += f"\n{gitignore_message}"

        return True, message

    def uninstall_hooks(self, global_: bool = False) -> tuple[bool, str]:
        """Remove slopometry hooks from Claude Code settings.

        Returns:
            Tuple of (success, message)
        """
        settings_dir = Path.home() / ".claude" if global_ else Path.cwd() / ".claude"
        settings_file = settings_dir / "settings.json"

        if not settings_file.exists():
            scope = "globally" if global_ else "locally"
            return True, f"No settings file found {scope}"

        try:
            claude_settings = ClaudeSettings.load(settings_file)
        except (json.JSONDecodeError, ValueError):
            return False, f"Invalid JSON in {settings_file}"

        removed_any = claude_settings.hooks.remove_slopometry_hooks()
        claude_settings.permissions.allow = [
            cmd for cmd in claude_settings.permissions.allow if cmd not in self.WHITELISTED_COMMANDS
        ]

        try:
            claude_settings.save(settings_file)
        except Exception as e:
            return False, f"Failed to write settings: {e}"

        scope = "globally" if global_ else "locally"
        if removed_any:
            return True, f"Slopometry hooks removed {scope} from {settings_file}"
        else:
            return True, f"No slopometry hooks found to remove {scope}"

    def check_hooks_installed(self, settings_file: Path) -> bool:
        """Check if slopometry hooks are installed in a settings file."""
        try:
            claude_settings = ClaudeSettings.load(settings_file)
            return claude_settings.hooks.has_slopometry_hooks()
        except (json.JSONDecodeError, ValueError):
            return False

    def get_installation_status(self) -> dict[str, bool | str]:
        """Get installation status for global and local hooks."""
        global_settings = Path.home() / ".claude" / "settings.json"
        local_settings = Path.cwd() / ".claude" / "settings.json"

        return {
            "global": self.check_hooks_installed(global_settings),
            "local": self.check_hooks_installed(local_settings),
            "global_path": str(global_settings),
            "local_path": str(local_settings),
        }
