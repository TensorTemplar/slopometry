"""Hook management service for Claude Code integration."""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from slopometry.core.settings import get_default_config_dir, get_default_data_dir, settings


class HookCommand(BaseModel):
    """A single hook command to execute."""

    type: str = "command"
    command: str


class HookConfig(BaseModel):
    """Configuration for a hook event handler."""

    matcher: str | None = None  # Only for PreToolUse/PostToolUse
    hooks: list[HookCommand]


class ClaudeSettingsHooks(BaseModel, extra="allow"):
    """Hooks section of Claude Code settings.json.

    Uses extra="allow" to tolerate unknown hook types from future Claude versions.
    """

    PreToolUse: list[HookConfig] = Field(default_factory=list)
    PostToolUse: list[HookConfig] = Field(default_factory=list)
    Notification: list[HookConfig] = Field(default_factory=list)
    Stop: list[HookConfig] = Field(default_factory=list)
    SubagentStop: list[HookConfig] = Field(default_factory=list)


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

    @staticmethod
    def _is_slopometry_hook_config(config: dict) -> bool:
        """Check if a hook config dict contains slopometry hooks.

        Args:
            config: Raw hook config dict from settings.json

        Returns:
            True if this config contains slopometry hook commands
        """
        try:
            parsed = HookConfig.model_validate(config)
            return any("slopometry hook-" in hook.command for hook in parsed.hooks)
        except Exception:
            return False

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

        settings_dir.mkdir(exist_ok=True)

        existing_settings = {}
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    existing_settings = json.load(f)
            except json.JSONDecodeError:
                return False, f"Invalid JSON in {settings_file}"

        if "hooks" in existing_settings and settings.backup_existing_settings:
            backup_file = settings_dir / f"settings.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, "w") as f:
                json.dump(existing_settings, f, indent=2)

        slopometry_hooks = self.create_hook_configuration()

        if "hooks" not in existing_settings:
            existing_settings["hooks"] = {}

        for hook_type, hook_configs in slopometry_hooks.items():
            if hook_type not in existing_settings["hooks"]:
                existing_settings["hooks"][hook_type] = []

            existing_settings["hooks"][hook_type] = [
                h for h in existing_settings["hooks"][hook_type] if not self._is_slopometry_hook_config(h)
            ]

            existing_settings["hooks"][hook_type].extend(hook_configs)

        if "permissions" not in existing_settings:
            existing_settings["permissions"] = {}
        if "allow" not in existing_settings["permissions"]:
            existing_settings["permissions"]["allow"] = []

        existing_allows = set(existing_settings["permissions"]["allow"])
        for cmd in self.WHITELISTED_COMMANDS:
            if cmd not in existing_allows:
                existing_settings["permissions"]["allow"].append(cmd)

        try:
            with open(settings_file, "w") as f:
                json.dump(existing_settings, f, indent=2)
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
            with open(settings_file) as f:
                settings_data = json.load(f)
        except json.JSONDecodeError:
            return False, f"Invalid JSON in {settings_file}"

        if "hooks" not in settings_data:
            return True, "No hooks configuration found"

        removed_any = False
        for hook_type in settings_data["hooks"]:
            original_length = len(settings_data["hooks"][hook_type])
            settings_data["hooks"][hook_type] = [
                h for h in settings_data["hooks"][hook_type] if not self._is_slopometry_hook_config(h)
            ]
            if len(settings_data["hooks"][hook_type]) < original_length:
                removed_any = True

        settings_data["hooks"] = {k: v for k, v in settings_data["hooks"].items() if v}

        if not settings_data["hooks"]:
            del settings_data["hooks"]

        if "permissions" in settings_data and "allow" in settings_data["permissions"]:
            settings_data["permissions"]["allow"] = [
                cmd for cmd in settings_data["permissions"]["allow"] if cmd not in self.WHITELISTED_COMMANDS
            ]
            if not settings_data["permissions"]["allow"]:
                del settings_data["permissions"]["allow"]
            if not settings_data["permissions"]:
                del settings_data["permissions"]

        try:
            with open(settings_file, "w") as f:
                json.dump(settings_data, f, indent=2)
        except Exception as e:
            return False, f"Failed to write settings: {e}"

        scope = "globally" if global_ else "locally"
        if removed_any:
            return True, f"Slopometry hooks removed {scope} from {settings_file}"
        else:
            return True, f"No slopometry hooks found to remove {scope}"

    def check_hooks_installed(self, settings_file: Path) -> bool:
        """Check if slopometry hooks are installed in a settings file."""
        if not settings_file.exists():
            return False

        try:
            with open(settings_file) as f:
                settings_data = json.load(f)

            hooks = settings_data.get("hooks", {})
            for hook_configs in hooks.values():
                for hook_config in hook_configs:
                    if self._is_slopometry_hook_config(hook_config):
                        return True
            return False
        except (json.JSONDecodeError, KeyError):
            return False

    def get_installation_status(self) -> dict[str, bool]:
        """Get installation status for global and local hooks."""
        global_settings = Path.home() / ".claude" / "settings.json"
        local_settings = Path.cwd() / ".claude" / "settings.json"

        return {
            "global": self.check_hooks_installed(global_settings),
            "local": self.check_hooks_installed(local_settings),
            "global_path": str(global_settings),
            "local_path": str(local_settings),
        }
