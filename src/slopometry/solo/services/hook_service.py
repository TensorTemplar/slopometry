"""Hook management service for Claude Code integration."""

import json
from datetime import datetime
from pathlib import Path

from slopometry.core.settings import settings


class HookService:
    """Handles Claude Code hook installation and management."""

    # Commands to whitelist so agents can investigate session details
    WHITELISTED_COMMANDS = [
        "Bash(slopometry solo:*)",
        "Bash(slopometry solo show:*)",
    ]

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

    def install_hooks(self, global_: bool = False) -> tuple[bool, str]:
        """Install slopometry hooks into Claude Code settings.

        Returns:
            Tuple of (success, message)
        """
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
                h
                for h in existing_settings["hooks"][hook_type]
                if not (
                    isinstance(h.get("hooks"), list)
                    and any("slopometry hook-" in hook.get("command", "") for hook in h.get("hooks", []))
                )
            ]

            existing_settings["hooks"][hook_type].extend(hook_configs)

        # Add slopometry commands to permissions.allow so agents can investigate
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

        scope = "globally" if global_ else "locally"
        return True, f"Slopometry hooks installed {scope} in {settings_file}"

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
                h
                for h in settings_data["hooks"][hook_type]
                if not (
                    isinstance(h.get("hooks"), list)
                    and any("slopometry hook-" in hook.get("command", "") for hook in h.get("hooks", []))
                )
            ]
            if len(settings_data["hooks"][hook_type]) < original_length:
                removed_any = True

        settings_data["hooks"] = {k: v for k, v in settings_data["hooks"].items() if v}

        if not settings_data["hooks"]:
            del settings_data["hooks"]

        # Remove slopometry commands from permissions.allow
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
            for hook_type in hooks:
                for hook_config in hooks[hook_type]:
                    if isinstance(hook_config.get("hooks"), list):
                        for hook in hook_config["hooks"]:
                            command = hook.get("command", "")
                            if "slopometry hook-" in command:
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
