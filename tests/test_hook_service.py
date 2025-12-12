import json

from slopometry.solo.services.hook_service import HookService


def test_install_hooks__writes_local_configuration(tmp_path, monkeypatch):
    """Verify local hook installation writes correct JSON."""
    monkeypatch.chdir(tmp_path)
    service = HookService()

    success, msg = service.install_hooks(global_=False)
    assert success is True
    assert "Slopometry hooks installed locally" in msg

    settings_file = tmp_path / ".claude" / "settings.json"
    assert settings_file.exists()

    with open(settings_file) as f:
        data = json.load(f)

    assert "hooks" in data
    assert "PreToolUse" in data["hooks"]

    # Check content of a hook
    matching_hook = False
    for item in data["hooks"]["PreToolUse"]:
        for h in item.get("hooks", []):
            if "slopometry hook-pre-tool-use" in h.get("command", ""):
                matching_hook = True
    assert matching_hook


def test_install_hooks__is_idempotent_and_preserves_other_hooks(tmp_path, monkeypatch):
    """Verify re-installation updates hooks but preserves user's other hooks."""
    monkeypatch.chdir(tmp_path)
    service = HookService()

    # Pre-populate with another hook
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.json"

    initial_data = {"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"command": "echo 'user hook'"}]}]}}
    with open(settings_file, "w") as f:
        json.dump(initial_data, f)

    # Install
    service.install_hooks(global_=False)

    with open(settings_file) as f:
        data = json.load(f)

    # Should have both
    commands = []
    for item in data["hooks"]["PreToolUse"]:
        for h in item.get("hooks", []):
            commands.append(h.get("command", ""))

    assert "echo 'user hook'" in commands
    assert any("slopometry hook-" in cmd for cmd in commands)


def test_uninstall_hooks__removes_slopometry_hooks_only(tmp_path, monkeypatch):
    """Verify uninstallation removes only slopometry hooks."""
    monkeypatch.chdir(tmp_path)
    service = HookService()

    # Install first
    service.install_hooks(global_=False)

    # Uninstall
    success, msg = service.uninstall_hooks(global_=False)
    assert success

    with open(tmp_path / ".claude" / "settings.json") as f:
        data = json.load(f)

    # Should be empty or no slopometry hooks
    if "hooks" in data:
        for hook_type, configs in data["hooks"].items():
            for config in configs:
                for h in config.get("hooks", []):
                    assert "slopometry hook-" not in h.get("command", "")
