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

    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.json"

    initial_data = {"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"command": "echo 'user hook'"}]}]}}
    with open(settings_file, "w") as f:
        json.dump(initial_data, f)

    service.install_hooks(global_=False)

    with open(settings_file) as f:
        data = json.load(f)

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

    service.install_hooks(global_=False)
    success, msg = service.uninstall_hooks(global_=False)
    assert success

    with open(tmp_path / ".claude" / "settings.json") as f:
        data = json.load(f)

    if "hooks" in data:
        for hook_type, configs in data["hooks"].items():
            for config in configs:
                for h in config.get("hooks", []):
                    assert "slopometry hook-" not in h.get("command", "")


def test_update_gitignore__creates_new_file_when_missing(tmp_path, monkeypatch):
    """Creates .gitignore with slopometry entry when file doesn't exist."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    service = HookService()
    updated, message = service._update_gitignore(tmp_path)

    assert updated is True
    assert ".slopometry/" in message

    gitignore = (tmp_path / ".gitignore").read_text()
    assert ".slopometry/" in gitignore
    assert "# slopometry" in gitignore


def test_update_gitignore__appends_to_existing_file(tmp_path, monkeypatch):
    """Appends to existing .gitignore without duplicating."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")

    service = HookService()
    updated, message = service._update_gitignore(tmp_path)

    assert updated is True
    gitignore = (tmp_path / ".gitignore").read_text()
    assert "*.pyc" in gitignore
    assert ".slopometry/" in gitignore


def test_update_gitignore__skips_if_already_present(tmp_path, monkeypatch):
    """Does not add duplicate entry if already present."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(".slopometry/\n")

    service = HookService()
    updated, message = service._update_gitignore(tmp_path)

    assert updated is False
    assert message is None

    gitignore = (tmp_path / ".gitignore").read_text()
    assert gitignore.count(".slopometry/") == 1


def test_update_gitignore__skips_non_git_directory(tmp_path, monkeypatch):
    """Does not create .gitignore in non-git directories."""
    monkeypatch.chdir(tmp_path)

    service = HookService()
    updated, message = service._update_gitignore(tmp_path)

    assert updated is False
    assert message is None
    assert not (tmp_path / ".gitignore").exists()


def test_install_hooks__updates_gitignore_for_local_install(tmp_path, monkeypatch):
    """Local install should update .gitignore if in git repo."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    service = HookService()
    success, message = service.install_hooks(global_=False)

    assert success is True
    assert ".slopometry/" in message
    assert (tmp_path / ".gitignore").exists()
    assert ".slopometry/" in (tmp_path / ".gitignore").read_text()


def test_install_hooks__preserves_unknown_fields(tmp_path, monkeypatch):
    """Verify install preserves unknown top-level fields and unknown hook types."""
    monkeypatch.chdir(tmp_path)
    service = HookService()

    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.json"

    initial_data = {
        "hooks": {
            "PreToolUse": [{"matcher": ".*", "hooks": [{"command": "echo 'user hook'"}]}],
            "UnknownHookType": [{"matcher": "special", "hooks": [{"command": "custom"}]}],
        },
        "permissions": {"allow": ["Bash(echo:*)"], "deny": ["Bash(rm:*)"]},
        "unknown_top_level": "should_be_preserved",
        "another_unknown": {"nested": "value"},
    }
    with open(settings_file, "w") as f:
        json.dump(initial_data, f)

    service.install_hooks(global_=False)

    with open(settings_file) as f:
        data = json.load(f)

    assert data["unknown_top_level"] == "should_be_preserved"
    assert data["another_unknown"] == {"nested": "value"}
    assert "UnknownHookType" in data["hooks"]
    assert data["hooks"]["UnknownHookType"] == [{"matcher": "special", "hooks": [{"command": "custom"}]}]
    assert "deny" in data["permissions"]
    assert data["permissions"]["deny"] == ["Bash(rm:*)"]


def test_uninstall_hooks__preserves_unknown_fields(tmp_path, monkeypatch):
    """Verify uninstall preserves unknown fields."""
    monkeypatch.chdir(tmp_path)
    service = HookService()

    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.json"

    initial_data = {
        "hooks": {
            "PreToolUse": [
                {"matcher": ".*", "hooks": [{"command": "slopometry hook-pre-tool-use"}]},
                {"matcher": ".*", "hooks": [{"command": "echo 'user hook'"}]},
            ],
            "CustomHookType": [{"hooks": [{"command": "special"}]}],
        },
        "custom_setting": True,
    }
    with open(settings_file, "w") as f:
        json.dump(initial_data, f)

    service.uninstall_hooks(global_=False)

    with open(settings_file) as f:
        data = json.load(f)

    assert data["custom_setting"] is True
    assert "CustomHookType" in data["hooks"]
    assert data["hooks"]["CustomHookType"] == [{"hooks": [{"command": "special"}]}]
    assert any("echo 'user hook'" in h.get("command", "") for item in data["hooks"]["PreToolUse"] for h in item.get("hooks", []))
