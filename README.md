# Slopometry

![assets/slopometry-logo.jpg](slopometry)  

A Python CLI tool that automatically tracks and analyzes Claude Code sessions by monitoring hook invocations. Works seamlessly with uvx for easy installation and usage.

## Installation

### Via uvx (Recommended)
```bash
# Install and use directly with uvx
uvx slopometry install     # One-time setup
claude                     # Normal usage - automatically tracked!
uvx slopometry list        # View tracked sessions
```

### Via uv/pip
```bash
# Install globally
uv tool install slopometry
# Or with pip
pip install slopometry
```

## Quick Start

1. **Install hooks** (one-time setup):
   ```bash
   slopometry install          # Install globally (~/.claude/settings.json)
   # or
   slopometry install --local  # Install for current project only
   ```

2. **Use Claude normally** - all sessions are automatically tracked:
   ```bash
   claude
   claude /path/to/project
   claude --help
   ```

3. **View your metrics**:
   ```bash
   slopometry list             # List recent sessions
   slopometry show <session-id> # Detailed session view
   slopometry status           # Check installation status
   ```

## Commands

### Installation Management
- `slopometry install [--global|--local]` - Install tracking hooks
- `slopometry uninstall [--global|--local]` - Remove tracking hooks
- `slopometry status` - Check installation status

### Session Analysis  
- `slopometry list [--limit N]` - List recent sessions
- `slopometry show <session-id>` - Show detailed session statistics

## How it works

1. **One-time setup**: `slopometry install` adds hooks to Claude Code settings
2. **Automatic tracking**: Every Claude session is tracked without user intervention
3. **Smart sessions**: Sessions are auto-detected based on process activity and timing
4. **Rich analytics**: View tool usage, timing, and error statistics

Session IDs use format: `YYYYMMDD_HHMMSS_PID` for easy identification.

## Features

- **Zero-friction tracking**: Install once, track automatically
- **Smart session detection**: Based on process activity and timing gaps
- **Rich terminal output**: Beautiful tables and trees via Rich
- **Tool usage analytics**: See which tools you use most
- **Performance metrics**: Average duration per tool type
- **Error tracking**: Monitor hook failures and issues
- **uvx compatible**: Perfect for standalone tool usage

## Configuration

Customize via `.env` file or environment variables:

- `SLOPOMETRY_DATABASE_PATH`: Database location (default: `.claude/slopometry.db`)
- `SLOPOMETRY_PYTHON_EXECUTABLE`: Python command for hooks (default: `uv run python`)
- `SLOPOMETRY_SESSION_ID_PREFIX`: Custom session ID prefix

## Architecture

- `models.py`: Pydantic models for events and statistics
- `database.py`: SQLite storage with session management
- `hook_handler.py`: Script invoked by Claude Code for each hook event
- `cli.py`: Click-based CLI interface with install/uninstall commands
- `settings.py`: Configuration management with uv compatibility

## Development

```bash
git clone <repo>
cd slopometry
uv sync --all-extras

# Test the CLI
uv run slopometry install
uv run slopometry status
```