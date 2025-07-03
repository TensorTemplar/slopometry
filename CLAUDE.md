# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slopometry is a Python CLI tool that tracks and analyzes Claude Code sessions by monitoring hook invocations. It collects statistics about tool usage, timing, and errors.

## Development Setup

```bash
# Install with development dependencies
uv sync --all-extras

# The CLI is available as 'slopometry' after installation

# Configure settings (optional)
cp .env.example .env
# Edit .env to customize settings
```

## Key Architecture

### Core Components
- **CLI** (`src/slopometry/cli.py`): Click-based interface with commands: install, uninstall, list, show, status
- **Database** (`src/slopometry/database.py`): SQLite storage (default: `.claude/slopometry.db` in project dir)
- **Hook Handler** (`src/slopometry/hook_handler.py`): Script invoked by Claude Code hooks to capture events
- **Models** (`src/slopometry/models.py`): Pydantic models for HookEvent, SessionStatistics
- **Settings** (`src/slopometry/settings.py`): Pydantic-settings configuration with .env support

### How It Works
1. `slopometry install` configures Claude Code hooks in settings.json
2. Each tool invocation automatically triggers the hook handler script
3. Events are persisted to SQLite with session IDs
4. Statistics are calculated and displayed via Rich tables/trees

## Important Implementation Details

- Session IDs are provided directly by Claude Code (no generated IDs)
- Hook handler reads JSON from stdin (provided by Claude Code)
- Tool name mapping is done via `TOOL_TYPE_MAP` in hook_handler.py
- Database uses sqlite-utils for schema-less flexibility
- All timestamps are stored as ISO format strings

## Testing a Hook Handler Change

When modifying the hook handler, test it manually using the actual Claude Code hook schema:
```bash
# Test PreToolUse hook
echo '{"session_id": "test123", "transcript_path": "/tmp/transcript.jsonl", "tool_name": "Bash", "tool_input": {"command": "ls"}}' | uv run python -m slopometry.hook_handler

# Test PostToolUse hook  
echo '{"session_id": "test123", "transcript_path": "/tmp/transcript.jsonl", "tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": {"success": true}}' | uv run python -m slopometry.hook_handler
```

## Adding New Tool Types

1. Add to `ToolType` enum in models.py
2. Update `TOOL_TYPE_MAP` in hook_handler.py
3. No database migration needed (sqlite-utils handles schema)
  
## Upstream Hook docs  
available in `./claude-hooks-doc.md`  

## Development guidelines  

You workflow should be incremental with a stepping back review phase after each milestone is hit (such as a new feature implemented or changed). After you gave your summary on the current work scope, run a subtask to review based on the following guidelines:
- When implementing or updating tests think about the larger picture and what the desired intent of the test is. The goal is not to make tests pass but to showcase the real software's behavior explicitly.
- Backwards compatibility is never required for this research codebase. Always clean up temporary files or update original ones with new implementations that replace previous logic. 
- Avoid cognitive complexity introduced from unnecessary branching or excessive api or config option flexibility
- test naming should follow this pattern: test files named `test_<name_of_file_under_test>.py` and each test case is `test_<name_of_function>__<expected_behavior_when_used_in_this_way>`
- Note any design choices or decisions you are not sure about and request the user for comments
- Look for and point out dead or duplicated code branches that could be DRYed
- Double check the README.md or other documentation to remove outdated sections
- Be wary of unconstrained and overly generic types in arguments and returns. Introduce ADT using dedicated Pydantic BaseModel and a domain-driven approach, if the current ones are not sufficient. Use pattern matches on these types instead of hasattr/getattr decomposition
- **Leverage Pydantic validation**: When adding new configuration parameters or architectural constraints, use Pydantic field validators (`@field_validator`) to catch errors early with helpful messages
- **Use domain models**: Replace isinstance/hasattr patterns with domain objects that use pydantic's `BaseModel`
- Always run tests with pytest as final verification step for newly added code
- any use of `hasattr` and `.get()` with defaults and similar existence checks on objects are code smells (!) and indication that proper configuration or domain object models need to be reviewed