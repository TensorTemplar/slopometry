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
# For development: copy to project directory
cp .env.solo.example .env  # For basic session tracking
# OR
cp .env.summoner.example .env  # For advanced experimentation

mkdir -p ~/.config/slopometry
cp .env.solo.example ~/.config/slopometry/.env  # For basic session tracking
# OR
cp .env.summoner.example ~/.config/slopometry/.env  # For advanced experimentation
```

## Installation as uv Tool

When installing slopometry as a global uv tool:
```bash
uv tool install .
```


## Key Architecture

### Core Components
- **CLI** (`src/slopometry/cli.py`): Hybrid CLI with flat core commands (install, uninstall, status, latest, shell-completion) and persona subcommands (solo, summoner)
- **Database** (`src/slopometry/core/database.py`): SQLite storage with platform-specific default locations
- **Hook Handler** (`src/slopometry/core/hook_handler.py`): Script invoked by Claude Code hooks to capture events
- **Models** (`src/slopometry/core/models.py`): Pydantic models for HookEvent, SessionStatistics
- **Settings** (`src/slopometry/core/settings.py`): Pydantic-settings configuration with .env support
- **LLM Wrapper** (`src/slopometry/summoner/services/llm_wrapper.py`): AI agents for analyzing git diffs and generating user stories

### How It Works
1. `slopometry install` configures Claude Code hooks in settings.json
2. Each tool invocation automatically triggers the hook handler script
3. Events are persisted to SQLite with session IDs
4. Statistics are calculated and displayed via Rich tables/trees

## Important Implementation Details

- Session IDs are provided directly by Claude Code (no generated IDs)
- Hook handler reads JSON from stdin (provided by Claude Code)
- Tool name mapping is done via `TOOL_TYPE_MAP` in hook_handler.py
- Database uses raw SQL with migration support for flexibility
- All timestamps are stored as ISO format strings

## Dataset Protocol

The dataset protocol automatically collects diff/user story pairs for AI training and analysis:

### Automatic Collection
- Every `summoner userstorify` call automatically saves to the dataset
- Includes git diff, generated user stories, model used, and prompt template
- Default rating of 3/5 for non-interactive mode

### Interactive Rating (Optional)
- Enable with `SLOPOMETRY_INTERACTIVE_RATING_ENABLED=true`
- Prompts user to rate user stories (1-5) and provide improvement guidelines
- Designed for human oversight and quality control

### Dataset Management
- `summoner user-story-stats`: View collection statistics and rating distribution
- `summoner list-user-stories`: Browse recent entries with metadata
- `summoner user-story-export`: Export user stories to Parquet format
- All data stored in SQLite with proper indexing for performance

### Export & Sharing
Export dataset locally:
```bash
# Export to Parquet
slopometry summoner user-story-export

# Export with custom path
slopometry summoner user-story-export --output my_user_stories.parquet
```

Upload to Hugging Face:
```bash
# Export and upload in one command
slopometry summoner user-story-export --upload-to-hf --hf-repo username/dataset-name

# With configured settings (SLOPOMETRY_HF_TOKEN and SLOPOMETRY_HF_DEFAULT_REPO)
slopometry summoner user-story-export --upload-to-hf
```

The dataset is automatically tagged with: `slopometry`, `userstorify`, `code-generation`, `user-stories`

### Optional Dependencies
For dataset export functionality:
```bash
# Basic export support
uv pip install "slopometry[dataset]"

# Full Hugging Face support
uv pip install "slopometry[huggingface]"
```

## Testing a Hook Handler Change

When modifying the hook handler, test it manually using the actual Claude Code hook schema:
```bash
# Test PreToolUse hook
echo '{"session_id": "test123", "transcript_path": "/tmp/transcript.jsonl", "tool_name": "Bash", "tool_input": {"command": "ls"}}' | slopometry hook-handler

# Test PostToolUse hook
echo '{"session_id": "test123", "transcript_path": "/tmp/transcript.jsonl", "tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": {"success": true}}' | slopometry hook-handler
```

## Adding New Tool Types

1. Add to `ToolType` enum in models.py
2. Update `TOOL_TYPE_MAP` in hook_handler.py
3. No database migration needed (sqlite-utils handles schema)

## Experiment Tracking

The experiment tracking feature includes:

### CLI Commands
- `summoner run-experiments`: Run parallel experiments across commit history
- `summoner analyze-commits`: Analyze complexity evolution between commits  
- `summoner userstorify`: Generate user stories from commit diffs using AI
- `summoner list-features`: Detect feature boundaries from merge commits
- `summoner user-story-stats`: Show statistics about collected diff/user story dataset
- `summoner list-user-stories`: Show recent user story entries
- `summoner user-story-export`: Export user stories to Parquet with optional HF upload
- `summoner list-experiments`: List all experiment runs
- `summoner show-experiment <id>`: Show detailed progress for an experiment
- `solo ls`: List recent sessions
- `solo show <session-id>`: Show detailed session statistics
- `latest`: Show latest session statistics

### Key Components
- **CLI Calculator**: Measures "Completeness Likelihood Improval" (0-1.0 scale)
- **Extended Complexity Analyzer**: Collects Cyclomatic Complexity + Halstead + MI metrics
- **Worktree Manager**: Creates isolated git environments for parallel experiments
- **Experiment Orchestrator**: Coordinates parallel experiment execution

### Important Notes
- All `radon` calls use `uvx radon` to ensure proper uv environment execution
- Experiments create temporary git worktrees for isolation
- CLI scores: 1.0 = perfect match, <0 = overshooting target (prevents overfitting)
- Database handles duplicate analysis runs gracefully
  
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