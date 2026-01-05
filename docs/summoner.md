# Summoner: Advanced Experimentation Features

The `summoner` persona provides advanced experimentation features for code quality analysis, user story generation, and cross-project comparison.

## Requirements

### Hard Requirements

- **Git**: All summoner commands require git. The repository must be initialized with at least one commit.
- **Python codebase**: Complexity analysis currently only supports Python files.

### LLM Configuration

User story generation and some analysis features require external LLM access. Configure in your `.env`:

```bash
# Required for userstorify and LLM-based features
SLOPOMETRY_LLM_PROXY_URL=https://your-proxy.example.com
SLOPOMETRY_LLM_PROXY_API_KEY=your-api-key
SLOPOMETRY_LLM_RESPONSES_URL=https://your-proxy.example.com/responses

# Disable LLM features (runs in offline mode)
SLOPOMETRY_OFFLINE_MODE=true
```

Without LLM configuration, the following commands will fail:
- `summoner userstorify`
- Any command with `--with-user-stories` flag

Commands that work without LLM:
- `summoner current-impact`
- `summoner analyze-commits`
- `summoner compare-projects`
- `summoner qpe`

## Installation

```bash
# For summoner users (advanced experimentation):
mkdir -p ~/.config/slopometry
curl -o ~/.config/slopometry/.env https://raw.githubusercontent.com/TensorTemplar/slopometry/main/.env.summoner.example

# Or if you have the repo cloned:
cp .env.summoner.example ~/.config/slopometry/.env

# Edit with your LLM proxy credentials
```

## Commands

### Current Impact Analysis

Analyze the last 100 commits for trend analysis caching vs. current changes:

```bash
slopometry summoner current-impact
```

### User Story Generation

Generate user stories from git diffs using AI:

```bash
# From a specific commit
slopometry summoner userstorify --base-commit abc1234

# From current changes
slopometry summoner userstorify
```

### QPE (Quality-Per-Effort) Score

Calculate the QPE score for a repository:

```bash
slopometry summoner qpe
slopometry summoner qpe --repo-path /path/to/project
```

### Cross-Project Comparison

Compare QPE scores across multiple projects with a persistent leaderboard:

```bash
# Show current leaderboard
slopometry summoner compare-projects

# Add a project to the leaderboard
slopometry summoner compare-projects --append /path/to/project

# Add current directory
slopometry summoner compare-projects --append .

# Add multiple projects
slopometry summoner compare-projects --append /path/a --append /path/b
```

The leaderboard persists entries with git commit hash tracking to monitor quality over time.

### User Story Dataset Management

```bash
# View collection statistics
slopometry summoner user-story-stats

# Browse recent entries
slopometry summoner list-user-stories

# Export to Parquet
slopometry summoner user-story-export

# Export and upload to Hugging Face
slopometry summoner user-story-export --upload-to-hf --hf-repo username/dataset-name
```

## Running Tests with LLM Integration

By default, LLM integration tests are skipped because `offline_mode` is enabled. To run the full test suite including LLM tests:

```bash
# Set up credentials in .env (copy from example)
cp .env.summoner.example .env
# Edit .env with your LLM proxy credentials

# Run tests with offline mode disabled
SLOPOMETRY_OFFLINE_MODE=false uv run pytest tests/test_llm_integration.py -v
```

The integration tests make real API calls to configured LLM providers and verify that agents return valid responses.
