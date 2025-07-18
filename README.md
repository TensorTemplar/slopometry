# Slopometry

A tool that lurks in the shadows, tracks and analyzes Claude Code sessions providing metrics that none of you knew you needed.

![slopometry-logo](assets/slopometry-logo.jpg)  


## Customer testimonials

### Claude Sonnet 4
![claude sonnet feedback](assets/sonnet.png)  
*"Amazing tool for tracking my own cognitive complexity!"*  
— C. Sonnet, main-author

### Claude Opus  
![opus feedback](assets/opus.png)  
*"Finally, I can see when I'm overcomplicating things."*  
— C. Opus, overpaid, infrequent contributor

### TensorTemplar
*"Previously i had to READ CODE and DECIDE WHEN TO RUN SLASH COMMANDS MYSELF, but now i just periodically prompt 'Cmon, claude, you know what you did...'"*  
— TensorTemplar, insignificant idea person for this tool

### sherbie
*"Let's slop up all the things."*
— sherbie, opinionated SDET

## Installation

### Install claude code (needs an account or api key)

```bash
curl -fsSL http://claude.ai/install.sh | bash
```

### Install slopometry as a uv tool

```bash
# Install as a global tool
uv tool install git+https://github.com/TensorTemplar/slopometry.git

# Or install from a local directory
git clone https://github.com/TensorTemplar/slopometry
cd slopometry
uv tool install .
```

## Quick Start

Note: tested on Ubuntu linux 24.04.1

```bash
# Install hooks globally (recommended)
slopometry install --global

# Use Claude normally
claude

# View tracked sessions
slopometry solo ls
slopometry solo show <session_id>

# View latest session
slopometry latest
```

![slopometry-roles.png](assets/slopometry-roles.png)  

## Shell Completion

Enable autocompletion for your shell:

```bash
# For bash
slopometry shell-completion bash

# For zsh  
slopometry shell-completion zsh

# For fish
slopometry shell-completion fish
```

The command will show you the exact instructions to add to your shell configuration.


## Upgrading

### Upgrade the uv tool

```bash
# Uninstall and reinstall to get the latest version
uv tool uninstall slopometry
uv tool install git+https://github.com/TensorTemplar/slopometry.git

# Or if installed from local directory
cd slopometry
git pull
uv tool uninstall slopometry
uv tool install .  --refresh

# Note: After upgrading, you may need to reinstall hooks if the default config changed
slopometry install
```

## Configuration

Slopometry can be configured using environment variables or a `.env` file:

1. **Global configuration**: `~/.config/slopometry/.env`
2. **Project-specific**: `.env` in your project directory

```bash
# Create config directory and copy example config
mkdir -p ~/.config/slopometry

# For solo-leveler users (basic session tracking):
curl -o ~/.config/slopometry/.env https://raw.githubusercontent.com/TensorTemplar/slopometry/main/.env.solo.example

# For summoner users (advanced experimentation):
curl -o ~/.config/slopometry/.env https://raw.githubusercontent.com/TensorTemplar/slopometry/main/.env.summoner.example

# Or if you have the repo cloned:
# cp .env.solo.example ~/.config/slopometry/.env
# cp .env.summoner.example ~/.config/slopometry/.env

# Edit ~/.config/slopometry/.env with your preferences
```

## Features

### Amplify your vibe coding session with some useful metrics

```bash
slopometry latest
```
<details>
Will show some metrics since the session start of the newest `claude code` session

![session statistics](assets/session-stat.png)  

![complexity metrics (CC)](assets/cc.png)  

![plan evolution](assets/plan-evolution.png)  

</details>

## Here be powerusers

### Use Agents to create user-stories for full-rewrites from existing features

1. configure a openai-compatible llm gateway or your life will be hard (litellm works)

<details>
set llm values in .env

```bash
SLOPOMETRY_USER_STORY_AGENTS=["o3", "claude-opus-4", "gemini-2.5-pro"]
SLOPOMETRY_LLM_PROXY_URL=http://proxy-url
SLOPOMETRY_LLM_PROXY_API_KEY=sk-whatever
```
</details>

2. chose how many commits back you want to go, ideally you should go back to a clean, atomic feature boundary

```bash
slopometry summoner userstorify --base-commit HEAD~2 --head-commit HEAD
```

If the vibes are not good update your `.env` with `SLOPOMETRY_INTERACTIVE_RATING_ENABLED=True`
Otherwise ratings are hardcoded to 3/5

![userstorify](assets/userstorify.png)

3. Analyze the same commit range for a future baseline

![analyze-commits](assets/analyze-commits.png)

NOTE: these are all saved to a local sqlite and can be exported later

### Dataset Collection & Export

Slopometry will automatically store stats, git diffs and AI-generated user stories as training data locally,
but you can also export it to parquet or huggingface for future training or sharing.

HF uploads are auto-tagged with `slopometry` and the type of dataset, e.g. `userstorify` or `analyze-commits`

<details>
```bash
# Generate user stories using multiple AI models (o3, claude-opus-4, gemini-2.5-pro) - requires extra settings
slopometry summoner userstorify --base-commit HEAD~5 --head-commit HEAD

# View collected data
slopometry summoner dataset-stats
slopometry summoner dataset-entries --limit 10

# Export to Parquet format
slopometry summoner dataset-export --output my_dataset.parquet

# Export and upload to Hugging Face
slopometry summoner dataset-export --upload-to-hf --hf-repo username/dataset-name
```

**Configuration for dataset features:**
- `SLOPOMETRY_INTERACTIVE_RATING_ENABLED=true` - Enable human rating of generated user stories
- `SLOPOMETRY_HF_TOKEN=your_token` - Hugging Face API token for uploads
- `SLOPOMETRY_HF_DEFAULT_REPO=username/repo-name` - Default HF repository
- `SLOPOMETRY_LLM_PROXY_URL=http://localhost:8000` - LLM gateway URL
- `SLOPOMETRY_LLM_PROXY_API_KEY=your_key` - LLM gateway API key

</details>

### Development Installation

```bash
git clone https://github.com/TensorTemplar/slopometry
cd slopometry
uv sync --extra dev
```

### Installation Management
- `slopometry install [--global|--local]` - Install tracking hooks
- `slopometry uninstall [--global|--local]` - Remove tracking hooks
- `slopometry status` - Check installation status

### Session Analysis  
- `slopometry solo ls [--limit N]` - List recent sessions
- `slopometry solo show <session-id>` - Show detailed session statistics
- `slopometry latest` - Show latest session statistics

### Dataset Generation & Export
- `slopometry summoner userstorify [--base-commit] [--head-commit]` - Generate user stories using multiple AI models
- `slopometry summoner dataset-entries [--limit N]` - View recent dataset entries
- `slopometry summoner dataset-stats` - Show dataset statistics
- `slopometry summoner dataset-export [--output] [--upload-to-hf] [--hf-repo]` - Export to Parquet and optionally upload to Hugging Face

### Experiment & Analysis
- `slopometry summoner analyze-commits [--base-commit] [--head-commit]` - Analyze complexity evolution
- `slopometry summoner run-experiments [--commits N] [--max-workers N]` - Run parallel experiments
- `slopometry summoner list-features [--limit N]` - List detected feature boundaries from merge commits

### Complexity Analysis Configuration
Configure complexity analysis via environment variables:
- `SLOPOMETRY_ENABLE_COMPLEXITY_ANALYSIS=true` - Collect complexity metrics (default: `true`)
- `SLOPOMETRY_ENABLE_COMPLEXITY_FEEDBACK=false` - Provide feedback to Claude (default: `false`)

Recommended: Keep analysis enabled for data collection, disable feedback for uninterrupted workflow.


Customize via `.env` file or environment variables:

- `SLOPOMETRY_DATABASE_PATH`: Custom database location (optional)
  - Default locations:
    - Linux: `~/.local/share/slopometry/slopometry.db`
    - macOS: `~/Library/Application Support/slopometry/slopometry.db`  
    - Windows: `%LOCALAPPDATA%\slopometry\slopometry.db`
- `SLOPOMETRY_PYTHON_EXECUTABLE`: Python command for hooks (default: uses uv tool's python)
- `SLOPOMETRY_SESSION_ID_PREFIX`: Custom session ID prefix
- `SLOPOMETRY_ENABLE_COMPLEXITY_ANALYSIS`: Collect complexity metrics (default: `true`)
- `SLOPOMETRY_ENABLE_COMPLEXITY_FEEDBACK`: Provide feedback to Claude (default: `false`)

## Architecture

- `models.py`: Pydantic models for events and statistics
- `database.py`: SQLite storage with session management
- `hook_handler.py`: Script invoked by Claude Code for each hook event
- `cli.py`: Click-based CLI interface with install/uninstall commands
- `settings.py`: Configuration management with uv compatibility

## Roadmap

[x] - Actually make a package so people can install this   
[ ] - Add hindsight-justified user stories with acceptance criteria based off of future commits  
[x] - Add plan evolution log based on claude's todo shenenigans   
[ ] - Use [NFP-CLI](https://tensortemplar.substack.com/p/humans-are-no-longer-embodied-amortization) (TM) training objective over plans with complexity metrics informing a process reward, while doing huge subtree rollouts just to win an argument on the internet  
[ ] - Add LLM-as-judge feedback over style guide as policy  
[ ] - Not go bankrupt from having to maintain open source in my free time, no wait...
