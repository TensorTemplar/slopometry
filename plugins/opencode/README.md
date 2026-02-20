# Slopometry OpenCode Plugin

Captures OpenCode session events in-process and forwards them to the slopometry CLI for storage, analysis, and mid-session feedback injection.

## How It Works

```
OpenCode (in-process)                    Slopometry (external)
┌─────────────────────┐                  ┌─────────────────────┐
│  Plugin hooks into:  │                  │                     │
│  - tool.execute.*    │  spawn + stdin   │  hook-opencode CLI  │
│  - bus events        │ ──────────────>  │  parses JSON        │
│  - promptAsync()     │  <── stdout ───  │  stores in DB       │
│                      │    (feedback)    │  generates feedback  │
└─────────────────────┘                  └─────────────────────┘
```

The plugin hooks into OpenCode's in-process event system and spawns
`slopometry hook-opencode --event-type <type>` per event, passing JSON on stdin.
This mirrors how slopometry integrates with Claude Code via shell hooks.

## Events Captured

| Event | Source | Data |
|-------|--------|------|
| `pre_tool_use` | `tool.execute.before` hook | tool name, session ID, call ID, args |
| `post_tool_use` | `tool.execute.after` hook | tool name, args, output (truncated), duration_ms |
| `message_updated` | `message.updated` bus event | per-message tokens, cost, model, agent |
| `todo_updated` | `todo.updated` bus event | full todo list with status/priority |
| `subagent_start` | `session.created` bus event (with parentID) | child session ID, parent ID |
| `stop` | `session.idle` bus event | aggregated tokens, cost, transcript via SDK |

## Feedback Injection

When slopometry returns feedback on stdout (code smells, context coverage warnings), the plugin injects it in two ways:

1. **Inline** — appended to tool output via `tool.execute.after` (visible with the tool result)
2. **Stop feedback** — on `session.idle`, if slopometry returns smell feedback, the plugin calls `client.session.promptAsync()` to send a synthetic user message that triggers a new agent turn to address the smells (mirrors Claude Code's blocking stop hook). An `awaitingFeedbackTurn` flag prevents the follow-up idle from looping, but re-arms for subsequent user turns.

## Installation

### Prerequisites

- **slopometry** must be installed and in PATH (`uv tool install slopometry` or from source)
- **OpenCode** v1.2+ with plugin support

### Method 1: Symlink into OpenCode config (recommended)

Find your OpenCode config directory:

```bash
# Usually one of:
# ~/.config/opencode/          (Linux default)
# $XDG_CONFIG_HOME/opencode/   (if XDG_CONFIG_HOME is set)
# ~/.opencode/                 (macOS / fallback)
```

Create a plugins directory and symlink:

```bash
mkdir -p $XDG_CONFIG_HOME/opencode/plugins
ln -sf /path/to/slopometry/plugins/opencode/index.ts \
       $XDG_CONFIG_HOME/opencode/plugins/slopometry.ts
```

OpenCode auto-discovers `plugins/*.ts` files in config directories and
auto-installs `@opencode-ai/plugin` (creates a managed `package.json`
and runs `bun install` in the config dir on startup).

### Method 2: file:// in opencode.json

Add the plugin path to `opencode.json` (project-level or in `.opencode/`):

```json
{
  "plugin": ["file:///path/to/slopometry/plugins/opencode"]
}
```

Then ensure dependencies are installed in the plugin directory:

```bash
cd /path/to/slopometry/plugins/opencode
bun install
```

### Method 3: slopometry install command

```bash
slopometry install --target opencode
```

This writes the `file://` plugin path into `opencode.json` in the current directory.

## Verification

1. Start an OpenCode session and use some tools (file reads, edits, bash commands)
2. Check that slopometry captured events:
   ```bash
   slopometry solo ls
   slopometry latest
   ```
3. Verify the session shows `source=opencode` in the database:
   ```bash
   slopometry solo show <session-id>
   ```

## Development

The plugin is a single TypeScript file (`index.ts`) that exports a `Plugin` function.
It has no build step — OpenCode (Bun) imports `.ts` files directly.

To iterate on the plugin:
1. Edit `index.ts`
2. Restart OpenCode (the plugin is loaded on startup)
3. Check `slopometry hook-opencode --help` for the CLI interface

To test the CLI handler in isolation:
```bash
echo '{"tool":"Bash","session_id":"test","call_id":"c1","args":{"command":"ls"}}' \
  | slopometry hook-opencode --event-type pre_tool_use
```
