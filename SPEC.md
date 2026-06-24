# Slopometry Memory System - Specification

## Overview

Add a memory system to `slopometry solo` that discovers Claude Code transcripts, extracts meaningful memory candidates using an LLM, and provides interactive management of memories per-project.

## Memory Types

Based on the guideline, memories are categorized into four types:

| Type | Description | Examples |
|------|-------------|----------|
| `user` | Who you are: role, expertise, stable preferences | "Sarah is a DevOps engineer specializing in k8s" |
| `feedback` | How I should work: corrections or confirmed approaches with the *why* | "Always check GPU availability before scheduling - learned from a 3am incident" |
| `project` | Ongoing work, goals, constraints not derivable from code/git | "We use dqlite voter topology for HA - not visible from files" |
| `reference` | Pointers to external resources | "Jira board: https://company.atlassian.net/board/TICKETS" |

### What NOT to Save
- Anything the repo already records (code structure, past fixes, git history, CLAUDE.md content)
- Things that only matter to the current conversation
- Reconstructable facts from code

### Hygiene Rules
- Convert relative dates to absolute
- Update existing memories rather than duplicate
- Delete memories that are wrong

---

## Feature 1: `slopometry solo find-memories`

### Purpose
Scan disk for Claude Code transcripts, filter to project-relevant ones, parse them to remove noise, generate memory candidates via LLM, and save to database.

### Transcript Discovery Paths

**Claude Code default locations:**
- Linux: `~/.claude/projects/`
- macOS: `~/Library/Application Support/Claude/projects/`
- Windows: `%APPDATA%\Claude\projects\`

**Slopometry saved transcripts:**
- `.slopometry/transcripts/` in project root

**Structure under project dirs:**
```
~/.claude/projects/<project_hash>/sessions/<session_id>/transcript.jsonl
```

### CLI Interface

```bash
slopometry solo find-memories [OPTIONS]

OPTIONS:
  --project-dir PATH          Project directory (default: cwd)
  --llm-endpoint URL          LLM endpoint (default: from env or localhost)
  --llm-model MODEL           Model name (default: gpt-4o-mini)
  --force                     Re-process already processed sessions
  --dry-run                   Show what would be done without doing it
  --min-importance SCORE      Minimum importance 0.0-1.0 (default: 0.5)
```

### Processing Pipeline

1. **Discovery Phase**
   - Scan Claude project directories
   - Scan `.slopometry/transcripts/` in project
   - Build list of `(session_id, transcript_path, project_dir)` tuples

2. **Filtering Phase**
   - Skip sessions already processed (unless `--force`)
   - Filter to sessions whose working directory matches `--project-dir`

3. **Parsing Phase**
   - Parse JSONL transcript
   - Remove noise: empty turns, auto-accepted suggestions, tool outputs > threshold
   - Extract conversation structure

4. **Memory Candidate Generation**
   - Send parsed conversation to LLM with the guideline prompt
   - Request JSON array of memory candidates

5. **Storage Phase**
   - Save memory candidates to `memories` table
   - Mark session as processed in `processed_memory_sessions` table

### LLM Prompt for Memory Extraction

```
You are analyzing a Claude Code session transcript to identify durable facts
that should be remembered across sessions.

MEMORY TYPES:
- user: Facts about the human's identity, role, expertise, stable preferences
- feedback: Guidance on how to work, corrections, confirmed approaches (always with WHY)
- project: Work goals, constraints, topology not derivable from code/git
- reference: External resource pointers (URLs, dashboards, tickets)

MEMORY CRITERIA:
A fact qualifies if "If I started fresh next session, would not knowing this
make me repeat a mistake, re-derive something hard, or act against preference?"
If YES → memory. If reconstructable from code or only relevant now → skip.

HYGIENE:
- Convert "19 days ago" to real date
- One fact per memory
- Include importance_score 0.0-1.0

Return a JSON array of memories:
[
  {
    "memory_type": "feedback",
    "content": "Always verify GPU availability before job submission - learned from a 3am incident where a job sat queued for 2 hours",
    "importance_score": 0.9,
    "source_context": "mentioned during discussion about cluster scheduling"
  }
]

Transcript to analyze:
<transcript_snippet>
```

---

## Feature 2: `slopometry solo show-memories`

### Purpose
List all memories for a project from the database and provide interactive management.

### CLI Interface

```bash
slopometry solo show-memories [OPTIONS]

OPTIONS:
  --project-dir PATH          Project directory (default: cwd)
  --type TYPE                 Filter by memory type (user|feedback|project|reference)
  --min-importance SCORE      Minimum importance score
  --limit N                   Max results (default: 50)
```

### Interactive Mode

When run without `--format`, enters interactive mode:

```
Slopometry Memories: ~/projects/myapp
┌─────────────────────────────────────────────────────────────────┐
│  [1] user        │ Importance: 0.85                            │
│  "Sarah is a DevOps engineer with 10 years experience in k8s"   │
│  Session: abc123 | Created: 2024-01-15                         │
├─────────────────────────────────────────────────────────────────┤
│  [2] feedback    │ Importance: 0.95                            │
│  "Check GPU availability before scheduling - 3am incident"      │
│  Session: def456 | Created: 2024-01-14                         │
└─────────────────────────────────────────────────────────────────┘

Actions: [r]etain  [d]elete  [e]dit  [f]ilter  [q]uit
>
```

**Interactive Commands:**
- `1`, `2`, etc. - Select a memory
- `r <id>` - Mark memory as retained (increases importance for similar)
- `d <id>` - Delete a memory
- `e <id>` - Edit memory content
- `f <type>` - Filter by type
- `q` - Quit

---

## Database Schema

### New Table: `memories`

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_dir TEXT NOT NULL,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('user', 'feedback', 'project', 'reference')),
    content TEXT NOT NULL,
    importance_score REAL NOT NULL DEFAULT 0.5,
    source_context TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    retained INTEGER NOT NULL DEFAULT 0,
    metadata TEXT
);

CREATE INDEX idx_memories_project_dir ON memories(project_dir);
CREATE INDEX idx_memories_session_id ON memories(session_id);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_importance ON memories(importance_score);
```

### New Table: `processed_memory_sessions`

```sql
CREATE TABLE processed_memory_sessions (
    session_id TEXT PRIMARY KEY,
    project_dir TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    memory_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, project_dir)
);
```

---

## Pydantic Models

```python
# models/memory.py

from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"

class MemoryEntry(BaseModel):
    id: str
    session_id: str
    project_dir: str
    memory_type: MemoryType
    content: str
    importance_score: float = Field(ge=0.0, le=1.0)
    source_context: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    retained: bool = False
    metadata: dict | None = None

class MemoryCandidate(BaseModel):
    memory_type: MemoryType
    content: str
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    source_context: str | None = None

class MemoryCreateRequest(BaseModel):
    session_id: str
    project_dir: str
    candidates: list[MemoryCandidate]
```

---

## Settings Extension

```python
# In settings.py

class Settings(BaseSettings):
    # ... existing fields ...

    # Memory system
    memory_llm_endpoint: str = Field(
        default="http://localhost:11434/v1",
        description="LLM endpoint for memory extraction"
    )
    memory_llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model for memory extraction"
    )
    memory_min_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )
    memory_retention_days: int = Field(
        default=365,
        description="Days to retain memories"
    )
```

---

## File Structure

```
src/slopometry/
  core/
    database.py         # Add memories table creation
    settings.py         # Add memory settings
  solo/
    services/
      memory_service.py    # Memory CRUD operations
      transcript_finder.py # Discovery logic
      memory_extractor.py  # LLM integration
    commands/
      find_memories.py     # find-memories command
      show_memories.py     # show-memories command
  core/models/
    memory.py           # Memory Pydantic models

tests/
  test_memory_service.py
  test_transcript_finder.py
  test_memory_extractor.py
```

---

## Embedding Readiness

To support future cross-project embedding queries:

1. **Memory table includes `project_dir`** - enables project-scoped queries
2. **Importance score** - for filtering during retrieval
3. **Metadata column** - for storing embedding vectors later
4. **Memory type enum** - for faceted search

Future embedding implementation would:
- Add `embedding` column to `memories` table
- Use `memory_type` + `importance_score` for filtering in vector search
- Enable "find similar memories across projects" queries

---

## Implementation Order

1. Add `MemoryType` enum and `MemoryEntry` model to `models/memory.py`
2. Add memory tables to `database.py` migration
3. Add memory settings to `settings.py`
4. Implement `MemoryService` in `solo/services/memory_service.py`
5. Implement `TranscriptFinder` in `solo/services/transcript_finder.py`
6. Implement `MemoryExtractor` in `solo/services/memory_extractor.py`
7. Add `find-memories` command to `solo/commands/`
8. Add `show-memories` command to `solo/commands/`
9. Add tests
10. Update `__init__.py` exports
