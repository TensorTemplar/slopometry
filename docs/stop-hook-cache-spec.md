# Stop-Hook Feedback Cache — Behavioral Spec

> Status: descriptive spec of the **current** implementation (`handle_stop_event` in
> `src/slopometry/core/hook_handler.py`) plus a review of whether it is principled.
> Goal we are working toward: *the Stop hook should surface feedback **only when source
> code (`.py`/`.rs`) content actually changed** — never on commits, branch switches,
> doc/config edits, or other "random" file churn.*

---

## 1. Purpose

On every Claude Code `Stop` event the hook may emit blocking feedback (code smells,
context-coverage gaps, dev guidelines). To avoid nagging the agent with the *same*
feedback repeatedly, a per-project cache (`.slopometry/feedback_cache.json`) records the
state for which feedback was last shown. The hook is supposed to stay **silent** until
the code state changes again.

The cache is keyed on a "working tree cache key". The central claim in the code
(`_compute_working_tree_cache_key` docstring) is:

> "The cache key depends only on the git commit and source file contents … This ensures
> the hook fires exactly once per code state change."

This spec documents what the code *actually* does and tests that claim.

---

## 2. Persisted state

`FeedbackCacheState` (`core/models/hook.py`), serialized to
`<project>/.slopometry/feedback_cache.json`:

| field         | meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `last_key`    | the working-tree cache key at the last fire (`blake2b(commit_sha:wt_hash)`) |
| `file_hashes` | `{rel_path: blake2b}` for each **modified** source file at last fire    |
| `commit_sha`  | `HEAD` SHA at last fire, used for the cheap fast-path check              |

`file_hashes` exists only to compute *which* files changed for smell scoping
(`get_files_changed_since`). It is **not** part of the firing decision.

---

## 3. What counts as a "source file"

`language_config.py` defines source = extensions `.py`, `.rs` only. Git pathspecs
`*.py`, `*.rs`. Everything else (`.md`, `.toml`, `.json`, lockfiles, etc.) is **not**
source. Modified-source detection (`_get_modified_source_files_from_git`) additionally:

- looks only at `git diff` / `git diff --cached` (tracked changes; **untracked new files
  are invisible** to this path),
- drops paths inside ignored dirs (`__pycache__`, `.venv`, `site-packages`, `build`, …)
  via `should_ignore_path`,
- drops paths inside declared submodules (`get_submodule_prefixes` + `--ignore-submodules=all`).

---

## 4. Firing decision (current control flow)

`handle_stop_event(session_id, parsed_input)`:

1. **Subagent guard** — `if parsed_input.stop_hook_active: return 0`.
2. **Resolve working dir** — `_resolve_working_directory(db.get_session_working_directory(...))`;
   bail (`return 0`) if unknown/unresolvable.
3. **Cheap fast-path (cache HIT → silent):** load `cached_state`. If it exists and has a
   `commit_sha`:
   - `current_sha = git rev-parse HEAD` (1 cmd)
   - if `current_sha == cached.commit_sha` **and** `not _has_source_modifications(wd)`
     → `return 0` (silent, no cache write).
   `_has_source_modifications` is two-tier: `git diff --quiet *.py *.rs` (staged +
   unstaged); only if that reports dirty does it enumerate + filter ignored/submodule
   paths to confirm a *real* source modification.
4. **Analyzable-files gate** — `if not _has_analyzable_source_files(wd): return 0`.
5. **Full key (cache compare):** `cache_key = _compute_working_tree_cache_key(wd)`:
   - `commit_sha = git_state.commit_sha or "unknown"`
   - `wt_hash = calculate_working_tree_hash(commit_sha)` if any source modifications, else `"clean"`
   - `cache_key = blake2b(f"{commit_sha}:{wt_hash}")`
   If `cached_state.last_key == cache_key` → `return 0` (silent).
6. **Build feedback** — load stats; compute `edited_files`
   (`get_files_changed_since(cached.file_hashes)`, or all modified files on first run);
   assemble `feedback_parts` from:
   - code smells scoped to edited/related files,
   - context-derived smells (`unread_related_tests`),
   - context-coverage gaps (if `enable_complexity_feedback`),
   - dev guidelines from `CLAUDE.md` (if `feedback_dev_guidelines`).
7. **Save cache** — `_save_feedback_cache(wd, cache_key, current_file_hashes, commit_sha)`
   (always, whether or not feedback is shown).
8. **Emit** — if `feedback_parts`: print `{"decision":"block","reason":…}`, `return 2`;
   else `return 0`.

### Firing invariant (as implemented)

> Feedback can fire **only** when `cache_key` differs from `last_key`.
> `cache_key = blake2b(commit_sha : working_tree_hash_of_modified_source_files)`.

So the firing trigger is: **`commit_sha` changed, OR the set/content of modified source
files changed.**

---

## 5. Review — is this principled?

### ✅ What it gets right
- Non-source churn (`.md`, `.toml`, JSON, lockfiles) never enters the key → cannot fire.
- Ignored dirs and submodule contents are filtered at every layer (key + fast-path).
- mtime-only touches are filtered out by content hashing (tier 2).
- The cheap fast-path avoids the 8–9-command key computation on the common idle Stop.

### ❌ Where it is *not* principled

**P1 — `commit_sha` is in the key, so commits fire the hook even when source content is unchanged.**
This is the prime suspect for "fires on random file changes." The key conflates *where*
the code lives (working tree vs history) with *what* the code is. Concretely:
- Agent edits `foo.py` → key `(sha_old, dirty_hash)`, fires once. ✅
- Agent commits `foo.py` → working tree clean, `HEAD` advances → key `(sha_new, "clean")`
  → **fires again** for content that is byte-for-byte identical. ❌
- Agent commits a **docs-only** / config-only change, or `git pull`/merge/rebase/branch
  switch lands → `commit_sha` changes, no `.py`/`.rs` content change → **fires.** ❌
The invariant should be "fires when source *content* changes," but `commit_sha` makes it
"fires when source content changes **or HEAD moves for any reason**."

**P2 — the key is built from the *diff against HEAD* (modified files), not from absolute source content.**
This is the root cause of P1. Because the hash is "the set of files that differ from
HEAD + their content," it is inherently relative to `HEAD`; moving `HEAD` (a commit)
necessarily changes the key even though the bytes on disk didn't. A principled key would
be a function of the *current content of the source tree* (committed + uncommitted),
invariant to whether a change has been committed.

**P3 — feedback content is decoupled from "what changed," so any key change re-shows unrelated feedback.**
`feedback_dev_guidelines` appends the `CLAUDE.md` guidelines whenever the section exists —
independent of the diff. `context_coverage` gaps likewise reflect the whole session, not
this change. So once the key flips for *any* reason (e.g. a P1 commit), the agent is
re-shown dev guidelines / coverage warnings that have nothing to do with the new state.
Even if smells are correctly scoped to `edited_files`, these two parts can carry a fire on
their own.

**P4 — untracked new source files are invisible to the modified-file path.**
`_get_modified_source_files_from_git` uses `git diff` only. A brand-new untracked `.py`
file contributes nothing to `working_tree_hash` (stays `"clean"`) until staged/committed.
So creating a new source file does **not** fire (until it's added), which contradicts
"fire when code changed." (Note `_has_analyzable_source_files` *does* see untracked files,
so the gate and the key disagree about what exists.)

**P5 — cache not written on the early-return branches, leaving `last_key` stale.**
If `cache_key` changed but `stats` is empty (`return 0` at step 6 pre-save) the cache is
not updated. Minor (no spurious fire), but means the "point of comparison" can lag.

### Severity ranking
P1/P2 are the same defect viewed two ways and are almost certainly the reported bug.
P3 amplifies it (turns a key flip into actual visible output). P4/P5 are correctness gaps
worth noting but secondary.

---

## 6. Implemented behavior — commit-invariant content key

The cache key is now a pure function of **current source content** (implemented in
`WorkingTreeStateCalculator.calculate_source_content_key`, delegated to by
`_compute_working_tree_cache_key`):

```
files     = git ls-files --cached --others --exclude-standard   # tracked + untracked, honors .gitignore
files     = [f for f in files if is_source_file(f)              # .py / .rs only
                                and not should_ignore_path(f)    # drop build/, dist/, .venv/, *.egg-info, …
                                and not path_in_submodule(f)]    # drop submodule contents
blob_shas = git hash-object --stdin-paths <files>               # C-level content hashing
key       = blake2b( sorted "rel_path:blob_sha" )
```

If `git hash-object` fails, a pure-Python fallback (`sha1("blob <len>\0" + content)`)
computes the **identical** git blob SHA, so the key is independent of which path ran.

Why `ls-files` + `hash-object` rather than the originally-sketched `git write-tree`: a
bare `write-tree` would exclude ignored dirs by `.gitignore` *only*, dropping the
`should_ignore_path` filter and breaking the build-artifact tests (a `build/`/`dist/`
`.py` that isn't gitignored). `ls-files` lets us reuse the existing extension /
`should_ignore_path` / submodule filtering verbatim, and never descends into submodules.

Properties (maps to the review's defects):
- **P1/P2 fixed — commit-invariant.** No `commit_sha` in the key. Committing identical
  bytes, branch switches, pulls, rebases, merges → same key → no re-fire. Verified by
  `TestCommitInvariance`.
- **P4 fixed / Q3 — fires on new untracked source.** `ls-files --others` includes new
  `.py`/`.rs`. Verified by `TestNewUntrackedFiles`.
- **Submodules** — `ls-files` lists a submodule only as its gitlink path (not source), and
  never recurses; `path_in_submodule` double-guards. Submodule edits / HEAD moves /
  `submodule.recurse` config never change the key. Verified by `TestSubmoduleHandling`.
- **Q2 (P3)** — dev-guidelines and context-coverage feedback **ride along** on any genuine
  source-content change; no extra per-part gating. Correctness comes from the key no
  longer flipping on commits / doc churn.

### Fast-path (optimization, unchanged in spirit)
`handle_stop_event` keeps a cheap silent-path guard: when the cached `commit_sha` equals
the current `HEAD` **and** `_has_source_changes` is False, it returns 0 without computing
the content key. `_has_source_changes` now reports True for **either** a tracked
modification **or** a new untracked source file, so the fast-path can never swallow a fire
for newly created source. `commit_sha` is only a hint here — a bare commit fails the SHA
match, takes the full-key path, and correctly matches `last_key` (silent) instead of
re-firing.

### Residual notes
- **N3:** the firing key and the per-file `file_hashes` are now cleanly separate concerns.
  `file_hashes` (modified-source-only, via `get_source_file_content_hashes`) still feeds
  `get_files_changed_since` for smell scoping; it is not part of the firing decision.
- **Cost:** on a fast-path miss with a dirty/untracked source tree, the content key hashes
  all in-scope source files via one `git hash-object` call (C-level). This is cheaper than
  the previous full path (which spawned ~9 git commands) and runs once per genuine Stop
  miss, not per idle Stop.
