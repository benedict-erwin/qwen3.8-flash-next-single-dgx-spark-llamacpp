# Pi setup used by the maintainer

Optional. These are the global Pi files I run this model with; take them as a starting
point or ignore them. The provider config itself is `../pi-models.json`.

| File | Goes to | What it does |
|---|---|---|
| `AGENTS.md` | `~/.pi/agent/AGENTS.md` | Global rules: language, a session tracker for continuity across sessions, how to read files without flooding a 131k context, git hygiene, what to ask before doing, how to report |
| `settings.json` | `~/.pi/agent/settings.json` (merge) | Thinking level `medium` for this model, cache-miss notices on, compaction reserve raised to 24k |

Why these values, all from measurements in `../OPTIMIZATION.md`:

- **Context is 131k by default** (262k with `CTX=262144`). A reasoning model spends a lot
  of it on thinking — one one-shot answer here ran to 16k tokens — and every token of the
  answer is resent as history on the next turn. The rules therefore tell the model to read
  files with `grep -n` + `sed -n` and to `tail` logs, and the thinking level defaults to
  `medium`; raise it per session with `/thinking` when the task warrants it.
- **`reserveTokens` 24576** instead of Pi's 16384: room for a long thinking block before
  compaction triggers, so the answer near the limit is not cut off. Compaction then
  happens at ~106k tokens on a 131k window.
- **`showCacheMissNotices`**: on this hybrid model a prompt-cache miss re-prefills the whole
  previous response (see the 2026-09-05 section of `OPTIMIZATION.md`), so it is worth
  seeing when one happens. Pair it with `LLAMA_SERVER_SLOTS_DEBUG=1` on the server.
- **Session tracker** (`.agent/SESSION_TRACKER.md`): compaction keeps context inside one
  session; the tracker carries state between sessions. The model asks before creating it.

The language rule (talk in Bahasa Indonesia) is a personal preference — change the first
bullet. Pi loads `AGENTS.md` from `~/.pi/agent/`, every parent directory of the cwd, and
the cwd, and concatenates them; project-specific rules belong in the project's own file.
