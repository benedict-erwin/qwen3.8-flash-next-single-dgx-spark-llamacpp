# Global rules

## Language
- Talk to the user in Bahasa Indonesia. Keep technical terms in English (prefill, commit, refactor, endpoint); do not translate them.
- Code, comments, commit messages, and README files are in English.

## Session continuity
- At startup, if `.agent/SESSION_TRACKER.md` exists: read it, acknowledge in 1-2 sentences what the current state is, then continue. If it does not exist, ask once whether to create it; never create it silently.
- The tracker is a status board, not a log. It has exactly three sections:
  1. **Current State** — stack/config in use, what is blocked, next steps. Max 30 lines. Overwrite, do not append.
  2. **Recent Activity** — last 5 sessions, 3-5 lines each, newest first. Drop the oldest when adding a sixth.
  3. **Decisions** — numbered decisions that still matter across sessions. Max 30 entries.
- Hard limit 120 lines. Long detail goes to `.agent/sessions/YYYY-MM-DD.md`, never into the tracker.
- Update the tracker when a meaningful unit of work is done and before ending a session. Convert relative dates ("yesterday") to absolute dates.
- `.agent/` is local only. Add it to `.git/info/exclude`, not to `.gitignore`, so the published repo carries no trace of it.

## Working method
- Read a file before editing it. Make targeted edits; do not rewrite a file from scratch unless asked.
- Context is limited. Read only the part of a file you need: `grep -n` to locate, then `sed -n 'A,Bp'`. Never `cat` a whole file over 200 lines.
- Keep tool output short: pipe test, build and log output through `tail -n 40` or `grep`; never print a full log.
- Do not re-read a file you just edited to verify; the edit result already tells you it applied.
- Before claiming something works, run it and show the actual output. If you did not run it, say so.
- Fix the code when tests fail. Do not change a test to make it pass unless the test is provably wrong, and say why.
- Do not add features, options, or "nice bonuses" that were not asked for. If you think one is needed, propose it in one sentence first.
- When a task is ambiguous in a way that changes the work, ask one precise question. Otherwise decide and state the assumption.
- Temporary files go in `tmp/` (or the system temp dir), never in the repo root.

## Claims
- Label claims you cannot back with a source: `[Inferensi]` (logically derived), `[Spekulasi]` (possible), `[Belum Terverifikasi]` (no source), `[estimate]` (cost or time figure).
- Never type a measured number from memory. Cite where it comes from (file path, log, command output).

## Git
- Conventional Commits, short, in English: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Stage files by name. Never `git add -A` or `git add .`.
- Never force-push, amend, or rebase a published commit. Ask before merge, reset, or deleting a branch.
- Never commit secrets, large data, `tmp/`, or `.agent/`.
- Commit at the end of a unit of work, not after every edit.

## Ask before
- Any process that will run longer than 30 minutes.
- Any download larger than 10 GB.
- Anything that touches services outside this repo (other containers, system services, other projects).
- Any destructive operation outside the repo.

## Reporting
- Lead with the result. Then what changed, then what is left. No preamble, no restating the question.
- Report failures plainly with the error output. Do not soften or hide them.
- When done, list what was verified and how, and what was not verified.
