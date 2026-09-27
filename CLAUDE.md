# Working rules

## Writing code: plan → write → check, three agents
Any task that writes or changes code runs as three subagents, cheapest model that can do each job:

1. **Plan — Opus** (`model: opus`). Reads the code and the relevant ADRs, then writes a plan
   detailed enough that a smaller model can implement it without judgment calls: exact files,
   function signatures, the logic step by step, the tests to add, the commands that verify it.
   Plans go in the scratchpad (or `~/.claude/plans/` if they must outlive the session).
2. **Write — Sonnet** (`model: sonnet`). Implements the plan exactly, runs the verification
   commands. If the plan is ambiguous or wrong, it stops and reports instead of improvising.
3. **Check/debug — Sonnet** (`model: sonnet`). Reviews the diff against the plan and the
   constraints below, runs typecheck/Jest/pytest, and debugs failures. Escalate to Opus only
   when Sonnet fails twice on the same bug.

Use Haiku for mechanical work (searches, renames, formatting, log scraping). The main session
coordinates and reports; it doesn't write the code itself. Trivial edits (a typo, a one-line
config change) skip the pipeline.

## Verification
- App changes: `cd app && npm run typecheck && npm test`.
- Pipeline changes (`app/src/pipeline/`, `ml/harness/`): also `cd ml && uv run --locked pytest`;
  Python is the reference.

## Constraints the checker enforces
- Nothing hard-codes a joint: indices, angle definitions and thresholds come from
  `shared/definitions/*.json` (whole-body goal; the ankle is only the first exercise).
- Refuse to measure rather than report a corrupted angle (ADR 0002 C5, ADR 0003 C2).
- Never use the MediaPipe SDK; nothing leaves the phone (ADR 0002 amendment 2026-09-25).
- Worklets: module constants used only in parameter defaults aren't captured on the frame
  processor thread; set defaults in the function body (see `app/src/pose/tracker.ts`).
