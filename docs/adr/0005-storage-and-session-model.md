# ADR 0005: Local storage — SQLite plus signal blobs, keypoint-subset retention, opt-in video

- **Status:** Accepted
- **Date:** 2026-09-21
- **Deciders:** Project owner
- **Builds on:** ADR 0002 (definitions as data, calibration record), ADR 0003 (filter
  parameters will be retuned), ADR 0004 (rule set must be versioned)

## Context

The obvious framing of this decision is capacity planning. That framing is wrong. The
binding constraint is **comparability of measurements over time**.

Two earlier decisions guarantee the processing pipeline will change. ADR 0003 records that
One Euro's parameters are hand-tuned with no optimality criterion, so they will be retuned
against the harness. ADR 0004 requires the rule set to be versioned because thresholds will
move. Both have the same implication for a progress chart: a measurement taken in March
under different filters and different rules than one taken in June cannot honestly be
plotted on the same axis. For an application whose entire purpose is showing whether range
of motion is improving, a chart that manufactures improvement out of a software change is
the most damaging defect available.

There are two escapes. Freeze the processing permanently, which is not credible for a
project that has not yet measured anything. Or **retain enough per-session data to re-score
the entire history whenever the pipeline changes**. This decision takes the second, which
converts the retention question from "how much history would be nice" into "what is the
minimum that keeps the record honest".

Additional forces: no accounts and no cloud sync (MVP non-goals); video never leaves the
device (hard constraint); the Python harness is decoupled from the app (ADR 0001,
Condition 2) but benefits enormously from real session data; and the schema must not
hard-code the ankle (ADR 0002 follow-up).

## Options considered

### 5a — Storage engine

**AsyncStorage** — string key-value. Appropriate for preferences, unsuited to time series or
to any query. Rejected.

**expo-sqlite** — already present in the Expo SDK, so no dependency is added. Relational and
indexed, which makes progress aggregation ("mean peak angle per week for this exercise") a
single query rather than a full scan in JavaScript.

**Realm / WatermelonDB** — both materially heavier. WatermelonDB is SQLite underneath and is
optimized for synchronization, an explicit non-goal. Realm is a large dependency oriented
toward a hosted backend. Rejected as paying for capabilities the project has excluded.

**Plain JSON files** (`expo-file-system`) — genuinely sufficient at MVP volume: a few hundred
files loaded and reduced in JavaScript. No schema, no migrations, no SQL. The cost appears
as history grows and as filtering by exercise, date and joint multiplies.

**Hybrid — SQLite for metadata and per-rep rows, files for per-frame signal.** Per-frame data
is only ever read whole, during re-scoring; it is never queried row-wise. Storing 3,600 rows
per session in a relational table buys nothing and costs insert time and overhead.

### 5b — Retention fidelity

Approximate figures for a two-minute session at 30fps, float32:

| Tier | Contents | Size | Permits |
|---|---|---|---|
| 0 | session summary | ~200 B | nothing |
| 1 | + per-rep records | ~4 KB | rep-level re-analysis |
| 2 | + per-frame angle series | ~30–100 KB | re-run filters and rep detection |
| 3 | + per-frame keypoints, exercise's declared subset | ~0.5 MB | re-derive angles, change joints, change anything below the model |
| 3-all | + all 133 keypoints | ~5.7 MB | the above, for joints the exercise never used |
| 4 | + video | hundreds of MB | re-run the pose model itself |

### 5c — Video retention

Local storage does not breach the "video never leaves the device" constraint, but it creates
a sensitive asset on the phone. Against that, real footage of real ankles is the highest-value
input to the evaluation harness, and the synthetic projections of ADR 0004 cannot substitute
for it. Options ranged from never storing, through per-session opt-in, to storing always with
automatic expiry.

## Decision

**Hybrid SQLite plus signal blobs; Tier 3 with the exercise's declared keypoint subset;
video opt-in per session and off by default.**

Tier 3-all was rejected despite being the future-proof choice. Shoulder keypoints from an
ankle session will never be analysed; a future shoulder exercise declares its own subset and
accumulates its own history from that point. The thirteenfold cost buys nothing real.

No ORM. Four tables of plain SQL with `PRAGMA user_version` and an ordered list of migration
statements is the entire requirement.

### Condition 1 — sessions store a definition snapshot, not a version reference

The exercise definition, its filter parameters, its thresholds, its rule set and the session
calibration are **denormalized onto the session** as JSON.

A version number is only resolvable while the build containing that definition exists.
Definitions ship inside the app bundle; updating the app makes old references dangle, and the
failure is silent — the session still displays, with the wrong parameters assumed. A few
kilobytes per session makes every session self-describing permanently.

### Condition 2 — provenance is mandatory, and a session without it is not plottable

Every session records the model variant and quantization in use, the app version, the
processing parameters in force, and the calibration record. Without these, a change in the
chart cannot be attributed to the ankle rather than to the software. Provenance is captured
as a hash over the processing configuration, and the chart layer treats a session lacking it
as unplottable rather than plotting it with assumptions.

### Condition 3 — charts must not silently mix provenance

When the pipeline changes, history is re-scored. Until it is, sessions scored under different
configurations are visibly demarcated on the chart rather than blended. Re-scoring is a normal
operation with a UI affordance, not a migration performed once and forgotten.

### Condition 4 — one fixture set, two implementations

The decoupling in ADR 0001 means the signal pipeline exists twice: TypeScript on-device and
Python in the harness. Re-scoring happens in both. Two implementations of One Euro and of the
rep state machine will drift, and the drift will appear as an accuracy discrepancy attributed
to the wrong cause.

Mitigation: a single canonical fixture set — recorded input signals with expected filtered
output and expected rep boundaries — committed once and asserted against by both
implementations. Disagreement is a build failure, not a research finding.

### Condition 5 — video handling when enabled

App-private storage, excluded from iCloud backup, never written to the photo library. Opt-in
is explicit and per session. Recording is visibly indicated while active, and deletion is
available per session without navigating a settings menu.

### Condition 6 — export is the backup story

A single archive containing session rows, signal blobs and definition snapshots is written to
the Files app via the share sheet. This is both the harness import path and, in the absence of
accounts or sync, the only protection against device loss. It is a user-facing feature, not a
developer convenience.

## Consequences

### Positive

- Any change to filters, thresholds, angle definitions or rule sets can be applied
  retroactively, so the progress chart remains a record of the ankle rather than of the
  codebase.
- Provenance makes the alternative explanation for any observed improvement checkable rather
  than assumed away.
- Sessions are self-describing indefinitely, surviving app updates that remove the definitions
  they were recorded under.
- Real device sessions become harness input through the same export path users have for
  backup, so no separate developer pipeline is needed.
- Storage cost is modest: roughly 0.5 MB per session, or about 75 MB across 150 sessions.
- The schema carries no ankle-specific columns; the keypoint subset and signal names come from
  the exercise definition, consistent with ADR 0002.

### Negative — accepted

- Growth is unbounded. No retention or pruning policy exists, which is correct at this scale
  and will need revisiting well before it becomes urgent.
- **The declared subset is a commitment.** An angle definition later requiring a keypoint that
  was not stored orphans the history for that measurement. Mitigation is to declare the subset
  with margin — the whole limb chain rather than the minimal pair — accepting some waste
  against a failure that cannot be repaired after the fact.
- The signal pipeline is implemented twice, in two languages. Condition 4 bounds the risk but
  does not remove the duplicated maintenance.
- No accounts and no sync means device loss is data loss unless the user has exported. This
  follows from the MVP non-goals and is accepted, with export as the mitigation.
- Opt-in video means most sessions leave no footage, so harness data comes predominantly from
  deliberate capture sessions rather than organic use. This is the intended trade.

### Follow-ups this decision creates

- **Decision 6 (model export)** must define a stable model variant identifier, since
  provenance records it.
- **Decision 7 (evaluation harness)** consumes the Condition 6 archive as its import format,
  and owns the Condition 4 fixture set.
- **Decision 8 (repo structure)** must place the exercise definition schema and the fixture
  set where both the app and the Python side consume them from one source. This is a concrete
  argument for a monorepo with a shared directory rather than separate repositories.
- Re-scoring on device has a runtime cost over a large history that has not been estimated;
  it may need to run incrementally or in the background.
