# ADR 0008: Repo structure, testing and CI — four directories, JSON Schema contract, no codegen

- **Status:** Accepted
- **Date:** 2026-09-21
- **Deciders:** Project owner
- **Builds on:** ADR 0001 (repo is the portfolio artifact, free macOS runners on public repos),
  ADR 0005 (shared schema, cross-language fixtures), ADR 0006 (conversion and runner
  containers, artifact identity by content hash, macOS requirement), ADR 0007 (versioned
  reports, fixtures in CI, derived data only)

## Context

Seven prior decisions have deposited structural requirements here. The repository must hold:
a React Native app and a Python analysis side; an exercise-definition schema and a
keypoint-series format that **both languages consume from one source**; fixtures that both
implementations of the signal pipeline must agree on; a conversion container and per-model
runner containers; a fetch path for model artifacts identified by content hash; a versioned
report directory; and a CI arrangement in which the interesting tests need macOS and the
interesting data cannot be committed.

Two properties of the project shape the answer. ADR 0006 established that the **MVP trains
nothing**, so there is no training pipeline to house. ADR 0001 established that the
**repository is the portfolio artifact**, since distribution is constrained by the absence of
a paid Apple Developer Program — so legibility to a reader is a functional requirement, not a
courtesy.

## Options considered

### 8a — Layout

The brief proposed `app/ ml/ eval/ docs/`. With no training in scope, `ml/` would contain a
conversion script and little else, while `eval/` would contain the runners consuming its
output — one workflow split across two directories. The alternative extreme, a `packages/`
monorepo with workspace tooling and task orchestration, supplies a dependency graph for two
languages whose only interaction is exchanging JSON files.

### 8b — Cross-language schema sharing

**JSON Schema as the contract, no codegen.** Definitions and fixtures are plain JSON data;
each language validates against the schema in its own test suite; the TypeScript interface is
hand-written.

**Codegen** (quicktype, datamodel-code-generator) guarantees the types match the schema, at
the cost of a generation step both sides must run and generated files appearing in diffs.

**Protobuf or FlatBuffers** provides the strongest contract and an efficient encoding, and is
a full toolchain for a handful of record types exchanged as files.

### 8c — Keypoint-series format

The requirement is asymmetric: **TypeScript only writes this format, Python only reads it.**
That asymmetry, rather than general elegance, should decide it.

`.npy` has an ASCII header, so a TypeScript writer is roughly thirty lines, and the Python
side reads it with `np.load` — no parsing code, with dtype and shape carried in the file.
CSV is inspectable by eye, which has real debugging value, at roughly four times the size and
a parser on both sides. JSON avoids binary handling entirely and is slow and large at 3,600
frames per session. Parquet and NPZ both fail the "TypeScript can write it without a
dependency" test.

### 8d — CI scope

Options ranged from running everything on every push, through a tiered arrangement, to adding
an iOS build job. The iOS build question turns on ADR 0001: without a paid Developer Program,
a CI-produced build cannot be distributed to anyone.

## Decision

```
app/        React Native (Expo dev build)
ml/         convert/ · runners/ · harness/ · reports/   (Python)
shared/     schemas/ · definitions/ · fixtures/         (language-neutral JSON)
docs/adr/
```

**JSON Schema as the contract with no codegen; `.npy` plus sidecar JSON for keypoint series;
Ubuntu tests on every push, macOS artifact tests on pull requests and tags, no iOS build job.**

### Condition 1 — no monorepo tooling

No workspaces, no Nx, no Turborepo. Two languages exchanging JSON files do not need a task
orchestrator, and adding one imports a dependency-graph abstraction with a single edge.

### Condition 2 — the schema is the contract, and both sides test against it

`shared/schemas/` holds the JSON Schema for exercise definitions, the session manifest
(ADR 0007, Condition 8) and the keypoint-series sidecar. Both test suites validate the same
files against the same schema. The TypeScript interface is written by hand and kept honest by
that validation rather than by generation.

### Condition 3 — the series format needs a round-trip test

Since TypeScript writes `.npy` and Python reads it, a hand-rolled writer is exactly the kind
of code that fails silently on an edge case — Fortran ordering, endianness, a dtype mismatch.
CI writes a fixture array from TypeScript, reads it with `np.load`, and asserts equality.
This is one test and it is not optional.

### Condition 4 — CI tiers, and the public-repository assumption

Ubuntu on every push: pytest, Jest, schema validation, the round-trip test. macOS on pull
requests to main and on tags: tests that execute the Core ML artifact, fetched from Releases.
No nightly job, consistent with ADR 0007, Condition 7.

**This design assumes the repository is public**, which is what makes macOS runners free.
ADR 0001 relied on the same assumption. Making the repository private changes the cost
structure of CI, and that dependency is recorded here rather than discovered later.

### Condition 5 — no iOS build job

Without a paid Developer Program a CI build cannot be distributed, so the job would burn
macOS minutes producing an artifact nobody can install. CI typechecks; builds happen on the
owner's machine.

### Condition 6 — model artifacts live in Releases, verified by hash

Never in git. The fetch script verifies the content hash that forms part of the model
identifier under ADR 0006, Condition 3, so provenance recorded in a session refers to a
verifiable artifact.

### Condition 7 — one model card carries the loose ends

`ml/MODEL_CARD.md` holds checkpoint provenance and license (ADR 0002, Condition 3), the
conversion manifest (ADR 0006, Condition 3), measured accuracy (ADR 0007) and known
limitations. This turns the licensing verification condition from a note into a deliverable
with a location, and gives a reader one page answering what the model is and how well it
works.

### Condition 8 — no end-to-end or device-automation testing

No Detox, no simulator automation. A solo project with one device does not earn the
maintenance, and the failures it would catch are ones the owner encounters immediately in
manual use.

## Consequences

### Positive

- Four directories and no build orchestration, so the structure is readable on first
  encounter — which matters given the repository is the portfolio artifact.
- One schema, validated identically on both sides, with no generated code in diffs.
- The analysis side reads session data with no parsing code at all, which is where the
  project's actual work happens.
- Feedback on ordinary commits is fast and runs on free Ubuntu runners; expensive macOS jobs
  run only where they change a decision.
- Licensing verification, conversion provenance and accuracy results acquire a single home,
  so ADR 0002's Condition 3 has somewhere to be satisfied.

### Negative — accepted

- **The hand-written TypeScript interface can drift from the schema** between validation
  runs. Schema validation in CI bounds the drift but does not prevent it in the editor, which
  codegen would.
- **`.npy` is opaque to eye inspection.** Debugging a suspicious series needs a small dump
  script, to be written when first needed rather than in advance.
- **Native build breakage will be found late**, on the owner's machine rather than in CI.
  This cost rises once vision-camera frame processor plugins are in play, and is the condition
  most likely to warrant revisiting.
- The macOS job running only on pull requests means artifact regressions can sit undetected on
  a working branch.
- Public-repository status is now a structural assumption rather than a preference.
- Folding `eval/` into `ml/` makes the research deliverable less discoverable, so the README
  must link the reports directory explicitly rather than relying on directory names.

### Follow-ups this decision creates

- The README is now load-bearing and needs: a link to the current accuracy report, the
  limitations and privacy section from ADR 0007 (n=1 corpus, derived-data-only, no clinical
  validation), the not-medical-advice framing, and a note that the repository is public by
  design for CI reasons.
- `shared/schemas/` must be written before either side has code that reads definitions, since
  retrofitting a contract across two languages costs more than declaring it.
- The `.npy` round-trip test and the cross-language pipeline fixtures are the first tests to
  exist, ahead of the code they constrain.
