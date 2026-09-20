# ADR 0004: Form assessment — cited rules, datasets as a projection lab, own-trend presentation

- **Status:** Accepted
- **Date:** 2026-09-20
- **Deciders:** Project owner
- **Builds on:** ADR 0002 (definitions as data, refuse-to-measure), ADR 0003 (filtered angle
  signal, hysteresis phase output, per-session calibration)

## Context

Form assessment conflates two things that have different requirements and different
evidence bases:

1. **Rep quality scoring** — "was that a good rep?", expressible as a number.
2. **Actionable corrective feedback** — "your knee is bending during the raise", which
   names a fault and how to fix it.

Users need (2). The available datasets support (1). That mismatch drives this decision.

The project's framing imposes two hard constraints from the brief. All guidance must be
**backed by cited clinical evidence**, and the app must not diagnose or prescribe. Both bear
directly on what form assessment is allowed to be.

### The datasets, surveyed

- **UI-PRMD** — 10 rehabilitation movements (deep squat, hurdle step, inline lunge, side
  lunge, sit-to-stand, standing active straight leg raise, shoulder abduction, shoulder
  extension, shoulder internal/external rotation, standing lunge), 10 healthy subjects,
  Kinect and Vicon. Incorrect performances are **simulated by healthy subjects**, which is
  not the distribution of real patient compensation patterns.
- **KIMORE** — 5 exercises (arm lifting, lateral trunk tilt, trunk rotation, pelvis
  rotation, squatting), ~78 subjects including genuine patient groups (Parkinson's, stroke,
  low back pain), with continuous quality scores assigned by physiotherapists. Kinect v2.

**Neither contains an ankle exercise.** For the MVP scope they supply no training data at
all, irrespective of how well the 3D-to-2D domain gap is handled. The widened product goal
(ADR 0002) makes their squat, lunge and shoulder content relevant to later joints, but not
to what ships.

### The domain gap, concretely

Kinect's ~25-joint skeleton and Vicon marker sets do not correspond cleanly to
COCO-WholeBody. Vicon markers sit on the skin surface; COCO keypoints follow annotator
convention — the COCO "ankle" is roughly where a human annotator sees the lateral
malleolus, which is not the same point as a marker placed on it. Systematic offsets of a
couple of centimetres should be expected. Neither dataset has usable foot detail.

## Options considered

### Form assessment

**Option A — rule-based thresholds, defined as data.** Per-exercise rules over the filtered
angle signal and derived geometry: peak angle as a fraction of calibrated range, left/right
symmetry, knee held near-extended through a calf raise, tempo bounds, sway magnitude.

The decisive property is not accuracy but **citability**. The brief requires guidance backed
by cited evidence; a network emitting a scalar cannot cite anything, structurally. If a rule
is a record — `{ id, signal, comparator, threshold, citation, message }` — the citation
requirement is satisfied by the data model rather than by discipline. Rules are also
explainable by construction, independently testable, need no training data, and degrade
gracefully when one input signal is unavailable.

Costs: hand-tuned thresholds, brittleness across body types, labour per exercise, and
blindness to movement that is visibly wrong without violating any single rule.

**Option B — supervised quality score trained on UI-PRMD / KIMORE.** Real machine-learning
content and genuine portfolio value. Unusable for the MVP: no ankle data. Beyond that,
shipping it would place an unexplainable health judgment — trained on roughly 78 subjects,
a different sensor, and a different exercise set — in front of a rehabilitation patient,
with no way to tell them what to change.

**Option C — per-user deviation baseline.** Model the user's own early reps and flag drift
from their personal norm. Needs no external dataset, sidesteps cross-subject generalization
entirely, and per-user normalization suits rehabilitation. Two weaknesses: it cannot name
the fault, and it assumes the early reps were good.

### Use of UI-PRMD and KIMORE

**Option D-1 — as training data.** Blocked for the MVP by the absence of ankle exercises.

**Option D-2 — as a projection lab.** Project the 3D skeletons to 2D at known, deliberately
varied camera poses. This produces exact ground truth for questions the MVP actually needs:
how many degrees of error a given camera misalignment introduces into a measured joint
angle, and whether the out-of-plane detector from ADR 0002 fires when it should. It treats
the 3D-to-2D gap as an instrument rather than an obstacle, requires no video capture or
annotation, and is available immediately.

**Option D-3 — do not use them in the MVP.** Defensible on focus grounds, but discards a
cheap source of ground truth for a feature (camera setup guidance) that is in scope.

### Presentation of normative comparisons

**Option E-1 — the user's own trend only.** Measured angles over time against their own
history. No population comparison, therefore no implicit classification of the user.

**Option E-2 — published norms shown alongside, explicitly framed as literature.** More
informative, and closer to the line separating information from assessment.

**Option E-3 — norms plus contralateral (injured versus uninjured side) comparison.**
Clinically meaningful and standard in practice, and the largest step toward implying a
diagnosis.

## Decision

**Option A** for user-facing feedback, **Option D-2** for the datasets, **Option E-1** for
presentation. Option B is deferred to a phase-2 research track that is evaluated in the
harness and never shipped in the MVP. Option C is noted as the natural successor to A once
the app has collected real session data.

### Condition 1 — no citation, no rule

The rule record requires a citation field, and a rule without one does not ship. This is
enforced by the schema, not by review. Where the evidence turns out not to exist for a rule
that seems intuitively obvious, the correct outcome is that the rule is dropped — not that
the citation requirement is relaxed.

### Condition 2 — rules never evaluate an unmeasurable rep

A rep flagged unmeasurable by the out-of-plane check (ADR 0002, Condition 5) or by
confidence collapse (ADR 0003, Condition 2) receives no form feedback. Delivering corrective
advice computed from an angle the system already declined to report would be the worst
available failure: confident, specific, and wrong. The user sees a setup prompt instead.

### Condition 3 — rules read the measurement path only

Rules consume the One Euro-filtered angle and the state machine's phase output, never the
display-smoothed coordinates (ADR 0003, Condition 5).

### Condition 4 — projection-lab ordering

Projected skeletons are unrealistically clean, so synthetic 2D must have noise injected that
matches RTMPose's real error distribution. That distribution comes from the goniometer
validation, which therefore has to happen first. Sequence: goniometer validation → measured
noise model → synthetic projection experiments. Running the projection lab before the noise
model produces optimistic numbers that will not survive contact with real video.

### Condition 5 — the framing hazard inherited from ADR 0003 applies here

Rules expressed as fractions of the user's calibrated range describe performance relative to
that user's current range, not achievement of a normal range. Rule messages must not imply
otherwise.

### Condition 6 — the learned track is harness-only

Work on Option B, when it begins, produces evaluation results and nothing shipped. The
question it answers — whether quality models trained on 3D Kinect data transfer to
single-camera 2D at all — is a legitimate research contribution, and answering it honestly
in the negative is a valid outcome.

## Consequences

### Positive

- Every piece of advice the app gives can be traced to a source, which is the project's
  stated differentiator rather than a compliance formality.
- Feedback names the fault and the correction, which is what makes it useful. A quality
  score would not.
- No training data is required to ship, so form assessment is not blocked behind data
  collection.
- Rules are per-exercise data, consistent with ADR 0002's no-hard-coded-ankle constraint,
  so the machinery transfers to other joints unchanged.
- The citation field is the seed of the phase-2 evidence library described in the brief;
  that feature starts with a populated corpus rather than an empty one.
- The projection lab produces camera setup guidance analytically, converting the domain gap
  from a liability into a source of ground truth.

### Negative — accepted

- Rules cannot detect movement that is wrong without violating a specific threshold. There
  is no holistic quality signal in the MVP.
- Thresholds are hand-tuned and will be brittle across body types until validated on more
  than one body.
- **The real cost of the all-round goal surfaces here, and it is not engineering.** Adding a
  joint means finding and reading the clinical literature for that joint's exercises. The
  bottleneck for generality is evidence curation, not code — which is worth knowing before
  the second joint is promised.
- The citation requirement will shrink the feedback set. Some intuitively obvious rules will
  turn out to have no supporting evidence and will have to be dropped. This is the
  requirement working as intended, and it will feel like a loss.
- **Own-trend-only presentation means a user with a genuinely abnormal range receives no
  signal that they should seek assessment.** The mitigation is the app's standing
  encouragement to obtain professional assessment, present regardless of measurements — not
  a threshold-triggered alert, which would be diagnosis by another name. This is a
  deliberate trade of usefulness for staying inside the stated constraint.
- Projection-lab findings carry the joint-definition mismatch described in Context. They
  support sensitivity analysis ("how much does camera angle matter?") and not absolute
  accuracy claims ("the measurement is within N degrees").

### Follow-ups this decision creates

- **Decision 5 (storage)** must version the rule set alongside sessions. Changing a
  threshold otherwise makes past feedback irreproducible, and a progress chart whose
  underlying criteria silently changed is misleading.
- **Decision 7 (evaluation harness)** needs clips containing deliberately incorrect form,
  labelled with the fault performed, in order to test rules at all. This is a data
  collection task with a protocol to design, and it is more work than recording good reps.
- The rule record schema is shared between the app and the harness and should be defined
  once, in a location both consume.
- Sourcing citations for the two or three MVP ankle exercises is a research task that can
  begin immediately and in parallel with the Condition 1 inference spike from ADR 0002.
