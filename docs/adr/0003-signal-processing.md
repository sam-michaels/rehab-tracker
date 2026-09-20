# ADR 0003: Signal processing — angle-domain filtering, One Euro live, hysteresis rep detection

- **Status:** Accepted
- **Date:** 2026-09-20
- **Deciders:** Project owner
- **Builds on:** ADR 0002 (RTMPose-WholeBody, per-keypoint SimCC confidence, per-session
  calibration record, exercise definitions as data)

## Context

RTMPose-WholeBody emits 133 `(x, y, confidence)` triples per frame. Those coordinates are
noisy in three distinct ways, and conflating them produces a filter that handles none of
them well:

1. **High-frequency jitter** — sub-pixel wobble present even on a stationary limb. Worst on
   the foot keypoints, which is where the ankle measurement lives.
2. **Occasional gross outliers** — a keypoint flipping to the wrong foot, or snapping to a
   background edge, for one or two frames. Averaging smears these across neighbouring
   frames; they need to be rejected, not attenuated.
3. **Confidence collapse** — occlusion or motion blur, reported honestly by the SimCC peak
   value. This is information, not noise, and the pipeline should act on it.

Two consumers of this data exist with different constraints. The **app** must be causal:
it cannot see future frames, and feedback latency is felt by the user. The **Python
harness** (ADR 0001, Condition 2) operates on recorded files with no causality constraint
and no latency budget, so it can use non-causal methods that the app cannot.

Two further inherited constraints shape this decision. Exercise definitions are data rather
than code (ADR 0002 follow-up), so anything exercise-specific here must be a parameter.
And a per-session calibration record already exists (ADR 0002, Condition 5), so anything
needing per-user normalization has somewhere to live.

## Options considered

### 3a — What gets filtered

The question that determines the other two, and the one most easily skipped.

**Option A-1 — filter coordinates, derive the angle afterwards.** The conventional
arrangement: one filter stage, one pipeline, angles computed from smoothed points.
Smoothing each keypoint independently, however, ignores the fact that a shank is a rigid
segment. Two endpoints can each be smoothed well while the angle between them is degraded,
and the tuning surface is 266 filters controlling one reported number.

**Option A-2 — filter the derived angle; smooth coordinates separately for display only.**
The angle is a single 1-D signal and is the quantity actually reported, so its noise and
lag can be traded off directly against the metric that matters. The skeleton overlay still
needs smoothing or it looks broken, but overlay lag is invisible to the user and harmless
to the measurement, so that filter can be aggressive and is tuned independently.

**Option A-3 — build both, decide empirically from the harness.** Defensible, and the
harness could answer it. Rejected as sequencing: the app needs a pipeline shape before the
harness has ground truth to adjudicate with.

### 3b — Smoothing filter

**Option B-1 — One Euro filter** (Casiez et al., 2012). An adaptive low-pass whose cutoff
frequency rises with the signal's velocity: heavy smoothing when the limb is still,
light smoothing when it moves fast. Two intuitive parameters (`min_cutoff` sets
standing-still jitter, `beta` sets how quickly it yields to motion) plus a derivative
cutoff. Roughly twenty lines of code, no dynamics model required. It was designed for
precisely this class of signal, and is what MediaPipe uses internally.

**Option B-2 — Kalman filter, constant-velocity model.** Optimal under Gaussian noise and
linear dynamics, and uniquely able to fold SimCC confidence in principled fashion by
modulating the measurement noise `R` per frame. Supplies a velocity estimate for free.
Two objections: `Q`/`R` tuning is hand-tuning with less intuitive knobs than One Euro's,
and a constant-velocity model overshoots at direction reversals — which is exactly where
peak plantarflexion, the measurement of interest, occurs.

**Option B-3 — Savitzky–Golay, and median.** Savitzky–Golay fits a local polynomial over a
centred window: zero lag by construction, preserves peak amplitude far better than a moving
average (which systematically clips extrema, and the extrema are the ROM measurement), and
yields clean derivatives. Non-causal, so unavailable live. A median filter of width 3–5
rejects the gross outliers of noise class 2 outright, and is cheap enough to run in both
environments.

A plain moving average was considered and rejected: it attenuates the peaks that constitute
the measurement.

### 3c — Rep detection

**Option C-1 — peak detection** (prominence plus minimum-distance constraints, e.g.
`scipy.signal.find_peaks`). Accurate boundaries and few assumptions, but a peak is only
identifiable once passed, so a causal version counts roughly half a rep period late.
Acceptable for a count, poor for live feedback.

**Option C-2 — state machine with hysteresis.** Two thresholds on a scalar signal, entered
and exited in sequence — a Schmitt trigger. Jitter near a single threshold cannot produce
spurious counts. Transitions fire with no latency, the behaviour is trivially inspectable
when it misbehaves, and the thresholds are naturally data rather than code. Rep boundaries
are cruder than peak detection's.

**Option C-3 — learned, or DTW template matching.** A temporal model would need labelled
rep data that does not exist for this project; DTW against a reference rep avoids training
but costs per-frame computation and is difficult to debug when wrong. Rep counting is not
where this project's research value lies — joint angle accuracy is — so spending the
research budget here is a misallocation.

## Decision

**A-2 + B-1/B-3 + C-2**, with the conditions below.

- **Measurement path:** median prefilter on the contributing keypoints → compute angle →
  One Euro on the angle signal. Live and causal.
- **Display path:** an independently-tuned, deliberately heavier coordinate smoother
  driving the skeleton overlay only.
- **Harness path:** median → angle → Savitzky–Golay over a centred window. Non-causal and
  zero-lag, serving as the accuracy ceiling.
- **Rep detection:** hysteresis state machine on a per-exercise scalar signal.

Kalman is deferred rather than rejected. The single capability it offers over One Euro is
principled confidence fusion, and Condition 2 below obtains most of that benefit in a few
lines. If measurement shows that insufficient, a Kalman is added then — with a motion model
that does not assume constant velocity through the turnaround.

### Condition 1 — real timestamps, never assumed frame rate

One Euro requires `dt`. Camera frame delivery is irregular, and dropped or late frames are
normal under thermal load. Every filter must consume the actual frame timestamp. Assuming
30fps would silently mistune the filter in exactly the conditions where accuracy matters
most, and would do so invisibly.

### Condition 2 — confidence gating by hold-and-interpolate

SimCC confidence below a per-keypoint threshold means the sample is discarded rather than
filtered. Short gaps are bridged by interpolation and the One Euro state coasts. Gaps
longer than a tuned limit escalate to the **refuse-to-measure** path already established in
ADR 0002, Condition 5: the rep is flagged unmeasurable rather than reported with a
corrupted angle. Both thresholds are tunable parameters, not constants in code.

### Condition 3 — rep thresholds are ROM-relative, not absolute degrees

Absolute-degree thresholds do not transfer between users; ankle range varies widely in
healthy populations and more so in rehabilitation. Thresholds are stored as fractions of
the user's calibrated range for that session, in the ADR 0002 calibration record — for
example `{enter: 0.8, exit: 0.3}`. This also yields partial-rep detection at no additional
cost, since the fraction actually achieved is already computed.

### Condition 4 — the counting signal is separate from the measured quantity

For calf raises the quantity to *measure* is the ankle angle, which depends on the
noisiest keypoints available. The quantity to *count* is heel height, a single clean
y-coordinate. The exercise definition names each independently. Forcing one signal to serve
both purposes would import foot-keypoint noise into the rep count for no benefit.

### Condition 5 — the display filter never feeds measurement

The two filter chains are separate paths. Overlay-smoothed coordinates must not reach the
angle computation. This is stated because the failure is silent: the app would keep working
and the reported angles would simply be laggy and wrong, with nothing to indicate it.

### Condition 6 — the harness reports the causal penalty

For every accuracy figure produced, the harness reports the offline (Savitzky–Golay,
peak-detected) result and the online (One Euro, state machine) result on the same clips.
The difference is the cost of real-time causal processing, expressed in degrees of angle
error and in rep-count accuracy.

This is the third decision resolved by the same pattern — an offline ceiling and an online
approximation, with the gap reported as a finding. ADR 0002 established it for model choice
(RTMW ceiling versus shipped variant); it recurs here for filtering and for rep detection.
It is the project's characteristic structure and should be treated as deliberate: a
quantified statement that real-time processing costs N degrees is a stronger result than any
single accuracy number.

## Consequences

### Positive

- One filter is tuned against the metric that is actually reported, rather than 266 filters
  tuned against a proxy.
- Gross outliers are rejected before they can propagate into an angle, and peak amplitude —
  the ROM measurement itself — is preserved by both the One Euro and Savitzky–Golay choices.
- Rep detection is inspectable. When a count is wrong, the state and the two thresholds
  explain why, which is not true of a learned counter.
- Partial-rep feedback arrives free, and is genuinely useful guidance rather than a
  contrived feature.
- The filter and rep-detection parameters are per-exercise data, consistent with the
  no-hard-coded-ankle constraint from ADR 0002, so a second joint inherits the machinery.

### Negative — accepted

- One Euro's parameters are hand-tuned, per signal and potentially per exercise. There is no
  optimality criterion to appeal to; tuning is empirical, against the harness.
- Confidence is handled heuristically rather than fused. A Kalman's principled treatment is
  given up until evidence demands it.
- Hysteresis produces cruder rep boundaries than peak detection, which slightly degrades
  any per-rep statistic computed from those boundaries. The harness's peak-detected
  segmentation is the reference for this.
- Very shallow reps that never cross the enter threshold are not counted at all. This is
  intended, but it means the count reflects reps meeting a quality bar rather than attempts
  made, and the UI must say so.
- **ROM-relative thresholds create a framing hazard that must not reach the user
  unqualified.** A user with severely restricted range has thresholds calibrated to that
  restricted range, so the app will report "full rep" for a small movement. That is correct
  for counting and actively misleading as a statement about range. Rep completion
  ("relative to your range today") and range itself ("measured in degrees, tracked over
  time") must be presented as distinct quantities. Given the project's not-medical-advice
  constraint, conflating them would be the most consequential error available here.
- **Calibrated thresholds go stale as rehabilitation succeeds.** Improving range is the
  entire point, so a range captured once becomes wrong. Calibration is therefore
  per-session, and progress is tracked on the absolute measured angle, never on rep
  completion rate — which is deliberately self-normalizing and would show no improvement.

### Follow-ups this decision creates

- **Decision 4 (form assessment)** operates on the filtered angle signal and the state
  machine's phase output; rule thresholds join the same per-exercise definition record.
- **Decision 5 (storage)** must persist the session calibration, the filter and threshold
  parameters in force, and enough signal to re-analyse a session later. Whether raw
  per-frame keypoints are retained or only the derived angle series is an open storage
  question with a real size implication, and it determines whether past sessions can be
  re-scored after a filter change.
- **Decision 7 (evaluation harness)** inherits Condition 6 and must report paired
  offline/online figures rather than a single accuracy number.
- Parameter tuning needs recorded video before it can begin, which makes test-clip capture
  a prerequisite task rather than a later one.
