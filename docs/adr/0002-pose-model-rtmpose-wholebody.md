# ADR 0002: Pose model — RTMPose-WholeBody on-device, MediaPipe as offline comparison

- **Status:** Accepted (conditional — see Condition 1)
- **Date:** 2026-09-20
- **Deciders:** Project owner
- **Supersedes:** ADR 0001, Condition 1 (the inference spike is redefined below)

## Context

ADR 0001 fixed the platform: React Native (Expo dev build), iOS-first, inference
on-device, with the Python evaluation harness decoupled from the app and operating on
recorded video.

The pose model must serve the MVP's three ankle exercises. What each one actually needs
from a skeleton:

| Exercise | Measurement | Landmarks required |
|---|---|---|
| Calf raise | rep count, peak plantarflexion | ankle, **heel**, **toe** (ankle height alone suffices for reps) |
| Dorsiflexion ROM (weight-bearing lunge) | tibia inclination from vertical | knee, ankle |
| Single-leg balance | sway, hold duration | hip, knee, ankle |

Two facts follow.

First, the standard clinically-reported home measure of ankle dorsiflexion is the
**weight-bearing lunge test**, expressed as knee-to-wall distance or as tibial
inclination from vertical. That is a knee→ankle vector. The project's most
evidence-defensible measurement is also the one least dependent on noisy landmarks, and
the app should lean on it.

Second, **plantarflexion cannot be measured without a foot segment.** Any model whose
lowest lower-limb keypoint is the ankle can count reps but can never report an ankle
joint angle. This is the decisive criterion.

A fourth force, stated by the owner during this decision: although the MVP scope is
ankle-only, the intended destination is a general rehab tool covering other joints. The
model choice is therefore a long-lived commitment in a way the MVP scope alone would not
suggest, and keypoint coverage beyond the foot carries real option value.

A third consideration is integration. ADR 0001 selected `react-native-vision-camera`
frame processors feeding `react-native-fast-tflite`, a path that expects a single
self-contained model file. Not every candidate fits that shape, and the mismatch is
front-loaded work that must be discovered before it is built upon.

## Options considered

### Option A — MediaPipe Pose Landmarker (BlazePose, 33 landmarks)

Apache-2.0, three size variants (lite/full/heavy). Provides ankle, **heel** and
**foot_index** per side, a per-landmark visibility score, and a second "world landmark"
output giving root-relative metric 3D coordinates.

- **Foot coverage:** adequate — three points per foot, sufficient for a foot-segment
  vector. These are, however, the lowest-confidence landmarks in the topology and
  degrade sharply with footwear, motion blur, and cluttered flooring.
- **Speed:** comfortably real-time at `full` on any recent iPhone GPU.
- **Ecosystem:** first-party iOS framework and first-party Python API (`mediapipe.tasks`),
  so the app and the evaluation harness can run bit-identical inference — app-versus-harness
  discrepancy is eliminated as a variable.
- **Integration cost:** higher than it appears. Pose Landmarker is a two-stage graph
  (person detector → ROI crop/rotate/scale → landmark model → decode → tracking) and most
  of that logic lives in the MediaPipe runtime, not in the model file. It cannot simply be
  dropped into `fast-tflite`. It requires either the MediaPipe Tasks iOS framework behind a
  thin native module (or `react-native-mediapipe`, whose maintenance-concentration risk
  ADR 0001 already noted), or a reimplementation of the two-stage pipeline in a worklet.

### Option B — MoveNet Lightning/Thunder (17 COCO keypoints)

Apache-2.0, single-stage, single file, the cheapest possible integration and a natural fit
for the `fast-tflite` path.

**Rejected on the decisive criterion:** no heel, no toe. The ankle is the terminal
lower-limb keypoint, so ankle joint angle is unobtainable at any accuracy. It could serve
rep counting and the lunge-test tibia angle and nothing further, which is a material
reduction of MVP scope.

Apple's `VNDetectHumanBodyPoseRequest` was considered inline and rejected for the same
reason (~19 joints, ankles terminal), despite being free and zero-integration-cost on an
iOS-first build.

### Option C — RTMPose-WholeBody / RTMW (133 keypoints, six per foot)

OpenMMLab. SimCC coordinate-classification head, which avoids heatmap decoding cost and
yields a per-keypoint confidence from the SimCC peak. COCO-WholeBody topology supplies
**big toe, small toe and heel per side** — the best foot coverage of any candidate — and
the strongest published accuracy.

- **Deployment cost:** the highest of the three. Top-down architecture means **two models**
  (person detector, typically RTMDet-nano, then the pose model), an ONNX → Core ML
  conversion step, and no established React Native precedent.
- **Lineage:** PyTorch-native, which matches the owner's existing strengths and the stated
  goal of learning model export and on-device deployment end to end.
- **Licensing:** MMPose code is Apache-2.0, but published checkpoints vary by training
  corpus, and some (AI Challenger, Human-Art) carry restrictive terms that the code license
  does not cover.

## Decision

**Option C — RTMPose-WholeBody on-device**, subject to the five conditions below.

The recommendation presented was Option A, on the grounds that it is the only candidate
that measures the ankle joint *and* ships on a realistic single-developer timeline. The
owner selected Option C. The deciding reason is **extensibility beyond the ankle**: the
COCO-WholeBody topology covers every joint a future rehab exercise might target, including
hands, which Option A cannot reach without adding a second model. Against an ankle-only
MVP this counts for nothing; against the owner's stated destination it is the dominant
consideration, and it outweighs foot-keypoint fidelity as a justification. Secondary
reasons: research-grade accuracy on the project's core measurement, and a PyTorch-native
export path that serves the learning goal more directly than consuming a prebuilt runtime
would. The deployment risk recorded in Consequences is accepted against these.

The risk is real and is recorded in Consequences. Conditions 1–3 exist to bound it.

### Condition 1 — the inference spike is redefined (supersedes ADR 0001 Condition 1)

The first work item is a throwaway spike: RTMDet-nano + a small RTMPose-WholeBody variant,
exported ONNX → Core ML, executed inside a `vision-camera` frame processor on a physical
iPhone. It must report:

1. measured end-to-end fps and latency, detector and pose stages timed separately;
2. **which compute unit actually executed each graph** (Neural Engine, GPU or CPU) —
   established via the Core ML performance report in Xcode, not inferred from the fact that
   inference completed. Silent CPU fallback on unsupported ops is the specific failure mode
   being probed, and it is doubled here relative to a single-model pipeline;
3. foot-keypoint confidence on real barefoot ankle footage, not on stock test images.

**Time box: two weeks.** On expiry, the fallback is **Option A via the MediaPipe Tasks iOS
framework** — change the model, not the platform. ADR 0001 named native Swift as the
fallback for a platform failure; that remains correct for a platform failure, but a model
deployment failure is cheaper to resolve by swapping the model.

### Condition 2 — amortize the detector

A top-down pipeline nominally runs two models per frame. It must not. The detector runs
once to acquire the person, after which each frame's bounding box is derived from the
previous frame's keypoints, with re-detection triggered only on confidence collapse or
tracking loss. This is what MediaPipe does internally and is the difference between a
viable and a non-viable frame budget. The spike measures both modes so the saving is
documented rather than assumed.

### Condition 3 — verify checkpoint licensing before any public claim

The specific checkpoint shipped must have its training corpus and license confirmed, and
recorded in the repository. Apache-2.0 on the MMPose codebase does not transfer to weights.
Where a permissive claim is wanted, prefer a COCO-WholeBody-only checkpoint. This is a
blocking item for the README, not a cleanup task, because the model is shipped rather than
merely used for research.

### Condition 4 — MediaPipe becomes the offline comparison baseline

With RTMPose shipping on-device, using RTMPose as the harness reference would compare a
model against itself and measure nothing. The roles invert. The Python harness runs, over
the same recorded clips and against the same goniometer ground truth:

- the shipped on-device RTMPose variant (quantized, phone-sized);
- **MediaPipe Pose Landmarker**, as the commodity baseline — answering "what would the
  easy, popular, low-integration-cost option have cost in degrees?";
- a large RTMW variant, as the accuracy ceiling — answering "what does shrinking to phone
  size cost in degrees?"

Both questions are genuine findings and neither is answerable once the harness runs a
single model.

### Condition 5 — out-of-plane detection by 2D calibration, not learned 3D

Angles are measured in **2D image space** with explicit camera-setup constraints. RTMPose
produces no world-landmark output, so the previously-considered approach of using a model's
3D regression as an out-of-plane signal is unavailable — and is not missed. Regressed 3D
coordinates are a learned prior about body shape, not a measurement; for the ankle, the
smallest and most occluded segment, they would look plausible while being wrong, with no
error signal attached. Measuring against them would silently convert the project's research
question from "is single-camera 2D pose accurate enough?" into "does the model agree with
itself?"

The replacement is geometric and calibrated per user during camera setup:

- at setup, capture a neutral reference frame and store the **foot-length : shank-length
  pixel ratio** for that user, that camera position, and that session;
- at runtime, deviation of that ratio beyond a tuned threshold indicates the limb has
  rotated out of the image plane;
- such reps are **flagged unmeasurable and refused**, not silently reported with a
  corrupted angle.

Refusing to measure is the correct behaviour for an application that presents health
information, and the threshold remains a tunable knob rather than a constant baked into
code.

## Consequences

### Positive

- The best available foot-keypoint fidelity, which is the project's binding measurement
  constraint. Six foot keypoints per side allow a foot-segment vector with redundancy,
  rather than the two-point minimum Option A would have provided.
- The export path (PyTorch → ONNX → Core ML) is exercised first-hand, which is the
  on-device ML deployment skill ADR 0001 named as a learning objective. Option A would have
  delegated that work to a prebuilt runtime.
- SimCC per-keypoint confidence is available for downstream filtering and for the
  refuse-to-measure logic.
- Every joint a future exercise could target is already covered, so extending past the
  ankle is an exercise-definition change rather than a model migration. This also makes
  UI-PRMD and KIMORE materially more useful than they would otherwise be: both consist
  largely of shoulder, trunk and knee movements, which an ankle-only system could not
  exploit.
- The comparison against MediaPipe (Condition 4) produces a defensible, quantified claim
  about model choice rather than an assertion, which is a stronger portfolio result than
  shipping either model alone.

### Negative — accepted

- **This is the highest-risk deployment path of the three options.** Two graphs, a
  conversion step, and no React Native precedent. Condition 1 bounds the exposure to two
  weeks, but the possibility of spending those two weeks and falling back to Option A is
  real and accepted.
- **Silent CPU fallback risk is doubled** relative to a single-model pipeline, since either
  graph can fail delegate op coverage independently.
- The app and the harness will not run identical inference code. Any discrepancy between
  on-device and offline results becomes a variable that must be investigated rather than
  excluded, which Option A would have eliminated.
- Most of the 133 keypoints are unused during the ankle-only MVP, and face and hand
  keypoints are pure compute cost until other joints are supported. Pruning the head to a
  body-plus-foot subset would buy latency now at the price of the extensibility that
  motivated this choice, so it is **explicitly not** an optimization to reach for. If the
  Condition 1 spike shows the full model cannot hold frame rate, the correct response is a
  smaller variant or a different quantization point, not a narrower keypoint set.
- Licensing is now a blocking pre-release item rather than a research footnote.

### Follow-ups this decision creates

- **Decision 3 (signal processing)** must filter SimCC-derived keypoints with per-keypoint
  confidence as a gating input, and must specify behaviour when the Condition 5 calibration
  check fails mid-rep.
- **Decision 5 (data model)** must persist the per-session calibration ratio from
  Condition 5 alongside session data, since measurements are only interpretable relative to
  the setup that produced them.
- **Decision 6 (model export)** is now substantially pre-decided toward ONNX → Core ML.
  The open sub-question is the quantization strategy and the size/accuracy point chosen for
  the shipped variant, which Condition 4's harness will quantify.
- **Decision 7 (evaluation harness)** inherits a three-model comparison rather than a
  single-model accuracy report.
- Nothing in the application may hard-code the ankle. Joint indices, the angle definition,
  the rep-detection thresholds and the form rules are per-exercise **data**, not code, so
  that adding a joint is a new exercise definition rather than a refactor. This is a
  constraint on Decisions 3, 4 and 5, and is cheap only if adopted before they are made.
  It does not license building an exercise plugin framework — a plain definition record
  per exercise is the whole requirement.
- Accuracy claims do not generalize across joints. Validating ankle angle against a
  goniometer says nothing about shoulder abduction accuracy; each supported joint needs
  its own ground-truth validation before any claim is made about it.
- Camera setup guidance now carries a functional duty, not merely an advisory one: it
  produces the calibration values the refuse-to-measure logic depends on.
