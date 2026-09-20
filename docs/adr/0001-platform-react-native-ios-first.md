# ADR 0001: Platform — React Native (Expo dev build), iOS-first

- **Status:** Accepted
- **Date:** 2026-09-20
- **Deciders:** Project owner

## Context

The app must run pose inference on-device: video never leaves the phone, which is a
hard privacy constraint of the project. That rules out any architecture where frames
are streamed to a server for inference.

The owner wants an *installable* artifact rather than a link-only web app, and is
experienced with Python/PyTorch and React/Vite/Tailwind, but new to pose estimation,
mobile/on-device ML, and time-series signal processing. The project is explicitly a
learning vehicle as well as a portfolio piece.

Available test hardware: **iPhone only**. No Android device.

This decision therefore has to satisfy: on-device inference, installable distribution,
maximum transferable learning, and a realistic path with one developer and one iOS
device.

## Options considered

### Option A — React Native (Expo dev build) with native frame processors

UI in React Native; camera and inference native. `react-native-vision-camera` frame
processors (worklets on a dedicated thread, running at camera frame rate) feed a
native inference runtime such as `react-native-fast-tflite`, which communicates over
JSI and can dispatch to the CoreML or Metal delegate on iOS.

- Performance: near-native; the JS bridge is not in the per-frame path.
- Complexity: medium, front-loaded. Camera-frame (YUV) to model-input (resized,
  normalized RGB) conversion is the usual throughput sink.
- Learning value: high and directly marketable — on-device runtimes, hardware
  delegates, quantization, latency budgeting.
- Reuse: React and Tailwind-equivalent styling knowledge carries to the UI and charts.
- Adjacent variants noted but not selected: community MediaPipe bindings
  (`react-native-mediapipe`), which return landmarks directly but concentrate
  maintenance risk in one package; and `react-native-executorch`, which would suit the
  owner's PyTorch background but is less mature than the vision-camera + TFLite path.

### Option B — Native iOS (Swift + AVFoundation + MediaPipe Tasks Vision)

- Performance: the ceiling. Best landmark quality per millisecond.
- Complexity: lowest for the ML integration (first-party libraries and sample apps),
  highest for everything else — Swift, SwiftUI, Xcode project management all new.
- Learning value: high, but a large share is spent on Apple platform idioms rather
  than on pose estimation or ML deployment.
- Reuse: essentially none of the owner's existing stack.

### Option C — PWA installed to the home screen

Vite + React + MediaPipe Tasks Vision JS (WASM/SIMD + WebGL).

- Performance: weakest. No control over exposure, focus lock, or capture frame rate.
  This matters more than usual here: autofocus hunting and exposure drift degrade
  landmark stability, and landmark stability is the measurement.
- Complexity: lowest; the owner already knows every piece.
- Learning value: lowest on the axis the owner named as a gap.
- Also fails the stated requirement that the artifact be installable.

## Decision

**Option A: React Native via an Expo dev build, targeting iOS first**, subject to two
conditions.

Option A is the only choice that satisfies both the installable requirement and the
on-device inference constraint while reusing existing skills and teaching the named
gaps. Option B buys better ML ergonomics at the price of learning Swift rather than
pose estimation. Option C fails the installable requirement outright.

An earlier draft of this decision recommended Android-first (free deployment, no
review queue, simpler CI, and "mid-range phone" being a fundamentally Android
benchmark). That was withdrawn once it was established that no Android test device is
available. Building for a platform one cannot test on is not viable.

### Condition 1 — de-risk the inference path before building on it

The first work item is a throwaway spike: run any pose model through a
vision-camera frame processor on a physical iPhone and report measured fps and
end-to-end latency. Nothing is built on top until that number exists. The specific
iOS risk to probe is delegate op coverage — if the pose model's operators are not
fully supported by the CoreML delegate, execution silently falls back to CPU and
throughput collapses. The spike must confirm *which* delegate actually runs the
graph, not merely that inference completes. If this fights for more than a week,
fall back to Option B.

### Condition 2 — decouple the research from the app

The project's core research question — whether single-camera 2D pose is accurate
enough for rehab measurement — does not require the app to answer. Recorded video
plus Python answers it completely and offline. The evaluation harness is therefore
built as pure Python operating on video files, independent of the client. A platform
setback then costs a demo, never the research result, and ML work can begin
immediately in parallel.

A short-lived PWA prototype (Option C as scratch work, not as a shipping target) is
also sanctioned, purely to get landmarks rendering on the owner's own ankle within
days and to surface the foot-landmark noise problem early.

## Consequences

### Positive

- Inference runs at native speed with the UI in a familiar framework.
- iOS gives strong camera control through AVFoundation — exposure lock, focus lock,
  explicit format and frame-rate selection. Because 2D angle accuracy depends on
  stable, repeatable capture, this is a genuine measurement advantage over both the
  Android device fragmentation story and the PWA.
- The owner learns on-device ML deployment, which was an explicit goal.
- CI for iOS requires macOS runners; on a public repository these are available at no
  cost, which suits an open-source portfolio project.

### Negative — accepted

- **Distribution is constrained.** Free Apple ID provisioning installs to the owner's
  own device but expires after seven days and requires re-signing. TestFlight
  distribution to anyone else, and any App Store listing, requires the paid Apple
  Developer Program (~$99/year). Consequence: nobody will casually try this app.
  The portfolio artifact is therefore the repository, the accuracy report, and a
  recorded demo video — not a link a reviewer clicks. A screen-capture demo in the
  README is not optional polish; it is the primary way the work will be seen.
- **The "mid-range phone" performance target loses its meaning.** There is no
  mid-range iPhone in the Android sense; every recent iPhone has a Neural Engine that
  will clear this workload easily. Reporting "30fps on an iPhone" is not a research
  finding. ADR on target metrics must reframe this: report latency against the named
  specific device, and add a device-independent cost measure (model MACs/FLOPs and
  CPU-only latency) so the number means something to a reader with different hardware.
- **Android is deferred indefinitely.** Cross-platform claims cannot be made or
  tested. React Native keeps the door open, but the door stays shut for the MVP.
- Some effort will be spent on frame-format plumbing that teaches mobile engineering
  rather than ML.

### Follow-ups this decision creates

- ADR 0002 (pose model) is constrained by what is available and performant as an
  on-device model with adequate foot landmarks.
- ADR on model export format (decision 6) must revisit TFLite-plus-delegate versus
  converting to Core ML directly. Core ML is the first-class Apple runtime and may
  outperform a delegated TFLite graph; the cost is losing a single portable artifact
  across platforms. The Condition 1 spike should gather evidence for this.
- ADR on target metrics must replace the "mid-range phone" framing per the above.
