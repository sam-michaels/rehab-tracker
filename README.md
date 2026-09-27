# rehab-tracker

On-device pose estimation for measuring joint range of motion during rehabilitation
exercises, starting with the ankle. Video never leaves the phone.

> **Not medical advice.** This is a research and learning project with no clinical
> validation. Do not use it to make decisions about your treatment.

## Layout

```
app/        React Native (Expo dev build), iOS-first
ml/         convert/ · runners/ · harness/ · reports/   (Python)
shared/     schemas/ · definitions/ · fixtures/         (language-neutral JSON)
docs/adr/   architecture decisions
```

Why it's built this way: [docs/adr/](docs/adr/).

## Reports

Measurement findings live in [ml/reports/](ml/reports/):

- [Inference spike (2026-09-22)](ml/reports/2026-09-22-inference-spike.md) — RTMDet-nano +
  RTMPose-s-WholeBody on iPhone: 30 fps at ~14 ms/frame with the detector amortized, and why
  the pose model is GPU-only for now.

What the shipped models are, where they came from and their licence terms:
[ml/MODEL_CARD.md](ml/MODEL_CARD.md).

## Licence

Code: [Apache-2.0](LICENSE). The app runs MediaPipe's pose models (Apache-2.0), fetched from
Google, and without the MediaPipe SDK, which would report usage metrics to Google
([ADR 0002 amendment](docs/adr/0002-pose-model-rtmpose-wholebody.md#amendment-2026-09-25-the-app-store-build-ships-mediapipe-rtmpose-stays-research-only)).
The RTMPose/RTMDet weights in release `v0-spike` are a research comparison only: they are
licensed for **non-commercial research** (pose model: CC BY-NC-SA 4.0) because of their
training data, and never ship in the app. See [ml/MODEL_CARD.md](ml/MODEL_CARD.md).

## Public by design

This repository is public partly so that macOS CI runners are free (ADR 0008, Condition 4).
Making it private would change what CI costs.
