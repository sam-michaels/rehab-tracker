# Inference spike report — RTMDet-nano + RTMPose-s-WholeBody on iPhone

- **Date:** 2026-09-22
- **Answers:** ADR 0002, Condition 1 (spike deliverable) and Condition 2 (detector amortization)
- **Verdict:** the pipeline meets the frame budget with headroom. Two Condition 1 items are
  **not yet answered** (compute-unit attribution for the detector; foot confidence on
  barefoot footage), and ADR 0002 Condition 3 (checkpoint licensing) is still open.

## Setup

| | |
|---|---|
| Device | iPhone 17 Pro Max, iOS 27 |
| Build | Expo dev build (Debug), `spike/inference` at `fed7e746` |
| Camera | front, 1280×720 (`useFrameOutput` default), 30 fps cap, indoor evening light |
| Subject distance | ~2–3 m, full body in frame |
| Models | `rtmdet-nano-person-320-fp16`, `rtmpose-s-wholebody-256x192-fp16` (release `v0-spike`) |
| Execution | Vision (`VNCoreMLRequest`) does crop, scale and colour conversion; detector `.all`, pose `.cpuAndGPU` |

Numbers are the rolling 120-frame summaries the spike screen logs every 2 s
(`src/pose/stats.ts`), taken while standing in view. Mode A re-detects every frame; mode B
derives the crop from the previous frame's body keypoints (ADR 0002, C2).

## 1. Frame rate and per-stage latency

| | Mode A (detect every frame) | Mode B (tracked crop) |
|---|---|---|
| fps | ~30 (camera cap) | ~30 (camera cap) |
| detector p50 / p95 | 10.4–11.5 / 12.0–12.7 ms | 0 ms when tracking holds |
| pose p50 / p95 | 11.4–12.4 / 12.5–14.1 ms | 13.5–14.6 / 15.4–16.5 ms |
| **total p50 / p95** | **22.0–24.0 / 24.2–25.6 ms** | **13.6–14.6 / 15.4–16.5 ms** |
| redetect rate | 1.0 | 0.0 while tracking; 0.3–0.5 with entry/exit |
| dropped frames | see note | see note |

**Condition 2 is satisfied and measured, not assumed: tracking removes ~9 ms per frame,
about 38% of the total.** Against a 33 ms budget at 30 fps, mode A leaves ~9 ms of headroom
and mode B ~19 ms.

Dropped frames are counted cumulatively per app launch and were not reset between mode
switches, so the counter does not attribute drops to a mode. Observed totals stayed in the
single digits to ~19 over multi-minute sessions that included walking in and out of frame;
`frame-was-late` warnings were frequent before the crop and compute-unit fixes and rare after.
Treat drop counts as indicative only.

One counter-intuitive result: **the pose stage is ~2 ms slower in mode B** (13.9 vs 11.5 ms
p50). The tracked crop is derived from keypoints and is often larger than the detector-derived
one, and is clamped to the frame edge when a standing body's 3:4 crop doesn't fit a 9:16
portrait frame. The detector saving dwarfs it.

## 2. Which compute unit executed each graph — **partially answered**

Condition 1 asks for this from Xcode's Core ML performance report. That was not run. What the
spike established instead, empirically and more sharply, is stronger than a fallback report:

**The pose model is wrong on the Neural Engine, and wrong on the CPU.** It does not fall back
silently — it returns confidently wrong output. Every body keypoint collapses to the SimCC
argmax extremes, mapping to (191.5, 0) in model input pixels, while scores stay plausible.
Measured by `ml/convert/parity.py` against the PyTorch reference:

| Compute units | Pose worst-case error |
|---|---|
| CPU only | 264.9 px |
| CPU + Neural Engine | 264.9 px |
| **CPU + GPU (shipped)** | **0.7 px** |
| `.all` (Core ML chooses) | 2.0 px |

`convert.py` already keeps the pose head's `linear`/`matmul`/`einsum` ops at FP32 because
they overflow FP16 — the Neural Engine computes FP16 only, so that pin cannot hold there. The
CPU failing identically was not predicted by that explanation and is **unexplained**: patching
the model's two FP16 `inf` clip bounds to 65504 changed nothing.

The detector is unaffected (0.8 px on the Neural Engine, 0.1 px on the GPU) and still runs
`.all`. **Which unit actually executes it is still unknown** and needs the Xcode performance
report to close Condition 1 item 2.

ADR 0006's Option B objection — that ONNX Runtime's CoreML provider falls back silently — was
correct in spirit but understated the risk. Core ML's own tooling gave no error either; only a
parity gate against the PyTorch reference caught it, and only once it was told to test each
compute unit explicitly. The gate now does (`COMPUTE_UNITS` in `parity.py`), which is the
durable outcome of this spike.

## 3. Foot-keypoint confidence — **provisional**

Mean score over COCO-WholeBody keypoints 17–22 (big toe, small toe, heel per side):

| View | Foot conf | Body conf (0–22) |
|---|---|---|
| Facing camera | 0.50–0.77 | 0.72–0.83 |
| Side-on (either direction) | 0.33–0.43 | ~0.6 |
| Leg lifted toward camera | drops, points go below the 0.3 gate | 0.5–0.6 |

Side-on is the calf-raise view and the weakest case: the near foot occludes the far one, and
both legs' keypoints converge. 0.33–0.43 sits just above the 0.3 gate, which is thin margin.
Heel height did track raises: the left heel's y moved over a ~0.03 normalized range across
snapshots, consistent with real raises.

**This does not yet satisfy Condition 1 item 3**, which requires *real barefoot ankle
footage*. Footwear during these sessions was not recorded, lighting was ordinary indoor
evening light, and no goniometer ground truth was involved. Re-run barefoot, side-on, with the
ankle well lit before treating these numbers as the model's verdict.

## 4. Defects found and fixed

Each of these individually produced plausible-looking output, which is why they stacked:

1. **Neural Engine collapse** (above). Pose now pinned to `.cpuAndGPU`.
2. **Pose crop aspect.** `expandToPoseROI` fitted the model's 3:4 in *normalized* units on a
   9:16 frame, stretching the subject ~1.8× wide into the model. Body conf 0.25 → 0.8.
3. **Detector input squashed.** `scaleFill` compressed the frame into 320×320; a couch
   outscored a person at distance. Now `.scaleFit` (letterbox) with the padding undone.
4. **Tracker crop from all 133 keypoints.** One stray confident face/hand point at distance
   blew the crop across the frame. Now body+feet (0–22) only.
5. **Worklet capture.** Constants used only in default parameter values are not captured into
   the frame-processor runtime; `DEFAULT_MIN_CONFIDENCE` was undefined on the camera thread,
   throwing once per frame. Defaults moved into function bodies.
6. **Vision ROI bounds.** Vision re-derives the ROI for the buffer orientation and rejects an
   edge-touching rect that rounds to -1e-17. ROIs are clamped and inset 1e-6.
7. **iOS 27 scene lifecycle.** The app aborted at launch until `enableSceneSupport` was set
   (`expo-build-properties`).

## 5. Verdict against the time box

The two-week box (ADR 0002, C1) has not expired and the fallback (MediaPipe via Option A) is
not triggered: the pipeline runs at 30 fps with ~19 ms of headroom in mode B, and the
measurement the MVP depends on — heel and toe tracking through a calf raise — works on device.

**Continue with Option C, subject to:**

1. **Re-convert the pose head at FP32** so the model is correct on every compute unit. Pinning
   `.cpuAndGPU` avoids the failure; it does not fix it, and Core ML may still schedule work on
   the CPU. Until then, CI cannot check pose (GitHub's macOS runners have no usable GPU), and
   the gate runs with `PARITY_SKIP=pose`.
2. **Confirm checkpoint licensing (ADR 0002, C3) — blocking.** `manifest.json` records the
   corpus as COCO-WholeBody + UBody (DWPose distillation). MMPose's Apache-2.0 does not
   transfer to weights, the weights are now published in release `v0-spike`, and the ADR makes
   this blocking for the README. If UBody's terms don't permit it, switch to a
   COCO-WholeBody-only checkpoint.
3. **Close Condition 1**: Xcode Core ML performance report for the detector, and foot
   confidence on barefoot footage.

## Reproducing

```sh
ml/convert/fetch.sh v0-spike                              # models + manifest, sha256 verified
uv run ml/convert/parity.py app/modules/pose/ios/models   # full gate; needs Apple silicon
cd app && npx expo run:ios --device                       # spike screen: mode toggle, auto-log every 2 s
```

The spike screen logs one JSON line per 2 s with fps, per-stage p50/p95, redetect rate,
foot/body confidence, detector score, dropped frames and the current crop. The app code is
throwaway by charter (ADR 0002, C1).
