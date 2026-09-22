# ADR 0006: ML workflow — Core ML via coremltools, FP16, harness executes the shipped artifact

- **Status:** Accepted
- **Date:** 2026-09-21
- **Deciders:** Project owner
- **Refines:** ADR 0002's follow-up, which anticipated an ONNX-to-Core ML conversion path
- **Builds on:** ADR 0001 (decoupled Python harness), ADR 0005 (provenance requires a stable
  model identifier)

## Context

### The MVP trains nothing

RTMPose is used pretrained. Fine-tuning for foot keypoints would require annotated ankle
imagery that does not exist for this project, and producing it is an annotation task, not an
infrastructure task. What the brief calls the "ML workflow" is therefore a **conversion and
evaluation** workflow. No training pipeline, no experiment tracking, no compute budget, and
none of the MLOps scaffolding a portfolio project is tempted to build before it has anything
to train.

Fine-tuning is deferred to phase 2, where it begins with annotation rather than with tooling.

### ADR 0002's assumed conversion path is not available

ADR 0002 recorded that this decision was "substantially pre-decided toward ONNX → Core ML".
That toolchain no longer exists: **coremltools removed its ONNX front-end** — deprecated at
version 5, removed at version 6. Apple's supported route converts from PyTorch directly, via
TorchScript trace or exported program into `ct.convert()`.

This is a simplification rather than an obstacle, since RTMPose is a PyTorch model and ONNX
was an unnecessary intermediate. Its consequence for this decision is that **ONNX is a
distinct architectural option rather than a waypoint**, and must be evaluated as one.

### Two inherited requirements constrain the choice

ADR 0002, Condition 1 requires the spike to establish **which compute unit actually executed
each graph**, because silent CPU fallback on unsupported operators is the specific failure
mode being probed. ADR 0005, Condition 2 requires a stable model identifier, because session
provenance records the model variant and quantization in force.

### Dependency fragility is a real force here

MMPose and MMDeploy carry tightly pinned dependency trees, with `mmcv` matched against
specific torch and CUDA versions. MediaPipe, coremltools and MMPose coexisting in one Python
environment is a predictable source of lost days. Where these dependencies are allowed to
live is part of this decision, not a detail of it.

## Options considered

### 6a — Client runtime and artifact format

**Option A — Core ML (`.mlpackage`) via coremltools.** PyTorch → TorchScript → `ct.convert()`.
First-party, best Neural Engine access. The decisive property is observability: **Xcode's
Core ML performance report reports which compute unit ran each layer**, which is exactly what
ADR 0002's Condition 1 demands and which no other option supplies as directly. Apple-only.

**Option B — ONNX with ONNX Runtime Mobile and the CoreML execution provider.** Retains a
single portable artifact, keeping Android reachable, and Microsoft publishes
`onnxruntime-react-native`. The objection is specific: the CoreML execution provider falls
back to CPU for unsupported operators **silently**, which is the precise failure the spike
exists to detect, and it offers less visibility into partitioning than Core ML's own tooling.
The React Native binding advantage is also smaller than it appears, since a native
vision-camera frame processor plugin must be written under either option.

**Option C — ExecuTorch with the Core ML backend.** PyTorch-native end to end and closest to
the owner's existing skills. ADR 0001 assessed its maturity as below the alternatives; that
assessment is unchanged.

### 6b — Quantization

**FP16.** The Neural Engine is fp16-native, so this is close to free: size halves, accuracy
loss is typically negligible, no calibration data required.

**FP16 with weight palettization (4/6/8-bit).** Further size reduction with activations left
at fp16, so accuracy risk stays modest. Additional conversion complexity.

**INT8 post-training quantization.** The conventional next step, and probably pointless on
this target: int8 pays off mainly on CPU and int8-oriented accelerators, while the Neural
Engine will run fp16 regardless. It is a size win, and size is not a constraint here — tens
of megabytes in an app bundle is unremarkable. It also carries genuine accuracy risk on a
coordinate-classification head.

### 6c — Which model the harness executes

**The PyTorch model.** Simplest, full framework access, intermediates inspectable. Measures a
model that differs from the shipped one by both conversion and quantization.

**The exported artifact.** coremltools can execute an `.mlpackage` directly in Python on
macOS, which the owner has. Same graph, same weights, same quantization as the device.

**Both, reporting the delta.** Consistent with the project's offline-ceiling pattern, at the
cost of a third set of numbers.

## Decision

**Core ML via coremltools; FP16; the harness executes the exported artifact.**

### Condition 1 — ONNX is the fallback and the Android path, not a waypoint

The conversion is PyTorch → TorchScript → Core ML, with no ONNX step. ONNX with ONNX Runtime
Mobile remains the documented fallback if Core ML conversion fails, and is the route Android
would take if it is ever revisited. This supersedes the corresponding follow-up in ADR 0002.

### Condition 2 — quantization is evaluated in degrees, not in OKS

Any change to numeric precision is assessed by its effect on **joint angle error against
goniometer ground truth**, not by a model-level metric. An OKS or mAP drop of a few tenths
says nothing about whether the ankle measurement moved by a degree and a half. The project
has a downstream task metric; model metrics are not a substitute for it.

FP16 is therefore provisional: it is the starting point, not a finding. If the Condition 1
spike shows latency headroom is absent, palettization is the next rung, and INT8 only with
measured justification.

### Condition 3 — conversion is a pinned, containerized script with a manifest

Not a notebook. The conversion runs in a container with pinned versions and emits the
artifact plus a manifest recording source checkpoint, conversion settings, precision, input
resolution and toolchain versions. The model identifier combines a readable name with a
content hash of the artifact — for example `rtmpose-s-wholebody-256x192-fp16-<sha8>` — so
that ADR 0005's provenance field refers to something verifiable rather than to a label
someone typed. Artifacts are published to GitHub Releases with a fetch script; binaries do
not go into git.

### Condition 4 — numerical parity gate before an artifact is accepted

Conversion silently changing behaviour is a known failure mode. Before any artifact is
released, PyTorch and Core ML outputs are compared on a fixed set of input frames, and the
maximum per-keypoint displacement in pixels must fall below a stated tolerance. An artifact
failing this gate is not shipped, regardless of how well it performs elsewhere. This check is
part of the conversion script, not a manual step.

### Condition 5 — model runtimes are isolated per runner

MMPose stays inside the conversion container and is never imported by the harness. Each
comparison model from ADR 0002, Condition 4 — the shipped Core ML artifact, MediaPipe, the
RTMW ceiling — runs as its own containerized runner with its own dependencies, emitting a
**common keypoint-series format**. The harness orchestrates runners and analyses their output;
it does not host their dependency trees. This is necessary rather than ornamental: MediaPipe,
MMPose and coremltools in one environment is a predictable source of lost days.

### Condition 6 — precompile the model at build time

Core ML models compile to `.mlmodelc`, and Neural Engine compilation can take seconds on
first load. Compilation happens at build time so that a user's first session does not begin
with a stall.

## Consequences

### Positive

- The compute unit that executes the graph is directly observable, so ADR 0002's Condition 1
  can actually be satisfied rather than approximated.
- The accuracy report describes the model that ships, including its conversion and
  quantization, rather than an upstream model that resembles it.
- The conversion step is reproducible and its output is verifiable by hash, which makes
  ADR 0005's provenance meaningful instead of decorative.
- The most fragile dependency tree in the project is confined to a container that runs rarely
  and is never a prerequisite for analysis work.
- No training infrastructure is built, and the absence is deliberate and recorded rather than
  an oversight.

### Negative — accepted

- **The shipped artifact is Apple-only.** Android would require a second conversion path and
  a second set of accuracy numbers. ADR 0001 already deferred Android indefinitely; this
  closes the door further.
- **The harness now requires macOS** to execute the shipped artifact, so accuracy checks
  cannot run on Linux CI runners. ADR 0001 established that macOS runners are free for public
  repositories, so this is affordable, but it is a constraint on CI design rather than a free
  choice.
- **Without training, accuracy is whatever the pretrained model gives.** If ankle measurement
  proves insufficient, the available levers are camera setup, model variant and measurement
  definition — not fine-tuning. This is a genuine project risk: the core research question
  could resolve negatively with no remediation path inside the MVP.
- FP16 is provisional and may not survive the spike.
- Per-runner containerization adds orchestration that a single Python environment would not
  need, and is justified by dependency conflict rather than by design elegance.

### Follow-ups this decision creates

- **Decision 7 (evaluation harness)** owns the runner architecture, the common keypoint-series
  format, and the Condition 4 parity fixtures. It also inherits a macOS requirement.
- **Decision 8 (repo structure)** must accommodate the conversion container, the artifact
  fetch script, and per-runner environments, alongside the shared schema and fixture set that
  ADR 0005 already requires.
- ADR 0002's Condition 1 spike becomes concrete: TorchScript trace of a small
  RTMPose-WholeBody variant, `ct.convert()` to FP16 Core ML, executed in a vision-camera frame
  processor, with the Xcode Core ML performance report as the evidence that the Neural Engine
  ran the graph.
- The common keypoint-series format is shared between runners, the harness, and ADR 0005's
  session export. It should be defined once.

## Amendment (2026-09-21): conversion and the parity gate run on different hosts

Condition 3 put conversion in a pinned container; Condition 4 requires the parity gate to
execute the exported artifact. Those two requirements are in tension, because Core ML
prediction only runs on macOS, and the container from Condition 3 is Linux (mmpose/mmcv have
no reliable macOS build path, which is exactly why they're contained in the first place).

The resolution is a split, not a single script: `ml/convert/convert.py` runs in the container
and does the PyTorch-side work — trace, `ct.convert()`, and running the *PyTorch* model
(not the Core ML one) over a fixed reference frame set, saving both as `.npy`.
`ml/convert/parity.py` runs separately on the macOS host, loads the `.mlpackage` outputs and
the saved PyTorch references, and does the actual PyTorch-vs-Core-ML comparison Condition 4
requires. CI mirrors the split: the existing Ubuntu job is unchanged, and a new macOS job
fetches released artifacts and runs `parity.py` there.

One conversion-correctness finding worth recording: at plain FP16, the pose head's
GAU/linear layers produced per-keypoint SimCC peaks up to ~100px off from the PyTorch
reference on some frames, traced to fp16-range overflow in constant-folded weights during
conversion (confirmed by converting the same graph at FP32, which matched exactly). The
fix is `ct.transform.FP16ComputePrecision` with an `op_selector` that keeps `linear` /
`matmul` ops at FP32 while the rest of the graph (the CSPNeXt backbone, where the compute
actually is) stays FP16 — not a retreat from Condition 2's FP16 starting point, since the
Neural Engine cost is concentrated in the backbone convolutions, not this handful of head
ops.
