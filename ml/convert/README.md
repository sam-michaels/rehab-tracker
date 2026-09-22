# ml/convert

PyTorch -> Core ML for the two spike models (ADR 0006). Conversion runs in a container
(mmdet/mmpose are too fragile to install next to the harness); the parity gate runs on
the macOS host because Core ML prediction doesn't work on Linux (ADR 0006 amendment).

## Convert

```sh
docker build --platform linux/amd64 -t rehab-convert ml/convert
docker run --rm --platform linux/amd64 -v "$(pwd)/ml/convert/out:/workspace/out" rehab-convert --out /workspace/out
```

`--platform linux/amd64` is needed on Apple Silicon: mmcv has no arm64 CPU wheels.
Writes `out/*.mlpackage`, `out/*.zip` (release upload form), `out/manifest.json`
(checkpoint provenance, toolchain versions, artifact hashes) and `out/reference/`
(PyTorch reference in/out `.npy` pairs for parity.py).

## Parity gate

```sh
uv run ml/convert/parity.py ml/convert/out
```

Runs on the macOS host. Compares Core ML predictions against the PyTorch reference on
the fixed frame set; fails if the detector box or any pose keypoint (scored > 0.3 in the
reference) drifts more than 2px. Each model is checked on the compute units the app allows
it (see `COMPUTE_UNITS`, mirrors `HybridPose.swift`).

**The pose model is GPU-only.** On CPU or Neural Engine its SimCC peaks collapse and every
keypoint lands at (191.5, 0), 265 px off -- so the app pins `.cpuAndGPU`. CI runs with
`PARITY_SKIP=pose` because GitHub's macOS runners have no usable GPU; run this gate on an
Apple silicon Mac whenever the models change. Fixing the model itself (whole pose head at
FP32, re-converted) is the real fix and is not done yet.

## Fetch (on a release)

```sh
ml/convert/fetch.sh <release-tag>
```

Downloads the zipped `.mlpackage`s + manifest from a GitHub Release, verifies sha256,
unpacks into `app/modules/pose/ios/models/`.
