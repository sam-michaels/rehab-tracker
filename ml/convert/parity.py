# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = ["coremltools>=7.2,<8", "numpy<2", "pillow"]
# ///
"""Parity gate (ADR 0006 C4): PyTorch vs Core ML on the reference frames convert.py wrote.
Runs on the macOS host -- Core ML prediction doesn't work in the Linux conversion container.

    uv run ml/convert/parity.py <outdir>

Exits non-zero if detector box or pose keypoints (scored keypoints only, ref score > 0.3)
drift more than TOLERANCE_PX from the PyTorch reference, on any compute units the app lets
Core ML use. Default units on a Mac pick the GPU, which hid that the Neural Engine (FP16-only)
collapses the pose head's FP32-pinned ops -- so each model is checked per unit explicitly.
"""
import json
import os
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

TOLERANCE_PX = 2.0
SCORE_THRESHOLD = 0.3

# Mirrors computeUnits in app/modules/pose/ios/HybridPose.swift; keep in sync. The detector
# runs with .all, which on iPhone may land on the Neural Engine or the GPU, so check both.
# ponytail: GitHub's macOS runners are VMs with no Neural Engine -- CPU_AND_NE falls back to
# CPU there, so the NE check only has teeth when run on Apple silicon hardware.
# The pose model only matches PyTorch on the GPU: on CPU and Neural Engine its SimCC peaks
# collapse (argmax pins every keypoint to 191.5, 0 -- 265 px off). The app pins .cpuAndGPU
# for that reason. GitHub's macOS runners have no usable GPU, so CI sets PARITY_SKIP=pose and
# this gate only has teeth for pose when run on Apple silicon hardware.
SKIP = {m for m in os.environ.get("PARITY_SKIP", "").split(",") if m}

COMPUTE_UNITS = {
    "detector": [ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU],
    "pose": [ct.ComputeUnit.CPU_AND_GPU],
}


def predict(model, image_hwc_uint8):
    img = Image.fromarray(image_hwc_uint8, mode="RGB")
    return model.predict({"image": img})


def reference_dir(out_dir):
    # Local dev: convert.py just wrote out/reference next to the .mlpackages. In CI,
    # out_dir is the fetch.sh destination (models + manifest only, no reference frames
    # -- those are small enough to commit, see ml/convert/reference/), so fall back to
    # the copy checked into git alongside this script.
    local = out_dir / "reference"
    return local if local.is_dir() else Path(__file__).parent / "reference"


def load(out_dir, manifest, units):
    pkg = out_dir / f"{manifest['id'].rsplit('-', 1)[0]}.mlpackage"
    return ct.models.MLModel(str(pkg), compute_units=units)


def check_detector(out_dir, manifest, units):
    model = load(out_dir, manifest, units)
    ref = reference_dir(out_dir)
    worst = 0.0
    for i in range(manifest["reference_frames"]):
        image = np.load(ref / f"det_input_{i}.npy")
        exp_box = np.load(ref / f"det_box_{i}.npy")
        pred = predict(model, image)
        box = np.asarray(pred["box"]).reshape(4)
        diff = float(np.abs(box - exp_box).max())
        worst = max(worst, diff)
        print(f"  detector frame {i}: max coord diff {diff:.3f} px "
              f"(pytorch={exp_box.tolist()}, coreml={box.tolist()})")
    print(f"  detector worst-case: {worst:.3f} px (tolerance {TOLERANCE_PX} px)")
    return worst <= TOLERANCE_PX


def check_pose(out_dir, manifest, units):
    model = load(out_dir, manifest, units)
    ref = reference_dir(out_dir)
    worst, checked = 0.0, 0
    for i in range(manifest["reference_frames"]):
        image = np.load(ref / f"pose_input_{i}.npy")
        exp_kp = np.load(ref / f"pose_keypoints_{i}.npy")
        exp_sc = np.load(ref / f"pose_scores_{i}.npy")
        pred = predict(model, image)
        kp = np.asarray(pred["keypoints"]).reshape(-1, 2)
        mask = exp_sc > SCORE_THRESHOLD
        if not mask.any():
            # Contract: tolerance only applies to keypoints the model is confident
            # about. Below threshold, SimCC peaks are flat/noisy and fp16 rounding
            # can flip the argmax bin -- real but meaningless drift, not a bug.
            print(f"  pose frame {i}: no keypoints above score {SCORE_THRESHOLD}, skipping")
            continue
        checked += mask.sum()
        disp = np.linalg.norm(kp[mask] - exp_kp[mask], axis=-1)
        frame_worst = float(disp.max())
        worst = max(worst, frame_worst)
        print(f"  pose frame {i}: {mask.sum()} scored keypoints, "
              f"max displacement {frame_worst:.3f} px")
    print(f"  pose worst-case: {worst:.3f} px over {checked} scored keypoints "
          f"(tolerance {TOLERANCE_PX} px)")
    if checked == 0:
        print("  no frame had any keypoint above the score threshold -- gate is vacuous", file=sys.stderr)
        return False
    return worst <= TOLERANCE_PX


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "ml/convert/out")
    manifest = json.loads((out_dir / "manifest.json").read_text())

    ok = True
    for name, check in (("detector", check_detector), ("pose", check_pose)):
        if name in SKIP:
            print(f"{name}: SKIPPED (PARITY_SKIP) -- run this gate on Apple silicon to check it")
            continue
        for units in COMPUTE_UNITS[name]:
            print(f"{name} ({units.name}):")
            ok &= check(out_dir, manifest[name], units)

    if ok:
        print("PASS")
        return
    print("FAIL", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
