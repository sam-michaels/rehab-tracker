#!/usr/bin/env bash
# Fetch MediaPipe's pose landmarker models (Apache-2.0) straight from Google, verify the pinned
# sha256, and unpack the two .tflite files the app runs into app/modules/pose/ios/models/.
# Not the MediaPipe SDK: the app runs these models itself (ADR 0002, amendment 2026-09-25).
#
#   ml/convert/fetch_blazepose.sh
set -euo pipefail

# Pinned version path ("1"), not "latest". Model card: ml/MODEL_CARD.md.
url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
expected="5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="$repo_root/app/modules/pose/ios/models"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -sfL "$url" -o "$tmp/pose_landmarker_full.task"
actual=$(shasum -a 256 "$tmp/pose_landmarker_full.task" | cut -d' ' -f1)
if [ "$actual" != "$expected" ]; then
  echo "sha256 mismatch for pose_landmarker_full.task: expected $expected, got $actual" >&2
  exit 1
fi
echo "pose_landmarker_full.task: sha256 OK"

mkdir -p "$dest"
unzip -oq "$tmp/pose_landmarker_full.task" pose_detector.tflite pose_landmarks_detector.tflite -d "$dest"
echo "unpacked pose_detector.tflite, pose_landmarks_detector.tflite into $dest"
