#!/usr/bin/env bash
# Fetch converted model artifacts from a GitHub Release, verify against manifest.json,
# unpack into app/modules/pose/ios/models/ (ADR 0008 C6: artifacts live in Releases, never git).
#
#   ml/convert/fetch.sh <release-tag>
set -euo pipefail

tag="${1:?usage: fetch.sh <release-tag>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="$repo_root/app/modules/pose/ios/models"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

gh release download "$tag" --dir "$tmp" --pattern '*.zip' --pattern 'manifest.json'

manifest="$tmp/manifest.json"
for key in detector pose; do
  file=$(python3 -c "import json;print(json.load(open('$manifest'))['$key']['artifact_file'])")
  expected=$(python3 -c "import json;print(json.load(open('$manifest'))['$key']['artifact_sha256'])")
  actual=$(shasum -a 256 "$tmp/$file" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    echo "sha256 mismatch for $file: expected $expected, got $actual" >&2
    exit 1
  fi
  echo "$file: sha256 OK"
done

mkdir -p "$dest"
for zip in "$tmp"/*.zip; do
  unzip -oq "$zip" -d "$dest"
done
cp "$manifest" "$dest/manifest.json"
echo "unpacked into $dest"
