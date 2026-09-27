"""Generate shared/fixtures/blazepose/*.json: real BlazePose model outputs on the COCO
reference images, plus what runners/blazepose.py makes of them.

The Swift port (app/modules/pose/ios/check/) and ml/tests/test_blazepose_fixtures.py both
assert these. Model outputs are stored sparsely: detector box rows only for anchors that pass
the score threshold, and only the 7x7 heatmap windows refinement reads, placed into a dense
heatmap filled with HEATMAP_FILL. The expectations are computed from that reconstructed input,
so the sparsity is exact rather than approximate.

Needs the models (ml/convert/fetch_blazepose.sh) and the reference images:

    uv run --with ai-edge-litert --with pillow python -m runners.make_blazepose_fixtures \\
        --models ../app/modules/pose/ios/models --images DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from runners import blazepose as bp

SHARED = Path(__file__).resolve().parents[2] / "shared"
IMAGES = ["000000000785", "000000040083", "000000197388"]  # as in ml/convert/convert.py
HEATMAP_FILL = 0.0  # sigmoid 0.5: large enough that a wrong kernel size changes the result


def dense_det(fx):
    boxes = np.zeros((len(bp.ANCHORS), 12))
    for i, row in fx["boxes"].items():
        boxes[int(i)] = row
    return boxes, np.array(fx["scores"])


def dense_heatmap(fx):
    hm = np.full((64, 64, bp.NUM_LANDMARKS), HEATMAP_FILL)
    for ch, (r0, c0, win) in enumerate(fx["windows"]):
        win = np.array(win)
        hm[r0:r0 + win.shape[0], c0:c0 + win.shape[1], ch] = win
    return hm


def expected(fx):
    """What the reference makes of a fixture's inputs. Also used by the test."""
    w, h = fx["image_size"]
    boxes, scores = dense_det(fx["detector"])
    det = bp.best_detection(*bp.decode_detections(boxes, scores))
    rect = bp.alignment_rect(*bp.unletterbox(det[2], w, h)[:2], w, h)
    lm = fx["landmarks"]
    lms = bp.decode_landmarks(np.array(lm["raw"]), np.array(lm["flag"]), dense_heatmap(lm), rect)
    kps, kscores = bp.to_wholebody(lms)
    return {
        "detection": {"score": det[0], "box": det[1].tolist(), "keypoints": det[2].tolist()},
        "rect": list(rect),
        "presence": lms.presence,
        "points": lms.points.tolist(),
        "visibility": lms.visibility.tolist(),
        "wholebody_keypoints": kps.tolist(),
        "wholebody_scores": kscores.tolist(),
        "next_rect": list(bp.rect_from_landmarks(lms, w, h)),
    }


def windows(raw, heatmap):
    """The 7x7 (edge-clipped) window refinement reads for each landmark, as (row0, col0, values)."""
    out, off = [], (bp.HEATMAP_KERNEL - 1) // 2
    for i, (x, y) in enumerate(np.asarray(raw).reshape(-1, 5)[:, :2] / bp.LM_SIZE):
        col, row = int(x * 64), int(y * 64)
        if not (0 <= col < 64 and 0 <= row < 64):
            out.append([0, 0, []])
            continue
        r0, c0 = max(0, row - off), max(0, col - off)
        out.append([r0, c0, heatmap[r0:min(64, row + off + 1), c0:min(64, col + off + 1), i].tolist()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--images", required=True, type=Path)
    args = ap.parse_args()
    models = bp.Models(args.models)
    out_dir = SHARED / "fixtures" / "blazepose"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in IMAGES:
        img = np.array(Image.open(args.images / f"{name}.jpg").convert("RGB"))
        h, w = img.shape[:2]
        boxes, scores = models.detect(img)
        keep = bp.sigmoid(np.clip(scores, -bp.SCORE_CLIP, bp.SCORE_CLIP)) >= bp.MIN_DET_SCORE
        rect = bp.rect_from_detection(boxes, scores, w, h)
        raw, flag, heatmap = models.landmarks(img, rect)
        fx = {
            "image": f"open-mmlab/mmpose v1.3.2 tests/data/coco/{name}.jpg",
            "image_size": [w, h],
            "detector": {
                "scores": [float(s) for s in scores],
                "boxes": {str(i): [float(v) for v in boxes[i]] for i in np.flatnonzero(keep)},
            },
            "landmarks": {"raw": [float(v) for v in raw], "flag": [float(flag[0])],
                          "windows": windows(raw, heatmap)},
        }
        fx["expected"] = expected(fx)
        (out_dir / f"{name}.json").write_text(json.dumps(fx) + "\n")
        print(name, "presence", round(fx["expected"]["presence"], 3), "anchors kept", int(keep.sum()))


if __name__ == "__main__":
    main()
