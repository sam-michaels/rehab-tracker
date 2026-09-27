"""BlazePose (MediaPipe Pose Landmarker models) without the MediaPipe SDK.

The Tasks SDK reports usage metrics to Google (ADR 0002, amendment 2026-09-25), so the app
runs the two Apache-2.0 .tflite models from pose_landmarker_full.task itself. This is the
reference implementation of the pre/post-processing around them. The Swift port
(app/modules/pose/ios/BlazePoseGeometry.swift) is checked against shared/fixtures/blazepose/.
It is also the MediaPipe runner for the harness (ADR 0002 C4).

Every constant below is taken from MediaPipe's own graphs and calculators:
modules/pose_detection/pose_detection_cpu.pbtxt, modules/pose_landmark/*.pbtxt, and
calculators/{tflite/ssd_anchors, tensor/tensors_to_detections, util/non_max_suppression,
util/alignment_points_to_rects, util/detections_to_rects, util/rect_transformation,
util/refine_landmarks_from_heatmap, util/landmark_projection}_calculator.cc.

Geometry is numpy only. The interpreter (ai-edge-litert) is imported only by `Models`, so the
test suite doesn't need it.

Coordinates: normalized [0,1] in the upright image, origin top-left, y down. A Rect is
(cx, cy, w, h, rotation) with w and h normalized to image width and height, and rotation in
radians, the same as MediaPipe's NormalizedRect.
"""
import math
from typing import NamedTuple

import numpy as np

DET_SIZE = 224
LM_SIZE = 256
NUM_LANDMARKS = 39  # 33 pose + 2 auxiliary (ROI alignment) + 4 unused
MIN_DET_SCORE = 0.5
NMS_IOU = 0.3
SCORE_CLIP = 100.0
ROI_SCALE = 1.25
POSE_PRESENCE = 0.5
HEATMAP_KERNEL = 7
HEATMAP_MIN_CONF = 0.5

# MediaPipe landmark -> COCO-WholeBody slot. Slots MediaPipe has no equivalent for
# (small toes 18/21, face 23-90, hands 91-132) are left at score 0.
MP_TO_WHOLEBODY = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,  # nose, eyes, ears
    11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,  # shoulders, elbows, wrists
    23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,  # hips, knees, ankles
    31: 17, 29: 19, 32: 20, 30: 22,  # foot index -> big toe, heel -> heel
}
NUM_WHOLEBODY = 133


class Rect(NamedTuple):
    cx: float
    cy: float
    w: float
    h: float
    rotation: float


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -100, 100)))


def anchors():
    """SSD anchor centres, [2254, 2]. fixed_anchor_size, so every w and h is 1."""
    strides = [8, 16, 32, 32, 32]
    out, layer = [], 0
    while layer < len(strides):
        # Layers sharing a stride are merged. Each contributes aspect 1.0 plus the
        # interpolated-scale anchor, so 2 anchors per cell per layer.
        n = 0
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            n += 2
            last += 1
        fm = math.ceil(DET_SIZE / strides[layer])
        for y in range(fm):
            for x in range(fm):
                out.extend([((x + 0.5) / fm, (y + 0.5) / fm)] * n)
        layer = last
    return np.array(out, dtype=np.float64)


ANCHORS = anchors()


def decode_detections(raw_boxes, raw_scores):
    """raw_boxes [2254, 12], raw_scores [2254] -> (scores [N], boxes [N, 4] xyxy, kps [N, 4, 2]).

    Normalized to the letterboxed 224 input. Only boxes with score >= MIN_DET_SCORE.
    """
    scores = sigmoid(np.clip(raw_scores, -SCORE_CLIP, SCORE_CLIP))
    keep = scores >= MIN_DET_SCORE
    raw, a = raw_boxes[keep], ANCHORS[keep]
    cx = raw[:, 0] / DET_SIZE + a[:, 0]
    cy = raw[:, 1] / DET_SIZE + a[:, 1]
    w = raw[:, 2] / DET_SIZE
    h = raw[:, 3] / DET_SIZE
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kps = raw[:, 4:12].reshape(-1, 4, 2) / DET_SIZE + a[:, None, :]
    return scores[keep], boxes, kps


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_detection(scores, boxes, kps):
    """First output of MediaPipe's weighted NMS: the top box, score-averaged with every box
    overlapping it by IoU > NMS_IOU (itself included). None if nothing passed the threshold.
    Single person (num_poses = 1), so only the first NMS output is ever used."""
    if len(scores) == 0:
        return None
    top = int(np.argmax(scores))
    group = [i for i in range(len(scores)) if iou(boxes[i], boxes[top]) > NMS_IOU]
    wts = scores[group][:, None]
    box = (boxes[group] * wts).sum(0) / wts.sum()
    kp = (kps[group].reshape(len(group), -1) * wts).sum(0).reshape(4, 2) / wts.sum()
    return float(scores[top]), box, kp


def letterbox(width, height):
    """(pad_x, pad_y): the padding on each side, as a fraction of the square detector input."""
    side = max(width, height)
    return (1 - width / side) / 2, (1 - height / side) / 2


def unletterbox(points, width, height):
    px, py = letterbox(width, height)
    p = np.asarray(points, dtype=np.float64)
    return np.stack([(p[..., 0] - px) / (1 - 2 * px), (p[..., 1] - py) / (1 - 2 * py)], axis=-1)


def normalize_radians(a):
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def alignment_rect(p0, p1, width, height):
    """AlignmentPointsRects (start 0, end 1, target 90 deg) then RectTransformation
    (scale 1.25, square_long). p0 is the hip centre, p1 the scale/rotation point."""
    x0, y0 = p0[0] * width, p0[1] * height
    x1, y1 = p1[0] * width, p1[1] * height
    size = 2 * math.hypot(x1 - x0, y1 - y0) * ROI_SCALE  # square in pixels
    rotation = normalize_radians(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
    return Rect(p0[0], p0[1], size / width, size / height, rotation)


def project(points, rect):
    """Crop-normalized points [N, 2] -> image-normalized, through the rect's rotation."""
    p = np.asarray(points, dtype=np.float64) - 0.5
    c, s = math.cos(rect.rotation), math.sin(rect.rotation)
    x = (c * p[:, 0] - s * p[:, 1]) * rect.w + rect.cx
    y = (s * p[:, 0] + c * p[:, 1]) * rect.h + rect.cy
    return np.stack([x, y], axis=1)


def refine_from_heatmap(points, heatmap):
    """points [39, 2] crop-normalized; heatmap [64, 64, 39] raw logits (HWC)."""
    hh, hw, _ = heatmap.shape
    out = np.array(points, dtype=np.float64)
    off = (HEATMAP_KERNEL - 1) // 2
    for i, (x, y) in enumerate(out):
        col, row = int(x * hw), int(y * hh)  # C++ float -> int truncates toward zero
        if not (0 <= col < hw and 0 <= row < hh):
            continue
        r0, r1 = max(0, row - off), min(hh, row + off + 1)
        c0, c1 = max(0, col - off), min(hw, col + off + 1)
        conf = sigmoid(heatmap[r0:r1, c0:c1, i].astype(np.float64))
        total = conf.sum()
        if conf.max() >= HEATMAP_MIN_CONF and total > 0:
            rows, cols = np.mgrid[r0:r1, c0:c1]
            out[i] = [(cols * conf).sum() / hw / total, (rows * conf).sum() / hh / total]
    return out


class Landmarks(NamedTuple):
    presence: float  # pose flag, sigmoid'd by the model
    points: np.ndarray  # [39, 2] image-normalized
    visibility: np.ndarray  # [39]


def decode_landmarks(raw, flag, heatmap, rect):
    """raw [195], flag [1] (already a probability), heatmap [64, 64, 39] -> Landmarks."""
    lm = np.asarray(raw, dtype=np.float64).reshape(NUM_LANDMARKS, 5)
    crop = refine_from_heatmap(lm[:, :2] / LM_SIZE, heatmap)
    return Landmarks(float(np.ravel(flag)[0]), project(crop, rect), sigmoid(lm[:, 3]))


def to_wholebody(lms):
    """Landmarks -> (keypoints [133*2] flat x,y, scores [133]) in COCO-WholeBody slots."""
    kps, scores = np.zeros(NUM_WHOLEBODY * 2), np.zeros(NUM_WHOLEBODY)
    for mp, wb in MP_TO_WHOLEBODY.items():
        kps[2 * wb: 2 * wb + 2] = lms.points[mp]
        scores[wb] = lms.visibility[mp]
    return kps, scores


def crop(image, rect, size):
    """Bilinear sample of the rotated rect into [size, size, 3] float in [0, 1], zero outside.
    Inverse of `project`: output pixel centre (u+.5)/size maps through the same rotation."""
    h, w = image.shape[:2]
    t = (np.arange(size) + 0.5) / size - 0.5
    a, b = np.meshgrid(t, t)
    c, s = math.cos(rect.rotation), math.sin(rect.rotation)
    x = ((c * a - s * b) * rect.w + rect.cx) * w - 0.5
    y = ((s * a + c * b) * rect.h + rect.cy) * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    img = image.astype(np.float64) / 255.0

    def px(yy, xx):
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        v = img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
        return v * ok[..., None]

    return ((px(y0, x0) * (1 - fx) + px(y0, x0 + 1) * fx) * (1 - fy)
            + (px(y0 + 1, x0) * (1 - fx) + px(y0 + 1, x0 + 1) * fx) * fy)


def detector_rect(width, height):
    """The whole image, letterboxed to a square (keep_aspect_ratio)."""
    side = max(width, height)
    return Rect(0.5, 0.5, side / width, side / height, 0.0)


class Models:
    """The two .tflite files from pose_landmarker_full.task (ml/convert/fetch_blazepose.sh)."""

    def __init__(self, model_dir):
        from ai_edge_litert.interpreter import Interpreter

        self.det = Interpreter(model_path=f"{model_dir}/pose_detector.tflite")
        self.lm = Interpreter(model_path=f"{model_dir}/pose_landmarks_detector.tflite")
        for it in (self.det, self.lm):
            it.allocate_tensors()

    @staticmethod
    def _run(it, x):
        it.set_tensor(it.get_input_details()[0]["index"], x[None].astype(np.float32))
        it.invoke()
        # Output order by name (Identity, Identity_1, ...), matching MediaPipe's tensor split.
        outs = sorted(it.get_output_details(), key=lambda d: d["name"])
        return [it.get_tensor(d["index"])[0] for d in outs]

    def detect(self, image):
        """-> (raw_boxes [2254, 12], raw_scores [2254])."""
        h, w = image.shape[:2]
        boxes, scores = self._run(self.det, crop(image, detector_rect(w, h), DET_SIZE) * 2 - 1)
        return boxes, scores[:, 0]

    def landmarks(self, image, rect):
        """-> (raw [195], flag [1], heatmap [64, 64, 39])."""
        raw, flag, _seg, heatmap, _world = self._run(self.lm, crop(image, rect, LM_SIZE))
        return raw, flag, heatmap


def rect_from_detection(raw_boxes, raw_scores, width, height):
    det = best_detection(*decode_detections(raw_boxes, raw_scores))
    if det is None:
        return None
    kp = unletterbox(det[2], width, height)
    return alignment_rect(kp[0], kp[1], width, height)


def rect_from_landmarks(lms, width, height):
    """Next frame's ROI from auxiliary landmarks 33 (centre) and 34 (scale/rotation)."""
    return alignment_rect(lms.points[33], lms.points[34], width, height)


class Tracker:
    """Detector on the first frame and after a lost pose; landmark-derived ROI otherwise
    (ADR 0002 C2), as MediaPipe's pose_landmark_cpu graph does."""

    def __init__(self, models):
        self.models, self.rect = models, None

    def __call__(self, image, force_detect=False):
        h, w = image.shape[:2]
        if force_detect or self.rect is None:
            self.rect = rect_from_detection(*self.models.detect(image), w, h)
            if self.rect is None:
                return None
        lms = decode_landmarks(*self.models.landmarks(image, self.rect), self.rect)
        self.rect = rect_from_landmarks(lms, w, h) if lms.presence >= POSE_PRESENCE else None
        return lms if lms.presence >= POSE_PRESENCE else None
