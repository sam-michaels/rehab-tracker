// ROI tracking for the frame-processor plugin contract (ADR 0002 Condition 2: amortize the
// detector). Mode A re-detects every frame; mode B derives the next ROI from the previous
// frame's keypoints and only re-detects on confidence collapse or the ROI leaving the frame.
// Pure, worklet-safe: plain data in and out, no closures over module state.

/** [x, y, w, h] normalized to the frame buffer, or null meaning "run the detector". */
export type Roi = [x: number, y: number, w: number, h: number];
export type Mode = 'A' | 'B';

/** Plugin output per the fixed contract (COCO-WholeBody 133 keypoints, x0,y0,x1,y1,...). */
export interface PoseResult {
  keypoints: number[];
  scores: number[];
  detBox?: number[];
  detScore?: number;
  detMs?: number;
  poseMs: number;
}

// COCO-WholeBody layout: 0-16 body, 17-22 feet, 23-90 face, 91-132 hands.
const BODY_END = 22;
// Pose model input is 192x256 (w x h) -> 3:4.
const POSE_ASPECT = 3 / 4;
const ROI_PADDING = 1.25;
// Reuse the definition's filter.min_confidence concept (shared/definitions/calf-raise.json).
const DEFAULT_MIN_CONFIDENCE = 0.3;

/**
 * Bbox over keypoints with score >= minConfidence, padded 1.25x and expanded to the pose
 * model's 3:4 (w:h) input aspect around its center. Aspect is computed in pixels via
 * frameAspect (buffer width/height) since the frame is not square. Null if fewer than
 * minKeypoints are confident.
 */
export function roiFromKeypoints(
  keypoints: number[],
  scores: number[],
  minConfidence = DEFAULT_MIN_CONFIDENCE,
  frameAspect = 1,
  minKeypoints = 4,
): Roi | null {
  'worklet';
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, n = 0;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] < minConfidence) continue;
    const x = keypoints[2 * i];
    const y = keypoints[2 * i + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    n++;
  }
  if (n < minKeypoints) return null;

  // Pixel-proportional units (H = 1, W = frameAspect) so padding/aspect-fit are isotropic.
  const px0 = minX * frameAspect, px1 = maxX * frameAspect;
  const cx = (px0 + px1) / 2, cy = (minY + maxY) / 2;
  let w = (px1 - px0) * ROI_PADDING;
  let h = (maxY - minY) * ROI_PADDING;
  if (w / h < POSE_ASPECT) w = h * POSE_ASPECT;
  else h = w / POSE_ASPECT;

  return [(cx - w / 2) / frameAspect, cy - h / 2, w / frameAspect, h];
}

/**
 * True on confidence collapse (mean score of body+foot keypoints 0-22 below
 * meanScoreThreshold) or when the ROI derived from these keypoints would leave the frame
 * mostly (in-frame overlap fraction below minInFrameFraction).
 */
export function shouldRedetect(
  keypoints: number[],
  scores: number[],
  frameAspect = 1,
  minConfidence = DEFAULT_MIN_CONFIDENCE,
  meanScoreThreshold = DEFAULT_MIN_CONFIDENCE,
  minInFrameFraction = 0.5,
): boolean {
  'worklet';
  let sum = 0;
  for (let i = 0; i <= BODY_END; i++) sum += scores[i];
  if (sum / (BODY_END + 1) < meanScoreThreshold) return true;

  const roi = roiFromKeypoints(keypoints, scores, minConfidence, frameAspect);
  if (roi === null) return true;

  const [x, y, w, h] = roi;
  const overlapW = Math.max(0, Math.min(x + w, 1) - Math.max(x, 0));
  const overlapH = Math.max(0, Math.min(y + h, 1) - Math.max(y, 0));
  const area = w * h;
  const inFrameFraction = area > 0 ? (overlapW * overlapH) / area : 0;
  return inFrameFraction < minInFrameFraction;
}

/** Mode A always re-detects; mode B tracks unless shouldRedetect fires. */
export function nextRoi(
  mode: Mode,
  result: PoseResult,
  frameAspect = 1,
  minConfidence = DEFAULT_MIN_CONFIDENCE,
): Roi | null {
  'worklet';
  if (mode === 'A') return null;
  if (shouldRedetect(result.keypoints, result.scores, frameAspect, minConfidence)) return null;
  return roiFromKeypoints(result.keypoints, result.scores, minConfidence, frameAspect);
}
