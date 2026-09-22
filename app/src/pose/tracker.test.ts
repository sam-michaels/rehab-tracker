import { nextRoi, personPresent, roiFromKeypoints, shouldRedetect, type PoseResult } from './tracker';

// 4 confident keypoints forming a 0.2x0.2 box centered at (0.5, 0.5); one low-confidence
// keypoint far outside it that must be excluded from the bbox.
const scores5 = [0.9, 0.9, 0.9, 0.9, 0.1];
const keypoints5 = [
  0.4, 0.5, // left
  0.6, 0.5, // right
  0.5, 0.4, // top
  0.5, 0.6, // bottom
  0.9, 0.9, // excluded (low score)
];

describe('roiFromKeypoints', () => {
  test('pads 1.25x and expands to 3:4 aspect in a square frame', () => {
    const roi = roiFromKeypoints(keypoints5, scores5, 0.3, 1);
    expect(roi).not.toBeNull();
    const [x, y, w, h] = roi!;
    expect(x).toBeCloseTo(0.375, 6);
    expect(y).toBeCloseTo(1 / 3, 6);
    expect(w).toBeCloseTo(0.25, 6);
    expect(h).toBeCloseTo(1 / 3, 6);
    expect(w / h).toBeCloseTo(3 / 4, 6); // pose model input is 192x256
  });

  test('aspect math accounts for a non-square frame (width/height in pixels)', () => {
    const roi = roiFromKeypoints(keypoints5, scores5, 0.3, 16 / 9);
    expect(roi).not.toBeNull();
    const [x, y, w, h] = roi!;
    expect(x).toBeCloseTo(0.375, 6);
    expect(w).toBeCloseTo(0.25, 6);
    expect(y).toBeCloseTo(0.203704, 5);
    expect(h).toBeCloseTo(0.592593, 5);
  });

  test('excludes keypoints below minConfidence', () => {
    // Without exclusion the far point at (0.9, 0.9) would blow the bbox open.
    const roi = roiFromKeypoints(keypoints5, scores5, 0.3, 1)!;
    expect(roi[0] + roi[2]).toBeLessThan(0.9);
  });

  test('returns null with fewer than minKeypoints confident keypoints', () => {
    const scores = [0.9, 0.9, 0.1, 0.1, 0.1]; // only 2 confident, default minKeypoints is 4
    expect(roiFromKeypoints(keypoints5, scores, 0.3, 1)).toBeNull();
  });

  test('minKeypoints is a parameter', () => {
    const scores = [0.9, 0.9, 0.1, 0.1, 0.1];
    expect(roiFromKeypoints(keypoints5, scores, 0.3, 1, 2)).not.toBeNull();
  });
});

// 23 body+foot keypoints (indices 0-22) clustered in a small box near the frame center.
function bodyKeypoints(cx: number, cy: number, half: number): number[] {
  const pts: number[] = [];
  for (let i = 0; i < 23; i++) {
    const dx = i % 2 === 0 ? -half : half;
    const dy = i % 4 < 2 ? -half : half;
    pts.push(cx + dx, cy + dy);
  }
  return pts;
}
const N = 23;

describe('shouldRedetect', () => {
  test('true on confidence collapse (mean body score below threshold)', () => {
    const scores = new Array(N).fill(0.1); // mean 0.1 < default 0.3
    const kps = bodyKeypoints(0.5, 0.5, 0.05);
    expect(shouldRedetect(kps, scores, 1)).toBe(true);
  });

  test('false when confident and the derived ROI stays in frame', () => {
    const scores = new Array(N).fill(0.6);
    const kps = bodyKeypoints(0.5, 0.5, 0.05);
    expect(shouldRedetect(kps, scores, 1)).toBe(false);
  });

  test('true when the derived ROI would leave the frame mostly', () => {
    const scores = new Array(N).fill(0.6); // confident, but tracking has drifted off-frame
    const kps = bodyKeypoints(-0.45, -0.45, 0.05);
    expect(shouldRedetect(kps, scores, 1)).toBe(true);
  });
});

describe('nextRoi', () => {
  const good: PoseResult = { keypoints: bodyKeypoints(0.5, 0.5, 0.05), scores: new Array(N).fill(0.6), poseMs: 10 };
  const collapsed: PoseResult = { keypoints: bodyKeypoints(0.5, 0.5, 0.05), scores: new Array(N).fill(0.1), poseMs: 10 };

  test('mode A always re-detects', () => {
    expect(nextRoi('A', good, 1)).toBeNull();
    expect(nextRoi('A', collapsed, 1)).toBeNull();
  });

  test('mode B tracks when confident and in frame', () => {
    const roi = nextRoi('B', good, 1);
    expect(roi).not.toBeNull();
    expect(roi).toEqual(roiFromKeypoints(good.keypoints, good.scores, 0.3, 1));
  });

  test('mode B re-detects on confidence collapse', () => {
    expect(nextRoi('B', collapsed, 1)).toBeNull();
  });
});

describe('personPresent', () => {
  test('thresholds the mean of body+foot scores 0-22, ignoring face/hands', () => {
    const scores = new Array(133).fill(0.9);
    expect(personPresent(scores)).toBe(true);
    scores.fill(0.1, 0, 23);
    expect(personPresent(scores)).toBe(false); // high face/hand scores don't count
  });
});

test('roiFromKeypoints ignores face/hand keypoints (index > 22)', () => {
  const scores = new Array(133).fill(0);
  const keypoints = new Array(266).fill(0.5);
  scores.fill(0.9, 0, 4); // 4 confident body points forming a 0.2x0.2 box
  keypoints.splice(0, 8, 0.4, 0.4, 0.6, 0.4, 0.4, 0.6, 0.6, 0.6);
  const bodyOnly = roiFromKeypoints(keypoints, scores, 0.3, 1);
  scores[100] = 0.9; // a confident hand point far away must not move the ROI
  keypoints[200] = 0.99;
  keypoints[201] = 0.99;
  expect(roiFromKeypoints(keypoints, scores, 0.3, 1)).toEqual(bodyOnly);
});
