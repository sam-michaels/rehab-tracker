import { createStats, percentile, pushSample, summary } from './stats';

describe('percentile', () => {
  test('nearest-rank p50/p95', () => {
    const values = [5, 3, 1, 4, 2]; // 1..5, unsorted
    expect(percentile(values, 50)).toBe(3);
    expect(percentile(values, 95)).toBe(5);
    expect(percentile(values, 0)).toBe(1);
  });

  test('empty input is 0', () => {
    expect(percentile([], 50)).toBe(0);
  });
});

describe('createStats / summary', () => {
  test('empty window summarizes to zeros', () => {
    const s = summary(createStats(5));
    expect(s).toEqual({
      samples: 0, fps: 0,
      detMsP50: 0, detMsP95: 0,
      poseMsP50: 0, poseMsP95: 0,
      totalMsP50: 0, totalMsP95: 0,
      redetectRate: 0,
    });
  });

  test('fps, percentiles and redetect rate over a filled window', () => {
    const state = createStats(5);
    for (let v = 1; v <= 5; v++) {
      pushSample(state, { ts: v, detMs: v, poseMs: v * 2, redetected: v % 2 === 1 });
    }
    const s = summary(state);
    expect(s.samples).toBe(5);
    expect(s.fps).toBeCloseTo(1, 6); // 4 intervals over 4 seconds (ts 1..5)
    expect(s.detMsP50).toBe(3);
    expect(s.detMsP95).toBe(5);
    expect(s.poseMsP50).toBe(6); // poseMs = 2x detMs
    expect(s.redetectRate).toBeCloseTo(3 / 5, 6); // v=1,3,5 redetected
  });

  test('rolling window drops the oldest samples once full', () => {
    const state = createStats(5);
    for (let v = 1; v <= 8; v++) {
      pushSample(state, { ts: v, detMs: v, poseMs: v, redetected: v % 2 === 1 });
    }
    const s = summary(state);
    // Only the last 5 pushes (4..8) should remain; 1..3 were overwritten.
    expect(s.samples).toBe(5);
    expect(s.detMsP50).toBe(6); // median of [4,5,6,7,8]
    expect(s.detMsP95).toBe(8);
    expect(s.fps).toBeCloseTo(1, 6); // (5-1) / (8-4)
    expect(s.redetectRate).toBeCloseTo(2 / 5, 6); // v=5,7 redetected among 4..8
  });

  test('detMs defaults to 0 when the detector did not run', () => {
    const state = createStats(3);
    pushSample(state, { ts: 0, poseMs: 10, redetected: false });
    const s = summary(state);
    expect(s.detMsP50).toBe(0);
    expect(s.totalMsP50).toBe(10);
  });
});
