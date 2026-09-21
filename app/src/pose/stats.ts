// Rolling perf stats for the frame-processor spike (ADR 0002 Condition 1: fps, per-stage
// latency, redetect rate). Fixed-size arrays in a plain object so state can cross the worklet
// boundary; pushSample mutates it in place, summary() reduces it to display numbers.

export interface StatsState {
  windowSize: number;
  ts: number[]; // frame timestamps, seconds
  detMs: number[];
  poseMs: number[];
  totalMs: number[];
  redetected: number[]; // 1 if the detector ran this frame, else 0
  head: number;
  count: number; // valid samples, <= windowSize
}

export interface FrameSample {
  ts: number;
  poseMs: number;
  detMs?: number; // absent when the detector didn't run this frame
  redetected: boolean;
}

export interface Summary {
  samples: number;
  fps: number;
  detMsP50: number;
  detMsP95: number;
  poseMsP50: number;
  poseMsP95: number;
  totalMsP50: number;
  totalMsP95: number;
  redetectRate: number;
}

export function createStats(windowSize = 120): StatsState {
  'worklet';
  return {
    windowSize,
    ts: new Array(windowSize).fill(0),
    detMs: new Array(windowSize).fill(0),
    poseMs: new Array(windowSize).fill(0),
    totalMs: new Array(windowSize).fill(0),
    redetected: new Array(windowSize).fill(0),
    head: 0,
    count: 0,
  };
}

export function pushSample(state: StatsState, sample: FrameSample): void {
  'worklet';
  const i = state.head;
  const detMs = sample.detMs ?? 0;
  state.ts[i] = sample.ts;
  state.detMs[i] = detMs;
  state.poseMs[i] = sample.poseMs;
  state.totalMs[i] = detMs + sample.poseMs;
  state.redetected[i] = sample.redetected ? 1 : 0;
  state.head = (i + 1) % state.windowSize;
  state.count = Math.min(state.count + 1, state.windowSize);
}

/** Nearest-rank percentile, 0-100, over the (unsorted) valid slice of a rolling buffer. */
export function percentile(values: number[], p: number): number {
  'worklet';
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[idx];
}

export function summary(state: StatsState): Summary {
  'worklet';
  const n = state.count;
  const ts = state.ts.slice(0, n);
  const redetected = state.redetected.slice(0, n);

  let minTs = Infinity, maxTs = -Infinity, redetectSum = 0;
  for (let i = 0; i < n; i++) {
    if (ts[i] < minTs) minTs = ts[i];
    if (ts[i] > maxTs) maxTs = ts[i];
    redetectSum += redetected[i];
  }

  return {
    samples: n,
    fps: n >= 2 && maxTs > minTs ? (n - 1) / (maxTs - minTs) : 0,
    detMsP50: percentile(state.detMs.slice(0, n), 50),
    detMsP95: percentile(state.detMs.slice(0, n), 95),
    poseMsP50: percentile(state.poseMs.slice(0, n), 50),
    poseMsP95: percentile(state.poseMs.slice(0, n), 95),
    totalMsP50: percentile(state.totalMs.slice(0, n), 50),
    totalMsP95: percentile(state.totalMs.slice(0, n), 95),
    redetectRate: n > 0 ? redetectSum / n : 0,
  };
}
