// Causal measurement pipeline, ported line-for-line from ml/harness/pipeline.py (the reference).
// Per frame: confidence gate -> causal median per keypoint coordinate -> angle -> One Euro, plus
// the out-of-plane check and the rep state machine on the counting signal (ADR 0003).
// shared/fixtures/pipeline/ is asserted by both implementations; disagreement fails the build.
import type { ExerciseDefinition, Keypoint, Segment } from '../types/definition';

export type Status = 'ok' | 'gap' | 'lost';
export type Point = [x: number, y: number];

export interface Calibration {
  foot_shank_ratio: number;
  counting_range: [rest: number, peak: number];
}

export interface Rep {
  start: number;
  end: number;
  peak_fraction: number;
  measurable: boolean;
  angle_min: number | null;
  angle_max: number | null;
}

export interface Measurement {
  angle: (number | null)[];
  status: Status[];
  out_of_plane: boolean[];
  reps: Rep[];
}

/** Casiez et al. 2012. Consumes real timestamps, never an assumed rate (ADR 0003 C1). */
export class OneEuro {
  private x: number | null = null;
  private dx = 0;
  private t = 0;

  constructor(private minCutoff: number, private beta: number, private dCutoff: number) {}

  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x: number, t: number): number {
    if (this.x === null) {
      [this.x, this.dx, this.t] = [x, 0, t];
      return x;
    }
    const dt = t - this.t;
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (x - this.x);
    this.t = t;
    return this.x;
  }
}

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

type Points = Map<Keypoint, Point>;
const at = (p: Points, kp: Keypoint) => p.get(kp)!;

function length([a, b]: Segment, p: Points): number {
  const dx = at(p, b)[0] - at(p, a)[0];
  const dy = at(p, b)[1] - at(p, a)[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/** Unsigned angle in degrees, 0–180, between segment vectors a and b. */
export function segmentAngle(a: Segment, b: Segment, p: Points): number {
  const [ax, ay] = [at(p, a[1])[0] - at(p, a[0])[0], at(p, a[1])[1] - at(p, a[0])[1]];
  const [bx, by] = [at(p, b[1])[0] - at(p, b[0])[0], at(p, b[1])[1] - at(p, b[0])[1]];
  return (Math.atan2(Math.abs(ax * by - ay * bx), ax * bx + ay * by) * 180) / Math.PI;
}

/**
 * Run the live pipeline over a whole series.
 * t: [T] seconds, strictly increasing. keypoints: [T][K][3] (x, y, conf), columns in
 * definition.keypoints order.
 */
export function measure(
  definition: ExerciseDefinition,
  calibration: Calibration,
  t: number[],
  keypoints: number[][][],
): Measurement {
  const f = definition.filter;
  const { segment_a: segA, segment_b: segB } = definition.measured_angle;
  const angleKps = [...segA, ...segB];
  const oop = definition.out_of_plane;
  const oopKps = [...oop.foot, ...oop.shank];
  const heel = definition.counting_signal.keypoint;
  const [rest, peak] = calibration.counting_range;
  const { enter, exit } = definition.rep_thresholds;

  const newFilter = () => new OneEuro(f.one_euro.min_cutoff, f.one_euro.beta, f.one_euro.d_cutoff);
  const buffers = new Map<Keypoint, Point[]>(definition.keypoints.map((kp) => [kp, []]));
  let euro = newFilter();
  let lastValidT: number | null = null;
  let held: number | null = null;
  const out: Measurement = { angle: [], status: [], out_of_plane: [], reps: [] };
  let phase: 'rest' | 'rising' | 'in_rep' = 'rest';
  let rep: { start: number; peak_fraction: number; measurable: boolean; angles: number[] } | null = null;

  for (let i = 0; i < t.length; i++) {
    const ti = t[i];
    // Gate, then causal median over the last median_window accepted samples.
    const smoothed: Points = new Map();
    definition.keypoints.forEach((kp, c) => {
      const [x, y, conf] = keypoints[i][c];
      if (conf >= f.min_confidence) {
        const buf = buffers.get(kp)!;
        buf.push([x, y]);
        if (buf.length > f.median_window) buf.splice(0, buf.length - f.median_window);
        smoothed.set(kp, [median(buf.map((b) => b[0])), median(buf.map((b) => b[1]))]);
      }
    });

    let status: Status;
    if (angleKps.every((kp) => smoothed.has(kp))) {
      held = euro.filter(segmentAngle(segA, segB, smoothed), ti);
      [lastValidT, status] = [ti, 'ok'];
    } else if (lastValidT !== null && ti - lastValidT <= f.max_gap_s) {
      status = 'gap'; // hold; filter state coasts until the next valid sample
    } else {
      // Refuse to measure (ADR 0003 C2). Restart cleanly rather than carry stale state.
      if (lastValidT !== null) {
        [euro, held, lastValidT] = [newFilter(), null, null];
        buffers.forEach((buf) => (buf.length = 0));
      }
      status = 'lost';
    }

    let outOfPlane = false;
    if (oopKps.every((kp) => smoothed.has(kp))) {
      const ratio = length(oop.foot, smoothed) / length(oop.shank, smoothed);
      outOfPlane = Math.abs(ratio / calibration.foot_shank_ratio - 1) > oop.tolerance;
    }

    out.angle.push(status !== 'lost' ? held : null);
    out.status.push(status);
    out.out_of_plane.push(outOfPlane);

    // Hysteresis on the counting signal as a fraction of calibrated range (ADR 0003 C3, C4).
    // A missing counting sample holds the state machine where it is.
    const h = smoothed.get(heel);
    const frac = h ? (h[1] - rest) / (peak - rest) : NaN;
    if (h) {
      if (phase === 'rest' && frac > exit) {
        [phase, rep] = ['rising', { start: i, peak_fraction: frac, measurable: true, angles: [] }];
      } else if (phase === 'rising' && frac <= exit) {
        [phase, rep] = ['rest', null]; // never reached enter: not counted
      } else if (phase === 'rising' && frac >= enter) {
        phase = 'in_rep';
      } else if (phase === 'in_rep' && frac <= exit) {
        const { angles, ...r } = rep!;
        const ok = r.measurable && angles.length > 0;
        out.reps.push({
          ...r,
          end: i,
          measurable: ok,
          angle_min: ok ? Math.min(...angles) : null,
          angle_max: ok ? Math.max(...angles) : null,
        });
        [phase, rep] = ['rest', null];
      }
    }
    if (rep !== null) {
      if (h) rep.peak_fraction = Math.max(rep.peak_fraction, frac);
      if (status === 'lost' || outOfPlane) rep.measurable = false;
      if (status === 'ok') rep.angles.push(held!);
    }
  }
  return out;
}
