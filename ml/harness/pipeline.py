"""Causal measurement pipeline: the reference the TypeScript port must match (ADR 0005 C4).

Per frame: confidence gate -> causal median per keypoint coordinate -> angle -> One Euro,
plus the out-of-plane check and the rep state machine on the counting signal (ADR 0003).
Deliberately plain Python loops so it reads line-for-line against app/src/pipeline/measure.ts.
Any change here must be mirrored there; shared/fixtures/pipeline/ catches drift.
"""
import math

OK, GAP, LOST = "ok", "gap", "lost"


class OneEuro:
    """Casiez et al. 2012. Consumes real timestamps, never an assumed rate (ADR 0003 C1)."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def length(seg, p):
    dx, dy = p[seg[1]][0] - p[seg[0]][0], p[seg[1]][1] - p[seg[0]][1]
    return math.sqrt(dx * dx + dy * dy)


def segment_angle(a, b, p):
    """Unsigned angle in degrees, 0-180, between segment vectors a and b."""
    ax, ay = p[a[1]][0] - p[a[0]][0], p[a[1]][1] - p[a[0]][1]
    bx, by = p[b[1]][0] - p[b[0]][0], p[b[1]][1] - p[b[0]][1]
    return math.degrees(math.atan2(abs(ax * by - ay * bx), ax * bx + ay * by))


def measure(definition, calibration, t, keypoints):
    """Run the live pipeline over a whole series.

    t: [T] seconds, strictly increasing. keypoints: [T][K][3] (x, y, conf), columns in
    definition["keypoints"] order. calibration: {"foot_shank_ratio", "counting_range": [rest, peak]}.
    Returns per-frame angle/status/out_of_plane and the list of counted reps.
    """
    f = definition["filter"]
    col = {kp: i for i, kp in enumerate(definition["keypoints"])}
    seg_a, seg_b = definition["measured_angle"]["segment_a"], definition["measured_angle"]["segment_b"]
    angle_kps = {*seg_a, *seg_b}
    oop = definition["out_of_plane"]
    heel = definition["counting_signal"]["keypoint"]
    rest, peak = calibration["counting_range"]
    enter, exit_ = definition["rep_thresholds"]["enter"], definition["rep_thresholds"]["exit"]

    new_filter = lambda: OneEuro(**f["one_euro"])  # noqa: E731
    buffers = {kp: [] for kp in definition["keypoints"]}  # recent gated (x, y) per keypoint
    euro, last_valid_t, held = new_filter(), None, None
    out = {"angle": [], "status": [], "out_of_plane": [], "reps": []}
    phase, rep = "rest", None

    for i, ti in enumerate(t):
        # Gate, then causal median over the last median_window accepted samples.
        smoothed = {}
        for kp, buf in buffers.items():
            x, y, c = keypoints[i][col[kp]]
            if c >= f["min_confidence"]:
                buf.append((x, y))
                del buf[:-f["median_window"]]
                smoothed[kp] = (median([b[0] for b in buf]), median([b[1] for b in buf]))

        if angle_kps <= smoothed.keys():
            held = euro(segment_angle(seg_a, seg_b, smoothed), ti)
            last_valid_t, status = ti, OK
        elif last_valid_t is not None and ti - last_valid_t <= f["max_gap_s"]:
            status = GAP  # hold; filter state coasts until the next valid sample
        else:
            # Refuse to measure (ADR 0003 C2). Restart cleanly rather than carry stale state.
            if last_valid_t is not None:
                euro, held, last_valid_t = new_filter(), None, None
                for buf in buffers.values():
                    buf.clear()
            status = LOST

        out_of_plane = False
        if {*oop["foot"], *oop["shank"]} <= smoothed.keys():
            ratio = length(oop["foot"], smoothed) / length(oop["shank"], smoothed)
            out_of_plane = abs(ratio / calibration["foot_shank_ratio"] - 1) > oop["tolerance"]

        out["angle"].append(held if status != LOST else None)
        out["status"].append(status)
        out["out_of_plane"].append(out_of_plane)

        # Hysteresis on the counting signal as a fraction of calibrated range (ADR 0003 C3, C4).
        # A missing counting sample holds the state machine where it is.
        if heel in smoothed:
            frac = (smoothed[heel][1] - rest) / (peak - rest)
            if phase == "rest" and frac > exit_:
                phase, rep = "rising", {"start": i, "peak_fraction": frac, "measurable": True,
                                        "angles": []}
            elif phase == "rising" and frac <= exit_:
                phase, rep = "rest", None  # never reached enter: not counted
            elif phase == "rising" and frac >= enter:
                phase = "in_rep"
            elif phase == "in_rep" and frac <= exit_:
                angles = rep.pop("angles")
                ok = rep["measurable"] and bool(angles)
                out["reps"].append({**rep, "end": i, "measurable": ok,
                                    "angle_min": min(angles) if ok else None,
                                    "angle_max": max(angles) if ok else None})
                phase, rep = "rest", None
        if rep is not None:
            if heel in smoothed:
                rep["peak_fraction"] = max(rep["peak_fraction"], frac)
            if status == LOST or out_of_plane:
                rep["measurable"] = False
            if status == OK:
                rep["angles"].append(held)

    return out
