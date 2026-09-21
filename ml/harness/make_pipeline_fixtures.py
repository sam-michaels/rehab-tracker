"""Generate shared/fixtures/pipeline/*.json: synthetic calf-raise inputs + reference outputs.

Run once, inspect the plots, commit. Regenerating overwrites the frozen expectations, so only
do it when the pipeline is changed on purpose (and then port the change to TypeScript).

    uv run python -m harness.make_pipeline_fixtures [--plot DIR]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from harness.pipeline import measure, segment_angle

SHARED = Path(__file__).resolve().parents[2] / "shared"
KNEE, ANKLE, BIG_TOE, SMALL_TOE, HEEL = 13, 15, 17, 18, 19

# Left foot, sagittal view, image coords (y down). The foot pivots about the big toe.
REST = {KNEE: (490, 1100), ANKLE: (470, 1500), BIG_TOE: (640, 1590), SMALL_TOE: (610, 1585),
        HEEL: (440, 1580)}
FULL_DEG = 30.0  # plantarflexion at calibrated peak


def pose(theta_deg, foot_scale=1.0):
    """Keypoints for a heel raise of theta_deg. foot_scale < 1 foreshortens the foot (out of plane)."""
    th = math.radians(theta_deg)
    px, py = REST[BIG_TOE]

    def rot(kp, sx=1.0):
        # Rotate in plane, then compress image-x: turning the foot out of plane foreshortens it
        # horizontally but leaves the heel lift (the counting signal) intact.
        vx, vy = REST[kp][0] - px, REST[kp][1] - py
        return (px + sx * (vx * math.cos(th) - vy * math.sin(th)), py + vx * math.sin(th) + vy * math.cos(th))

    p = {BIG_TOE: (px, py), SMALL_TOE: rot(SMALL_TOE, foot_scale), HEEL: rot(HEEL, foot_scale),
         ANKLE: rot(ANKLE)}
    p[KNEE] = (p[ANKLE][0] + REST[KNEE][0] - REST[ANKLE][0], p[ANKLE][1] + REST[KNEE][1] - REST[ANKLE][1])
    return p


def calibration():
    rest, top = pose(0), pose(FULL_DEG)
    foot = math.dist(rest[HEEL], rest[BIG_TOE])
    shank = math.dist(rest[KNEE], rest[ANKLE])
    return {"foot_shank_ratio": foot / shank, "counting_range": [rest[HEEL][1], top[HEEL][1]]}


def synth(seed, reps, duration):
    """reps: list of dicts {amp, low_conf: (kp, start_s, len_s) | None, spike: bool, foot_scale}."""
    rng = np.random.default_rng(seed)
    t, times = 0.0, []
    while t < duration:
        times.append(round(t, 4))
        t += 1 / 30 + rng.uniform(-0.004, 0.004)
        if rng.random() < 0.03:  # dropped frame
            t += 1 / 30
    order = [KNEE, ANKLE, BIG_TOE, SMALL_TOE, HEEL]
    frames, spiked = [], set()
    for ti in times:
        theta, scale, conf_drop, spike = 0.0, 1.0, set(), False
        for r, spec in enumerate(reps):
            t0 = 1.0 + r * 3.5
            if t0 <= ti <= t0 + 2.5:
                u = (ti - t0) / 2.5
                theta = spec["amp"] * (0.5 - 0.5 * math.cos(2 * math.pi * u))
                scale = spec.get("foot_scale", 1.0)
                if lc := spec.get("low_conf"):
                    kp, s, n = lc
                    if t0 + s <= ti < t0 + s + n:
                        conf_drop.add(kp)
                if spec.get("spike") and u >= 0.5 and r not in spiked:  # exactly one frame
                    spike = True
                    spiked.add(r)
        p = pose(theta, scale)
        row = []
        for kp in order:
            x, y = p[kp]
            x, y = x + rng.normal(0, 1.5), y + rng.normal(0, 1.5)
            if spike and kp == BIG_TOE:
                x += 80
            c = 0.1 if kp in conf_drop else float(np.clip(rng.normal(0.85, 0.05), 0.5, 1.0))
            row.append([round(x, 3), round(y, 3), round(c, 3)])
        frames.append(row)
    return times, frames


CASES = {
    "clean": dict(seed=1, duration=19.0, reps=[{"amp": FULL_DEG}] * 5),
    "gaps": dict(seed=2, duration=15.5, reps=[
        {"amp": FULL_DEG},
        {"amp": FULL_DEG, "low_conf": (BIG_TOE, 1.1, 0.15)},  # short gap: bridged, still measurable
        {"amp": FULL_DEG, "low_conf": (BIG_TOE, 1.0, 0.5)},   # long gap: refused
        {"amp": FULL_DEG},
    ]),
    "edge": dict(seed=3, duration=15.5, reps=[
        {"amp": 17.5},                         # shallow: never reaches enter, not counted
        {"amp": FULL_DEG, "spike": True},      # one-frame toe outlier: median rejects it
        {"amp": FULL_DEG, "foot_scale": 0.7},  # foot rotated out of plane: refused
        {"amp": FULL_DEG},
    ]),
}


def plot(name, definition, t, frames, cal, result, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {kp: i for i, kp in enumerate(definition["keypoints"])}
    raw = [segment_angle(definition["measured_angle"]["segment_a"], definition["measured_angle"]["segment_b"],
                         {kp: f[col[kp]] for kp in col}) for f in frames]
    rest, peak = cal["counting_range"]
    frac = [(f[col[HEEL]][1] - rest) / (peak - rest) for f in frames]
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    a1.plot(t, raw, lw=0.6, color="0.6", label="raw angle")
    a1.plot(t, [np.nan if a is None else a for a in result["angle"]], lw=1.2, label="measured (One Euro)")
    for i, s in enumerate(result["status"]):
        if s != "ok" or result["out_of_plane"][i]:
            a1.axvspan(t[i] - 0.017, t[i] + 0.017, color={"gap": "gold", "lost": "red"}.get(s, "purple"),
                       alpha=0.3, lw=0)
    a1.set_ylabel("angle (deg)")
    a1.legend(loc="upper right", fontsize=8)
    a2.plot(t, frac, lw=0.8)
    for y in definition["rep_thresholds"].values():
        a2.axhline(y, ls="--", color="0.5", lw=0.8)
    for r in result["reps"]:
        a2.axvspan(t[r["start"]], t[r["end"]], color="green" if r["measurable"] else "red", alpha=0.15)
    a2.set_ylabel("heel fraction of range")
    a2.set_xlabel("t (s)   yellow=gap, red=lost, purple=out of plane; spans = counted reps")
    fig.suptitle(f"{name}: {len(result['reps'])} reps, "
                 f"{sum(r['measurable'] for r in result['reps'])} measurable")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", type=Path, help="write a PNG per case into this directory")
    args = ap.parse_args()
    definition = json.loads((SHARED / "definitions" / "calf-raise.json").read_text())
    cal = calibration()
    out = SHARED / "fixtures" / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in CASES.items():
        t, frames = synth(**spec)
        result = measure(definition, cal, t, frames)
        fixture = {"definition": definition["id"], "definition_version": definition["version"],
                   "calibration": cal, "input": {"t": t, "keypoints": frames}, "expected": result}
        (out / f"{name}.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
        print(name, len(t), "frames,", [(r["start"], r["end"], r["measurable"]) for r in result["reps"]])
        if args.plot:
            args.plot.mkdir(parents=True, exist_ok=True)
            plot(name, definition, t, frames, cal, result, args.plot)


if __name__ == "__main__":
    main()
