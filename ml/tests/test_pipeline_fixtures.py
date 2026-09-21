"""The reference pipeline reproduces the frozen fixtures (ADR 0005 C4).

app/src/pipeline/measure.test.ts asserts the TypeScript port against the same files.
"""
import json
from pathlib import Path

import pytest

from harness.pipeline import measure

SHARED = Path(__file__).resolve().parents[2] / "shared"
FIXTURES = sorted((SHARED / "fixtures" / "pipeline").glob("*.json"))
TOL_DEG = 1e-6


def approx_list(xs):
    return [None if x is None else pytest.approx(x, abs=TOL_DEG) for x in xs]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture(path):
    fx = json.loads(path.read_text())
    definition = json.loads((SHARED / "definitions" / f"{fx['definition']}.json").read_text())
    assert definition["version"] == fx["definition_version"], "definition changed: regenerate fixtures"
    got = measure(definition, fx["calibration"], fx["input"]["t"], fx["input"]["keypoints"])
    exp = fx["expected"]
    assert got["status"] == exp["status"]
    assert got["out_of_plane"] == exp["out_of_plane"]
    assert got["angle"] == approx_list(exp["angle"])
    assert len(got["reps"]) == len(exp["reps"])
    for g, e in zip(got["reps"], exp["reps"]):
        assert g == {k: pytest.approx(v, abs=TOL_DEG) if isinstance(v, float) else v for k, v in e.items()}


def test_fixtures_exist():
    assert {p.stem for p in FIXTURES} == {"clean", "gaps", "edge"}
