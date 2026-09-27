"""The BlazePose reference reproduces the frozen fixtures (runners/make_blazepose_fixtures.py).

app/modules/pose/ios/check/ asserts the Swift port against the same files.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from runners import blazepose as bp
from runners.make_blazepose_fixtures import expected

SHARED = Path(__file__).resolve().parents[2] / "shared"
FIXTURES = sorted((SHARED / "fixtures" / "blazepose").glob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture(path):
    fx = json.loads(path.read_text())
    got, exp = expected(fx), fx["expected"]
    got.update({f"detection.{k}": v for k, v in got.pop("detection").items()})
    exp = dict(exp) | {f"detection.{k}": v for k, v in exp["detection"].items()}
    del exp["detection"]
    for k in exp:
        np.testing.assert_allclose(np.asarray(got[k], dtype=float), np.asarray(exp[k], dtype=float),
                                   atol=1e-9, err_msg=k)


def test_fixtures_exist():
    assert len(FIXTURES) == 3


def test_anchor_layout():
    # 28x28x2 + 14x14x2 + 7x7x6 (three stride-32 layers merged).
    assert len(bp.ANCHORS) == 2254
    assert bp.ANCHORS[0].tolist() == [0.5 / 28, 0.5 / 28]
    assert bp.ANCHORS[-1].tolist() == [6.5 / 7, 6.5 / 7]


def test_project_inverts_crop_geometry():
    # A rect rotated 90 degrees: the crop's right edge centre lands below the rect centre.
    rect = bp.Rect(0.5, 0.5, 0.2, 0.4, math.pi / 2)
    np.testing.assert_allclose(bp.project([[1.0, 0.5]], rect), [[0.5, 0.7]], atol=1e-12)


def test_alignment_rect_upright_person():
    # Scale point straight above the hip centre -> no rotation; square in pixels, 1.25x of 2x dist.
    r = bp.alignment_rect((0.5, 0.6), (0.5, 0.4), 200, 100)
    assert r.rotation == pytest.approx(0.0)
    assert (r.w * 200, r.h * 100) == (pytest.approx(50.0), pytest.approx(50.0))
