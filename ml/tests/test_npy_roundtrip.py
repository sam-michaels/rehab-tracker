"""TypeScript writes .npy, Python reads it (ADR 0008 C3). Not optional.

Runs the TS writer via npx, so Node and app/node_modules must be present.
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[2] / "app"
CASES = {
    "f4_1d": ((7,), np.float32),
    "f4_1xN": ((1, 5), np.float32),
    "f4_TxKx3": ((4, 5, 3), np.float32),
    "f8_1d": ((9,), np.float64),
    "f4_empty": ((0, 5, 3), np.float32),
}


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("npy")
    subprocess.run(["npx", "tsx", "scripts/write-npy-fixtures.ts", str(out)], cwd=APP, check=True)
    return out


@pytest.mark.parametrize("name", CASES)
def test_roundtrip(written, name):
    shape, dtype = CASES[name]
    expected = (np.arange(np.prod(shape), dtype=np.float64) * 0.25 - 3).astype(dtype)
    if expected.size > 3:
        expected[1:4] = [np.nan, np.inf, -np.inf]
    got = np.load(written / f"{name}.npy", allow_pickle=False)
    assert got.dtype == np.dtype(dtype).newbyteorder("<")
    assert got.shape == shape
    assert got.flags.c_contiguous
    np.testing.assert_array_equal(got, expected.reshape(shape))
