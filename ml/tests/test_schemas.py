"""Shared JSON files validate against the shared schemas (ADR 0008 C2).

The TypeScript suite validates the same files against the same schemas.
"""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

SHARED = Path(__file__).resolve().parents[2] / "shared"
DEFINITIONS = sorted((SHARED / "definitions").glob("*.json"))


def load(path: Path):
    return json.loads(path.read_text())


def validator(name: str) -> Draft202012Validator:
    schema = load(SHARED / "schemas" / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


def test_definitions_exist():
    assert DEFINITIONS


@pytest.mark.parametrize("path", DEFINITIONS, ids=lambda p: p.name)
def test_definition_valid(path):
    d = load(path)
    validator("exercise-definition").validate(d)
    # Cross-field invariants JSON Schema can't express.
    assert d["id"] == path.stem
    assert d["rep_thresholds"]["enter"] > d["rep_thresholds"]["exit"]
    used = {*d["measured_angle"]["segment_a"], *d["measured_angle"]["segment_b"],
            d["counting_signal"]["keypoint"],
            *d["out_of_plane"]["foot"], *d["out_of_plane"]["shank"]}
    assert used <= set(d["keypoints"]), f"indices not retained: {used - set(d['keypoints'])}"


def test_rule_without_citation_rejected():
    """ADR 0004 C1: enforced by the schema, not by review."""
    d = load(SHARED / "definitions" / "calf-raise.json")
    rule = {"id": "r", "signal": "s", "comparator": "<", "threshold": 1, "message": "m"}
    for bad in (rule, {**rule, "citation": ""}):
        broken = copy.deepcopy(d)
        broken["rules"] = [bad]
        with pytest.raises(ValidationError):
            validator("exercise-definition").validate(broken)


@pytest.mark.parametrize("name", ["keypoint-series", "clip-manifest"])
def test_examples_valid(name):
    validator(name).validate(load(SHARED / "fixtures" / "examples" / f"{name}.json"))
