from pathlib import Path

import pytest

from geodude_teleoperation.calibration.io import (
    load_arm_calib,
    merge_section,
    parse_float_list,
    save_arm_calib,
    validate_payload,
    validate_tcp_pose,
)


def test_parse_float_list_ok():
    assert parse_float_list("1,2,3", 3) == [1.0, 2.0, 3.0]


def test_parse_float_list_bad_len():
    with pytest.raises(ValueError):
        parse_float_list("1,2", 3)


def test_validate_tcp_pose_ok():
    assert validate_tcp_pose([0, 0, 0.1, 0, 0, 0]) == [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]


def test_validate_tcp_pose_bad_len():
    with pytest.raises(ValueError):
        validate_tcp_pose([0, 0, 0.1])


def test_validate_payload_rejects_nonpositive_mass():
    with pytest.raises(ValueError):
        validate_payload(0.0, [0, 0, 0])


def test_validate_payload_ok():
    mass, cog = validate_payload(0.98, [0.01, -0.02, 0.05])
    assert mass == 0.98
    assert cog == [0.01, -0.02, 0.05]


def test_merge_preserves_other_sections(tmp_path: Path):
    path = tmp_path / "right.yaml"
    data = {
        "arm": "right",
        "tcp": {"pose": [0.0] * 6, "frame": "tool0", "source": "manual"},
    }
    save_arm_calib(path, data)
    loaded = load_arm_calib(path)
    merged = merge_section(
        loaded,
        "right",
        "payload",
        {"mass_kg": 0.98, "cog": [0.01, 0.0, 0.05], "source": "manual"},
    )
    save_arm_calib(path, merged)
    again = load_arm_calib(path)
    assert again["tcp"]["pose"] == [0.0] * 6
    assert again["payload"]["mass_kg"] == 0.98
    assert "updated_at" in again
