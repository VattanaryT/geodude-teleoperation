from pathlib import Path

import pytest

from geodude_teleoperation.calibration.io import (
    load_arm_calib,
    merge_section,
    parse_float_list,
    save_arm_calib,
    validate_wrench,
)


def test_parse_float_list_ok():
    assert parse_float_list("1,2,3", 3) == [1.0, 2.0, 3.0]


def test_parse_float_list_bad_len():
    with pytest.raises(ValueError):
        parse_float_list("1,2", 3)


def test_validate_wrench_ok():
    assert validate_wrench([0, 0, 1, 0, 0, 0]) == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_validate_wrench_bad_len():
    with pytest.raises(ValueError):
        validate_wrench([0, 0, 1])


def test_merge_preserves_other_sections(tmp_path: Path):
    path = tmp_path / "right.yaml"
    data = {
        "arm": "right",
        "tcp": {"pose": [0.0] * 6, "frame": "tool0", "source": "pendant"},
        "payload": {"mass_kg": 0.49, "cog": [0.019, -0.033, 0.03]},
    }
    save_arm_calib(path, data)
    loaded = load_arm_calib(path)
    merged = merge_section(
        loaded,
        "right",
        "ft_sensor",
        {
            "zeroed_at": "2026-08-06T00:00:00-07:00",
            "bias_wrench": [0.0] * 6,
            "sample_count": 10,
            "note": "",
        },
    )
    save_arm_calib(path, merged)
    again = load_arm_calib(path)
    assert again["tcp"]["pose"] == [0.0] * 6
    assert again["payload"]["mass_kg"] == 0.49
    assert again["ft_sensor"]["sample_count"] == 10
    assert "updated_at" in again
