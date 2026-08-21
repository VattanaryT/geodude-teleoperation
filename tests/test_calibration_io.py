from pathlib import Path

from geodude_teleoperation.calibration.io import load_arm_calib, merge_section, save_arm_calib


def test_merge_preserves_other_sections(tmp_path: Path):
    path = tmp_path / "right.yaml"
    save_arm_calib(
        path,
        {
            "arm": "right",
            "tcp": {"pose": [0.0] * 6, "frame": "tool0", "source": "pendant"},
            "payload": {"mass_kg": 0.49, "cog": [0.019, -0.033, 0.03]},
        },
    )
    merged = merge_section(
        load_arm_calib(path),
        "right",
        "ft_sensor",
        {"bias_wrench": [0.0] * 6, "sample_count": 10, "note": ""},
    )
    save_arm_calib(path, merged)
    again = load_arm_calib(path)
    assert again["tcp"]["pose"] == [0.0] * 6
    assert again["payload"]["mass_kg"] == 0.49
    assert again["ft_sensor"]["sample_count"] == 10
