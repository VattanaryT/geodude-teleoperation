"""gafropy_compat probe tests."""
import subprocess

from geodude_teleoperation import gafropy_compat


def test_gafropy_importable_cached(monkeypatch):
    gafropy_compat.clear_gafropy_importable_cache()
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1

        class R:
            returncode = 1

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert gafropy_compat.gafropy_importable() is False
    assert gafropy_compat.gafropy_importable() is False
    assert calls["n"] == 1


def test_gafropy_importable_true(monkeypatch):
    gafropy_compat.clear_gafropy_importable_cache()

    def fake_run(*args, **kwargs):
        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        assert gafropy_compat.gafropy_importable() is True
    finally:
        gafropy_compat.clear_gafropy_importable_cache()
