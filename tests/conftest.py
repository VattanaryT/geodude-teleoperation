"""Make `pytest` work from a bare checkout.

The repo declares no pytest configuration (no `[tool.pytest.ini_options]` in
pyproject.toml, no pytest.ini/setup.cfg/tox.ini) and the package lives under
`src/`, so `pytest` from the repo root cannot import `geodude_teleoperation`
unless the caller remembers `PYTHONPATH=src` or has done a `pip install -e .`.
This only adds `src/` when the package is *not* already importable, so an
installed copy still wins.
"""
import importlib.util
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

if importlib.util.find_spec("geodude_teleoperation") is None and _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
