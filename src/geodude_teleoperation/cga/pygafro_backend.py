"""Lazy ``pygafro`` import (CGA backend for in-contact DTC)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def pygafro_available() -> bool:
    try:
        import pygafro  # noqa: F401

        return True
    except Exception:
        return False


def require_pygafro():
    """Import ``pygafro`` or raise ``ImportError`` with a clear message."""
    try:
        import pygafro as ga

        return ga
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "in_contact actuation=dtc / dtc_pure requires pygafro "
            "(pip install pygafro). PyPI gafropy SIGILLs on some CPUs; "
            "use pygafro instead."
        ) from exc
