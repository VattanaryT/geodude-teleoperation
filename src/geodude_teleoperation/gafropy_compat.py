"""Safe gafropy detection — import can SIGILL on some CPUs (not catchable)."""
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def gafropy_importable() -> bool:
    """True only if ``import gafropy`` succeeds in a child process.

    On some machines the gafropy wheel raises SIGILL on import (illegal CPU
    opcode). That cannot be caught with try/except in-process, so we probe
    once in a subprocess and cache the result.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import gafropy"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def clear_gafropy_importable_cache() -> None:
    """Test helper."""
    gafropy_importable.cache_clear()
