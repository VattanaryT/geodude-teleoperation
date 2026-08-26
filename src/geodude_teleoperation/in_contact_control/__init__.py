"""In-contact control loops ported from ``exp/in-contact-dtc``.

``force_pid`` is the press loop that has the most hardware hours behind it: PID
on ``Fe = Fd - Fs`` along a surface normal, with a 5 Hz LPF and deadband on the
measurement plus real-time auto-stabilize (oscillation detector, spike guard,
overshoot retract, soft-approach cap, force-rate damping).

The files here are kept as close to verbatim copies of the DTC branch as
possible so they stay diffable against it; see ``types.py`` for the one
deliberate deviation.
"""

from geodude_teleoperation.in_contact_control.force_pid import (
    ForcePIDConfig,
    ForcePIDController,
)
from geodude_teleoperation.in_contact_control.types import TwistCommand

__all__ = ["ForcePIDConfig", "ForcePIDController", "TwistCommand"]
