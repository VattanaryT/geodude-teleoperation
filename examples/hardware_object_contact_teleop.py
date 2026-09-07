"""Hardware object-contact teleop for geodude.

Free space: gafropy speedJ + SpaceMouse (approach mode).

Contact (``--contact-backend speedj``): same pose-spring speedJ plant as free
space.
The mode button (``--mode-button``, default 1) latches force-hold until it is
pressed again — no auto-drop to free-space approach. The press PID overlays
tool-Z; it owns Fd and nothing else does.
Stick XY still field-follows (``project()`` rewrites R; stick Z ignored).
It must not switch to line admittance.

Three switches exist for the contact lift-off fixes, so a hardware session can
bisect which one matters. ``--press-loop legacy --no-free-along-normal
--no-snap-follow-to-surface`` restores the previous behavior exactly:

* ``--press-loop dtc`` uses the ``in_contact_control`` ForcePIDController
  (LPF + deadband on the measurement, oscillation/spike/overshoot handling)
  instead of the bare ``clip(kp*(Fd-Fs))`` on fz alone.
* free-along-normal projects the pose spring's velocity off the surface normal
  while in contact. ``stiffness[5,5]=0`` alone only frees the body Z of the
  pose error, and field-follow slerps tool Z behind the normal on purpose, so
  the spring otherwise keeps a component along the normal and lifts the tool.
* snap-follow-to-surface takes the along-normal component of the commanded
  point from ``vf.project()`` so the next query stays on the mesh.

``--contact-backend force_mode`` is UR hybrid force/position (not admittance).
``--contact-backend dtc`` (the default): startup align under speedJ, then one
handoff to Direct Torque for all teleop (free space + contact); contact is
Z-overlay only.

**Defaults are the measured-best free-space DTC config**, so the bare command
is now equivalent to::

    --contact-backend dtc --dtc-law wrench --dtc-lambda --dtc-friction-comp
    --dtc-fn-lin 3.0 --dtc-fn-ang 3.0 --dtc-damping-ratio 0.3
    --dtc-damping-ratio-ang 0.1 --dtc-damping-max 12 --dtc-err-clip-lin 0.05
    --max-lead-m 0.05 --dtc-tau-max 50 50 50 6 6 30 --align-duration 4

To go back to the speedJ plant end-to-end, pass ``--contact-backend speedj
--max-lead-m 0.005``: the 50 mm lead is a wrench-law force ceiling and is far
too much slack for a velocity carrot chased by a position servo.

**DTC compliance hold** — push-return spring physics on top of the normal flow
(Enter → align → SpaceMouse field-follow teleop). ``--dtc-compliance-hold`` is
now equivalent to::

    --dtc-tau-max 140 140 140 25 25 25 --no-dtc-hold-at-tick --dtc-lambda
    --dtc-fn-lin 3.0 --dtc-damping-ratio 0.318 --dtc-damping-ratio-ang 0.15
    --dtc-joint-hold-kp 0 --dtc-joint-hold-kd 0

``hold_at_tick=false`` is the whole mode: the idle hold otherwise re-bases the
pose target onto the live EE at 200 Hz, which erases an external push before the
spring can answer it. Turning it off leaves the target where it was commanded,
so a push builds spring error and the arm returns on release. SpaceMouse still
drives field-follow; only the idle re-base is off, and only once DTC is the live
actuator (``follow_rebase_on_idle`` -- the speedJ align segment keeps it, since
the speedJ spring has no input gate and would otherwise sag).

The 0.318/0.15 are the ratios that *run*, not a 10:1 K/D rule: ``_cap_zeta``
clamps both blades at ``D <= --dtc-damping-max``. See COMPLIANCE_HOLD_ZETA_LIN.

``--dtc-push-test`` is the bench A-side of that: same gains, but no startup
align, no input device and no field-follow, so the arm holds the pose it is
already in and a hand push is the only excitation::

    --dtc-push-test --no-viz

The gain chain, per blade, under ``--dtc-lambda``::

    omega_n = 2*pi*fn                     rad/s   --dtc-fn-lin / --dtc-fn-ang
    K       = omega_n**2                  1/s^2
    D       = 2*zeta*omega_n = 4*pi*zeta*fn  1/s  --dtc-damping-ratio[-ang]
    D_eff   = min(D, dmax)                1/s     --dtc-damping-max  (_cap_zeta)
    K/D     = omega_n/(2*zeta) = pi*fn/zeta   1/s
    W       = Lambda @ (K*clip(e) - D*(J @ qdot))     N and N.m
    tau     = clip(J.T @ W, +-tau_max)

Note ``clip`` is on the *spring* term only (``--dtc-err-clip-lin/-rot``); the
damping term is unbounded, so ``tau_max`` is the only thing bounding a limit
cycle's amplitude.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import signal
import select
import socket
import sys
import termios
import threading
import time
import tty

import gafropy as ga
import numpy as np
import yaml
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geodude_teleoperation import (
    AdmittanceTeleoperationController,
    ContactTeleoperationController,
    KeyboardTeleop,
    OrientationField,
    SpaceMouseTeleop,
    apply_vention_heights,
    compute_resting_orientation,
    is_idle_twist,
    load_object_mesh,
    load_points,
    table_placement_translation,
)
from geodude_teleoperation.ee_viz import (
    add_ghost_chamfer_visual,
    add_right_chamfer_visual,
    chamfer_world_from_wrist3,
    ghost_flange_world,
    hide_gripper_visuals,
    live_tcp_xyz_from_flange,
    matrix_from_motor,
    motor_from_matrix,
    StartupAlign,
    SimTcpProbe,
)
from geodude_teleoperation.field_follow import (
    FieldFollow,
    drive_field_follow,
    field_tcp_to_ee,
    format_follow_lag,
)
from geodude_teleoperation.path_session import PathSession
from geodude_teleoperation.force_mode_contact import (
    ForceModeContactSession,
    reopen_gafropy,
    shutdown_gafropy_for_handoff,
)
from geodude_teleoperation.laser_frames import (
    DEFAULT_LASER_OFFSET,
    DEFAULT_TAG0_OFFSET,
    is_laser_left_base_object,
    load_t_vention,
    place_laser_object,
)
from geodude_teleoperation.speedj_plot import LIVE_FIGSIZE
from geodude_teleoperation.mesh_fixture import (
    clip_mesh_to_scan_points,
    orient_surface_toward,
    reconstruct_mesh_from_points,
)
from geodude_teleoperation.spacemouse import (
    format_spacemouse_paths,
    list_spacemouse_paths,
    resolve_spacemouse_path,
    verify_spacemouse_connection,
)
from geodude_teleoperation.tcp_force import (
    connect_tcp_force_reader,
    force_from_tcp_force,
    fz_from_tcp_force,
)
from geodude_teleoperation.direct_torque import print_dtc_rate_verification
from geodude_teleoperation.torque_backend import (
    LAMBDA_MAX_LIN_KG,
    DEFAULT_DTC_HZ,
    DEFAULT_I_MAX,
    DEFAULT_JOINT_HOLD_KD,
    DEFAULT_JOINT_HOLD_KP,
    DEFAULT_STICTION_ASSIST_NM,
    DEFAULT_STICTION_MAX_NM,
    DEFAULT_KI,
    DEFAULT_KV,
    DEFAULT_RAIL_OSC_FLIPS,
    DEFAULT_RAIL_OSC_HOLD_S,
    RAIL_OSC_WINDOW_S,
    DirectTorqueBackend,
)
from geodude_teleoperation.virtual_fixture import FieldAlignPreview, workstation_field_z_arrows

DEFAULT_OBJECT = str(ROOT / "assets" / "laser_profilometer" / "merged.ply")
MODEL = str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml")
DEFAULT_HARDWARE_CONFIG = str(ROOT / "assets" / "robots" / "geodude" / "geodude_hardware.yaml")

DEFAULT_VENTION_HEIGHT = 0.51

# --dtc-tau-max default for --dtc-law wrench, which is now this example's
# default backend. Deliberately NOT torque_backend.DEFAULT_TAU_MAX: that tuple
# is the velocity law's (30/30/30/10/10/10) and stays where it is. Under the
# wrench law the shoulders have to make real lateral newtons through J.T while
# the wrists must not -- a railed wrist is the fault that ended in
# `C283A2 wrist 1a, robot is moving when in stop mode`.
#
# j3/j4 were pinned at 1.0 N.m for that reason and it was too low to move them:
# on dtc_20260829_232839.csv (bare defaults) they clipped on **32.9% and 45.3%**
# of ticks, which is below UR5e wrist breakaway -- the two joints that carry a
# width traverse's rotation could not turn at all, so the field's orientation
# was never reached and the linear blade shoved the misaligned tool instead.
# 6.0 N.m cleared it (dtc_20260829_233332.csv: 1.0% and 0.1%) without any wrist
# rail. The C283A2 fault it was guarding against was a *publish-side tau chop*,
# fixed separately in the streamer's decay and --dtc-stale-tau-ms.
#
# 6.0 does still bind once the angular blade can do its job -- at --dtc-fn-ang
# 3.0 (dtc_20260830_002759.csv) j3 clipped on 4.2% of ticks and 14.7% of the
# third quartile -- and it was briefly raised to 10.0 for that reason. It went
# straight back: on dtc_20260830_005006.csv the arm ran a 24.5 Hz limit cycle
# (see resolve_wrench_zeta) and j4 sat on the 10 N.m rail for 18.9% of ticks,
# which is the *amplitude* of that cycle. During an oscillation tau_max is the
# only thing bounding how hard the wrist shakes, so it goes back to the last
# value that was measured on a stable arm and is re-raised only after the
# oscillation is confirmed gone. One variable at a time.
WRENCH_TAU_MAX = (50.0, 50.0, 50.0, 6.0, 6.0, 30.0)


# Measured operational-space linear inertia and tool-frame breakaway friction on
# this rig, for the startup force-ceiling check only -- never used in control.
# Lambda: dtc_20260902_154438.csv settled at 9.66 N over 4.39 mm with K_lin
# 355.3, so |W|/(K*e) = 6.19 kg; dtc_20260903_130725.csv reads 6.4 kg the same
# way. Breakaway: the force the arm holds while stalled, 17.3-18.3 N across
# dtc_20260903_130302/130725/131641.csv. Both are pose-dependent -- these are
# the middle of the measured range, good enough for an order-of-magnitude gate.
DEFAULT_PRESS_KP = 0.5
LAMBDA_LIN_NOMINAL_KG = 6.2
BREAKAWAY_LIN_NOMINAL_N = 18.0

# --dtc-rearm-wrench-s default. The wrench-law counterpart to --rearm-slerp-s,
# which only ever ramped the commanded *rotation*. On dtc_20260830_115039.csv
# the input_active 0->1 edge stepped |W_lin| 9.25 -> 38.30 N and |W_ang|
# 1.09 -> 10.35 N.m inside one 8 ms tick: while the stick is quiet the idle
# hold pins the target on the live pose, so err_lin is ~0, and the first active
# tick hands back the full field-follow carrot (up to --max-lead-m of lead).
# 0.15 s is ~30 ticks at the 200 Hz default, which turns a 38 N step into ~1 N
# per tick and is well inside the operator's own reaction time. 0 off.
DEFAULT_REARM_WRENCH_S = 0.15

# --dtc-law wrench damping ratios and the no-Lambda angular frequency, resolved
# by resolve_wrench_zeta / resolve_wrench_fn_ang.
WRENCH_ZETA_LIN = 0.3
# 0.10 let the rig's ~15 Hz structural mode ring on 19.4% of q-dot power
# (dtc_20260830_151553.csv); 0.15 cut that to 2.5-3.3% with every release
# still decaying (dtc_20260830_152659/154729.csv) -- cheap, hardware-A/B'd,
# never a source of the delayed-damper pathology (that mode sits at 40 deg
# of phase here, well under the 90 deg line). Promoted to the default
# 2026-08-31. It is not sufficient alone at every --dtc-fn-lin/--dtc-damping-max
# pairing -- dtc_20260831_180840.csv still read 30.9% at 14.42 Hz at these
# bare defaults (fn_lin 3, dmax 12) with 0.15 -- because the mode is a fixed
# mechanical resonance excited by the operator, not a control pole: it decays
# to 0.000 rad/s on every release, and its frequency does not move across
# fn 3/4, floor/no-floor, or zeta 0.10/0.15 runs. That rules out both "gain
# too high" (a real control pole's frequency tracks sqrt(K)) and "damping
# too low, closed loop is unstable" (an unstable loop does not need the
# stick to keep ringing). Raising zeta_ang further is bounded by
# --dtc-damping-max, same as the linear side.
WRENCH_ZETA_ANG = 0.15
WRENCH_FN_ANG_NO_LAMBDA = 1.5

# --- --dtc-compliance-hold: push-return spring gains. -----------------------
#
# **These are the ratios that actually RUN, not the ones that were asked for.**
# The push test this mode was validated on (2026-09-01, commit d90b1cd) was
# launched with ``--dtc-damping-ratio 0.94 --dtc-damping-ratio-ang 0.94`` on
# the reasoning that ``zeta = pi*fn/10`` gives ``K/D = 10`` at fn 3 Hz. Neither
# 0.94 ever reached the loop: ``_cap_zeta`` clamps both blades to
# ``D <= WRENCH_DAMPING_MAX_1_S`` = 12.0 1/s, which at fn 3.0 is
# ``zeta = 0.318`` and ``K/D = 29.6``. So the 10:1 rule of thumb has never been
# on this rig, nothing about the push test's behavior is evidence for it, and
# the request/run gap was invisible except as one line of yellow at startup --
# the same trap CLAUDE.md records for the 0.4 -> 0.239 experiment that silently
# ran *less* damping than the run it was meant to exceed. Store what runs.
COMPLIANCE_HOLD_ZETA_LIN = 0.318

# The angular blade keeps the teleop default, because the push test could not
# test it. In compliance-hold-as-a-push-test (d90b1cd) ``startup_align`` was
# off, no SpaceMouse was opened and field-follow never ran, so
# ``set_hold_rotation`` was never called: ``_hold_rotation`` stayed None,
# ``hold_pose_at`` snapped target := live pose, and the angular error was
# identically zero for the whole run. **Whatever the operator ran on the
# angular blade, the push test did not exercise it.** Under teleop (bf916d1)
# field-follow commands the field orientation continuously, so those gains load
# for the first time -- same command line, different physics.
#
# The run remembered as optimal was ``--dtc-damping-ratio-ang 0.15
# --dtc-fn-ang 2.0``, i.e. already this zeta with a *detuned* fn (K_ang 157.9
# against 355.3 at fn_ang 3.0, D_ang 3.77 against 5.65). fn_ang 2.0 sits in the
# regime resolve_wrench_fn_ang documents as worse on conformity -- 1.5 held
# tool Z 4.17 deg / 22.75 deg p95 off the field normal where 3.0 held
# 1.33 / 5.28 -- so it is a variable for the A/B, not a value to inherit.
COMPLIANCE_HOLD_ZETA_ANG = WRENCH_ZETA_ANG

# ~90% of the UR5e's rated joint torques (150/150/150/28/28/28), which is the
# operator's stated intent for this mode: a push test wants the arm to resist a
# hand rather than yield at a cap. Note this deliberately raises j3/j4 by 4.2x
# over WRENCH_TAU_MAX, and what that buys is *authority*, not stability.
#
# Read it as an amplitude setting, because that is what it is. tau_max is the
# only thing bounding the damping half of the wrench (`--dtc-err-clip-*` clips
# the K*e term; `-Lambda*D*v` is unclipped, so during a limit cycle the pose
# error is ~0 and 100% of the command comes from the unbounded term). And
# WRENCH_TAU_MAX records what a raised wrist cap cost the last time: 6.0 -> 10.0
# was tried and reverted because j4 sat on the 10 N.m rail for 18.9% of ticks
# during the 24.5 Hz limit cycle.
#
# In a *push test* this is free -- nothing commands rotation, so j3/j4 carry
# almost no task torque and never approach the rail, which is why 30 30 30 read
# as harmless there and was the wrist cap in the run remembered as optimal.
# Under teleop they carry the field. So if a run rings on j3/j4, this is the
# first number to bring down, and `clip=N@jK` in the 1 Hz line names the joint.
# 25 rather than 30 also keeps every wrist inside the 28 N.m rating; the j5 30
# in WRENCH_TAU_MAX predates this and is over it.
COMPLIANCE_HOLD_TAU_MAX = (140.0, 140.0, 140.0, 25.0, 25.0, 25.0)

# Ceiling on the angular fn that resolve_wrench_fn_ang *derives* from
# --dtc-fn-lin under Lambda. An explicit --dtc-fn-ang still wins.
#
# Under Lambda, "fn is the closed-loop frequency" holds only where Lambda is
# the arm's real apparent inertia. On the roll axis it is not: Lambda there is
# ~0.001 kg.m^2 and LAMBDA_MIN_ANG_KGM2 clamps it up to 0.02, so the real
# natural frequency on that axis is inflated by sqrt(0.02/0.002) ~ 3.2x. At
# fn 3.0 that is a true ~9.5 Hz loop, which this rig holds; at fn 4.0 it is
# ~12.6 Hz, and dtc_20260830_132604.csv duly put 33.7% of the qdot power at
# **14.43 Hz**, peaked on j0/j4, with j3 clipping 9.8% and rising to 20.4% by
# the last quartile and wrist 3 stick-slipping at 0.215 rad/s with the stick at
# rest. The same run's *linear* numbers all improved (err_lin 8.85 -> 5.81 mm,
# conformity 0.73 -> 0.51 deg), which is the tell: the benefit was linear and
# the damage was angular, so the two blades must stop moving together above
# this. The floor buys the roll axis authority, and it costs it bandwidth.
WRENCH_FN_ANG_MAX_LAMBDA = 3.0

# Ceiling on the pose spring's damping coefficient D = 2*zeta*omega_n, in 1/s,
# under --dtc-lambda only (without Lambda D is N.s/m and N.m.s/rad and this
# number means nothing). This is NOT the rigid-body delayed-damper bound
# (D*T ~ 1) that CLAUDE.md carries -- that bound is far too permissive, because
# the frequency that actually limits D is a *structural* one. See
# resolve_wrench_zeta. 12.0 admits the two ratios measured stable on this rig
# (D_lin = 11.3 at zeta 0.3 / fn 3.0, D_ang = 3.8 at zeta 0.1) and refuses the
# 26.4 that limit-cycled.
WRENCH_DAMPING_MAX_1_S = 12.0
# ...and the control rate it was measured at. The cap is a DELAY bound wearing
# a gain's clothes, so it has to move when the delay does. See
# resolve_wrench_damping_max.
WRENCH_DAMPING_MAX_REF_HZ = 125.0
WRENCH_DAMPING_MAX_SCALE_LIMIT = 2.0


def resolve_wrench_damping_max(hz, damping_max=None):
    """The D ceiling. 12.0 at every rate unless --dtc-damping-max says otherwise.

    **This used to scale as 12.0 * hz/125 and that was wrong, on hardware.**
    The argument was sound as far as it went: what the cap bounds is the
    negative damping a late `-D*v` pumps into a structural mode, at 125 Hz the
    12 ms delay put this rig's 24.5 Hz mode at 108 deg (a source), and at
    200 Hz the same mode sits at 68 deg (still a sink), so the specific failure
    12.0 was built to prevent cannot happen at 200 Hz.

    What it missed is that `f_90 = 1/(4*T)` is not a fixed target the modes sit
    below -- it is a *sweep line*, and doubling the rate sweeps it up into
    structure that was previously out of reach. dtc_20260830_141034.csv ran
    D_lin = 19.1 at 200 Hz and limit-cycled at **32.1 Hz**, which is f_90 for
    this loop to within a bin: tau alternated +50/-50 N.m on j1 AND j2 (the
    50 N.m shoulder and elbow, which rail in no other run) with the stick quiet
    and the pose error at 0.0 mm, and the arm swung into the workstation. The
    old failure did not reappear at 24.5 Hz; a new one appeared at 32.
    Raising the ceiling with the rate bought the loop a fresh mode to
    destabilise every time.

    So the number stays where it was measured. The empirical bracket at 200 Hz
    is now D_lin 15.08 clean (dtc_20260830_135043.csv) and 19.1 catastrophic,
    and 12.0 is the last value with a quiet log on both sides of it. Going
    above it is an explicit `--dtc-damping-max` and wants
    `RailOscillationGuard` armed, a short run and a look at the rail section of
    analyze_dtc_log.py before anyone leans on the stick.

    Historical: the scaling read ``12.0 * min(max(hz/125, 1), 2)``, i.e. 19.2
    at the 200 Hz default. ``hz`` is kept in the signature so the caller does
    not change and a future rate-aware ceiling has somewhere to go, but it must
    come with logs at that rate, not with an extrapolation.
    """
    if damping_max is not None:
        return float(damping_max)
    return WRENCH_DAMPING_MAX_1_S


def _cap_zeta(zeta, fn, damping_max, blade, warn):
    """Clamp one blade's zeta so D = 2*zeta*omega_n stays under ``damping_max``."""
    if fn is None or float(fn) <= 0.0 or damping_max is None or damping_max <= 0:
        return float(zeta)
    omega_n = 2.0 * math.pi * float(fn)
    d = 2.0 * float(zeta) * omega_n
    if d <= float(damping_max):
        # --dtc-damping-max is a CEILING, not a damping setter. Raising it past
        # the D the ratio already asks for changes nothing at all, and a run
        # launched to "raise damping" that way is a no-op that reads as a
        # successful experiment: dtc_20260905_170736 (dmax 12) and _171033
        # (dmax 16) have byte-identical D_lin=11.99 and zeta_lin=0.318. Say so,
        # so the header is not the only place it shows.
        if warn is not None and float(damping_max) > d * 1.05:
            warn(f"\033[93mdtc: --dtc-damping-max {float(damping_max):.1f} is above "
                 f"the {blade} D this run actually uses ({d:.2f} 1/s at "
                 f"zeta={float(zeta):.3f}, fn={float(fn):g} Hz), so it clamps "
                 f"NOTHING. To raise damping use --dtc-damping-ratio"
                 f"{'-ang' if blade == 'angular' else ''}; the cap only bites "
                 f"once 2*zeta*omega_n exceeds it.\033[0m")
        return float(zeta)
    capped = float(damping_max) / (2.0 * omega_n)
    if warn is not None:
        warn(f"\033[93mdtc: {blade} damping clamped, zeta {float(zeta):.3f} -> "
             f"{capped:.3f} (D {d:.1f} -> {damping_max:.1f} 1/s at fn={float(fn):g} Hz). "
             f"Velocity feedback through this loop's delay is NEGATIVE damping "
             f"at the rig's 24.5 Hz mode; raise --dtc-damping-max only with a "
             f"log that shows no limit cycle.\033[0m")
    return capped


def resolve_wrench_zeta(use_lambda, zeta_lin=None, zeta_ang=None,
                        fn_lin=None, fn_ang=None, damping_max=None,
                        warn=None):
    """(linear, angular) damping ratios for the pose spring under the wrench law.

    ``D = 2*sqrt(K)*zeta`` is critical damping for **M = 1**. Without Lambda
    that is false (~3 kg on the linear blades, ~0.05 kg.m^2 on the wrist) and
    0.3/0.1 are those two masses folded into the gain by hand. Under Lambda the
    EE really is a unit mass in every direction, so the same numbers read as a
    37%- and a 73%-overshoot second-order system and look badly under-damped.

    **They are not, and raising them is what broke the arm.** Set to 0.7/0.7 on
    that reasoning, dtc_20260830_005006.csv ran a **24.5 Hz limit cycle** with
    the stick at rest: tau and qdot on every joint peak at 24.5 Hz in the FFT,
    j4 sat on its rail 18.9% of ticks and reached 1.0 rad/s of joint velocity
    that nobody commanded, while the *tracking* numbers looked excellent
    (conformity 0.59 deg median, re-arm edges 2 and under 2 N). The rigid-body
    bound in CLAUDE.md said this was safe -- ``D*T`` was only 0.21 at the
    median tick -- and it was wrong, because the frequency that limits D is not
    the 3 Hz rigid-body mode:

    - A damper is ``-D*v`` applied one loop delay late. At frequency ``f`` that
      is a phase lag of ``2*pi*f*T_d``, and past 90 deg the term is **negative**
      damping. This loop holds 8.06 ms with a republish thread behind it, so
      ``T_d`` is ~1.5 ticks = ~12 ms and the lag reaches 90 deg at ~20 Hz.
    - The rig has a lightly damped structural mode at **24.5 Hz** (measured: it
      is the FFT peak of both tau and qdot on all six joints, and it is common
      to every joint, which a control-law mode would not be). There the lag is
      ~108 deg, so ``D*cos(108 deg)`` = -0.31*D of damping is being pumped
      straight into it. At D_ang 3.8 (zeta 0.1) that is -1.2 and the mode's own
      damping absorbs it; at 26.4 (zeta 0.7) it is -8.2 and it does not.

    So zeta under Lambda is bounded by ``D <= damping_max``, not by overshoot,
    and the defaults go back to the two values measured stable on this rig.
    This also makes a --dtc-fn-lin sweep safe: D = 2*zeta*omega_n grows with
    fn, so raising fn at fixed zeta walks straight back into the cycle, and the
    cap takes zeta down instead. Without Lambda nothing is capped -- D is in
    physical units there and ``damping_max`` does not apply.
    """
    lin = WRENCH_ZETA_LIN if zeta_lin is None else float(zeta_lin)
    ang = WRENCH_ZETA_ANG if zeta_ang is None else float(zeta_ang)
    if not use_lambda:
        return lin, ang
    if damping_max is None:
        damping_max = WRENCH_DAMPING_MAX_1_S
    return (_cap_zeta(lin, fn_lin, damping_max, "linear", warn),
            _cap_zeta(ang, fn_ang, damping_max, "angular", warn))


def resolve_wrench_fn_ang(use_lambda, fn_lin, fn_ang=None):
    """Angular natural frequency (Hz), defaulting to the linear one under Lambda.

    Lambda carries the two blades' different apparent M exactly, so under it
    there is no reason left for the two frequencies to differ -- the 20:1 split
    existed only to paper over that difference. Confirmed on hardware: the
    bare-default run (dtc_20260830_002254.csv) held tool Z 4.17 deg off the
    field normal at the median and 22.75 deg at p95, and re-running it with
    --dtc-fn-ang 3.0 (= fn_lin) cut that to 1.33 / 5.28 deg, with err_ang
    median 14.25 -> 6.95 deg. Without Lambda the split is doing real work and
    this leaves it alone.

    Tracking the linear blade stops at WRENCH_FN_ANG_MAX_LAMBDA, though. The
    equality argument above is that Lambda carries the real apparent M, and on
    the roll axis it does not -- LAMBDA_MIN_ANG_KGM2 clamps that axis up off
    its true ~0.001 kg.m^2 on purpose, so its real closed-loop frequency runs
    ~3.2x the fn asked for. The cap is on what this function *derives*; an
    explicit --dtc-fn-ang is the operator saying they mean it, and passes
    through.
    """
    if fn_ang is not None:
        return float(fn_ang)
    if not use_lambda:
        return WRENCH_FN_ANG_NO_LAMBDA
    fn = float(fn_lin)
    if fn > WRENCH_FN_ANG_MAX_LAMBDA:
        print(f"\033[93mdtc: angular fn held at "
              f"{WRENCH_FN_ANG_MAX_LAMBDA:g} Hz, not --dtc-fn-lin {fn:g} -- "
              f"the roll axis sits on Lambda's angular floor, so its real "
              f"loop runs ~3x the fn asked for. --dtc-fn-ang {fn:g} to "
              f"override.\033[0m", flush=True)
        return WRENCH_FN_ANG_MAX_LAMBDA
    return fn

STICK_LINEAR_SCALE = 0.0015
STICK_ANGULAR_SCALE = 0.05


def stick_is_idle(twist):
    """Hands-off check using the same scales as SpaceMouseTeleop in this example.

    ``is_idle_twist`` defaults are linear_scale=0.01. Idle cap uses this
    example's stick scale so a 30% Z lift is not treated as rest.
    """
    return is_idle_twist(
        twist,
        linear_scale=STICK_LINEAR_SCALE,
        angular_scale=STICK_ANGULAR_SCALE,
    )


def stick_has_linear(twist):
    """True if the device already passed its deadzone and sent a linear step.

    Field-follow idle uses a 20% cap at this stick scale, which ate slow creep.
    """
    linear = np.asarray(twist.get_linear(), dtype=float).reshape(3)
    return float(np.linalg.norm(linear)) > 1e-12


def follow_linear(lin, *, ignore_z):
    """Stick linear for FieldFollow. Contact drops Z; DualLoop owns press."""
    out = np.asarray(lin, dtype=float).reshape(3).copy()
    if ignore_z:
        out[2] = 0.0
    return out


def field_follow_in_mode(mode, backend):
    """Approach always follows. speedj/dtc contact do too; force_mode does not.

    dtc is deliberately identical to speedj here: everything above the actuation
    seam -- run_field_follow, OrientationField, StartupAlign -- is byte-identical
    between the two, so any visible divergence in the field arrows or the EE
    frame is a bug in the q accessor or the control-group splice, not a gain.
    """
    if mode == "approach":
        return True
    return mode == "contact" and backend in ("speedj", "dtc")


def _dashboard_bring_up(ip: str) -> None:
    sock = socket.create_connection((ip, 29999), timeout=5)
    sock.recv(4096)

    def command(text: str) -> str:
        sock.sendall((text + "\n").encode())
        time.sleep(0.3)
        return sock.recv(4096).decode(errors="replace")

    command("power on")
    time.sleep(2)
    command("brake release")
    for _ in range(30):
        mode = command("robotmode")
        if "RUNNING" in mode:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"{ip}: robot never reached RUNNING mode")
    sock.close()
    print(f"{ip}: RUNNING")


def _make_approach_controller(interface, task_space, home_ee_motor,
                              free_along_normal=True, dls_damping=0.05,
                              per_joint_velocity=True, xy_stiffness=None,
                              ang_stiffness=None, wrench_law=False,
                              k_lin=3000.0, k_ang=50.0, damping_ratio=0.3,
                              damping_ratio_ang=0.1,
                              err_clip_lin=0.01, err_clip_rot=0.2,
                              use_lambda=False, fn_lin=2.0, fn_ang=2.0,
                              rearm_wrench_s=DEFAULT_REARM_WRENCH_S):
    # These are NOT hardware_free_teleop's defaults, despite what this comment
    # used to claim -- that file constructs AdmittanceTeleoperationController()
    # with no arguments, so it runs the class defaults (stiffness 3000,
    # max_joint_velocity 1.0). The values below (2000, 0.8) are what actually
    # runs on hardware and are the ones to reason about; the class defaults are
    # inherited from ufactory_demos' gafro_teleoperation via the root commit and
    # have never been validated against this cell.
    # Do NOT attach OrientationField here: vf.project() every stick sample snaps
    # orientation and makes speedJ feel discontinuous vs the SpaceMouse.
    # Blade order is [angular, linear], so 3 and 4 are the lateral (tool X/Y)
    # axes. Softening only those is the cheapest form of contact compliance: a
    # stuck tool then generates a smaller commanded path, which is what the UR
    # was refusing to follow. It costs lateral tracking speed in proportion --
    # the spring delivers dt*K/M * error, so at the 4 mm max_step_m cap, K=2000
    # gives 8 mm/s and K=500 gives 2 mm/s.
    # Blade indices 0..2 are angular. In contact the stick writes xyz only, so
    # every bit of commanded rotation comes from the field's R -- which means
    # lowering k[3:5] alone changes the *ratio* of sliding to turning, not just
    # the shove. At K_xy=500 the spring commands 2.5 mm/s of slide against
    # 0.04 rad/s of turn, which is 4 mm/s at the tool tip on a 10 cm lever: the
    # arm visibly rotates toward the field instead of sliding, whichever way
    # the stick is pushed. --ang-stiffness is the other half of that ratio.
    k = [2000.0] * 6
    d = [20.0] * 6
    if wrench_law:
        # Under tau = J.T @ W these are *real* gains -- N.m/rad on the angular
        # blades (0..2) and N/m on the linear ones (3..5) -- not the velocity
        # gain the number above is. A flat 2000 is violent on the angular half,
        # so split them the way the reference UR OSC loop does (osc_kp_pos 1000
        # / osc_kp_rot 50, a 20:1 ratio) and take the damping from
        # D = 2*sqrt(K)*zeta rather than leaving the flat 20, which is heavily
        # under-damped at these stiffnesses.
        #
        # k_lin is where lateral authority comes from, and it is deliberately
        # the knob rather than field-follow's lead: the lead is the path
        # contract that must stay comparable with speedj for the A/B, so more
        # newtons come from K, not from letting the tool lag further behind the
        # target. At 5 mm of lead, k_lin=3000 is ~15 N of slide -- past the
        # breakaway that kv*qdot could not reach.
        #
        # D = 2*sqrt(K)*zeta is the critical-damping formula for **M = 1**, and
        # the two blades do not have the same M: ~3 kg of apparent end-point
        # mass on the linear axes, but ~0.05 kg.m^2 on the wrist. At zeta = 1
        # that made D_ang = 14.1 N.m.s/rad against 0.05 kg.m^2, and a damper fed
        # back through a delay T goes unstable around (D/M)*T ~ 1: 2.3 at the
        # nominal 8 ms tick and 6.5 at the 23 ms this loop was measured holding.
        # That is what drove tau to the +/-30 N.m rail 89 times and ended in an
        # emergency stop, with the stick already quiet -- a damper needs no
        # input to pump energy, it feeds on the motion it made.
        #
        # So the ratio is split, and both halves are sized against the real M:
        # zeta_lin 0.3 -> D_lin 33, (D/M)*T = 0.25 even at 23 ms; zeta_ang 0.1
        # -> D_ang 1.4, 0.65 at 23 ms. --dtc-damping-ratio 1.0
        # --dtc-damping-ratio-ang 1.0 restores the pre-fix values.
        k = [float(k_ang)] * 3 + [float(k_lin)] * 3
        zeta_blade = (
            [float(damping_ratio_ang)] * 3 + [float(damping_ratio)] * 3)
        if use_lambda:
            # Under tau = J.T @ (Lambda @ a) the spring output is an
            # acceleration, so K is 1/s^2 and K = omega_n^2 -- a *frequency*,
            # which is the whole reason to shape: it is the same number at every
            # pose, where N/m is only right at the pose it was tuned at.
            #
            # The 20:1 lin/ang split disappears with it. That ratio existed only
            # to paper over the two blades having different apparent M (~3 kg vs
            # ~0.05 kg.m^2); Lambda carries that difference exactly, so both
            # blades take the same natural frequency.
            #
            # Sizing: a sampled PD loop wants omega_n well under the control
            # rate. This loop measures 124 Hz median with 29 ms outliers, and it
            # limit-cycled at 5.9 Hz when k_lin=6000 put omega_n at 7.1 Hz
            # (dtc_20260828_195146.csv). 2 Hz leaves a factor of ~60.
            k = [(2.0 * math.pi * float(fn_ang)) ** 2] * 3 + \
                [(2.0 * math.pi * float(fn_lin)) ** 2] * 3
        # D = 2*zeta*sqrt(K) is 2*zeta*omega_n when K = omega_n^2, so this line
        # is correct in both domains and deliberately not branched.
        d = [2.0 * math.sqrt(ki) * zi for ki, zi in zip(k, zeta_blade)]
    if xy_stiffness is not None:
        k[3] = k[4] = float(xy_stiffness)
    if ang_stiffness is not None:
        k[0] = k[1] = k[2] = float(ang_stiffness)
    approach = AdmittanceTeleoperationController(
        inertia=np.diag([1.0] * 6),
        damping=np.diag(d),
        stiffness=np.diag(k),
        dt=0.001,
        max_joint_velocity=0.8,
        twist_linear_scale=1.0,
        max_qddot=5.0,
        free_along_normal=free_along_normal,
        dls_damping=dls_damping,
        per_joint_velocity=per_joint_velocity,
        err_clip_lin=err_clip_lin,
        err_clip_rot=err_clip_rot,
        rearm_wrench_s=rearm_wrench_s,
    )
    approach.set_task_space(task_space)
    approach.set_interface(interface)
    approach.set_objective(ga.control.SingleArmMotorControlObjective(home_ee_motor))
    return approach


def _wrench_law_kwargs(args):
    """Pose-spring gains for --dtc-law wrench, or nothing at all otherwise.

    Applied at DTC handoff (and force_mode gafropy reopen), not during the
    speedJ align phase -- Lambda/fn gains are 1/s^2 and would corrupt speedJ.
    """
    if getattr(args, "contact_backend", None) != "dtc" or args.dtc_law != "wrench":
        return {}
    fn_ang = resolve_wrench_fn_ang(
        args.dtc_lambda, args.dtc_fn_lin, args.dtc_fn_ang)
    dmax = resolve_wrench_damping_max(args.dtc_hz, args.dtc_damping_max)
    zeta_lin, zeta_ang = resolve_wrench_zeta(
        args.dtc_lambda, args.dtc_damping_ratio, args.dtc_damping_ratio_ang,
        fn_lin=args.dtc_fn_lin, fn_ang=fn_ang,
        damping_max=dmax,
        warn=lambda m: print(m, flush=True))
    # Unconditional, because the interesting case is the one where the cap did
    # *not* warn and the reader still needs to know how close it came.
    note = (f"dtc wrench gains: zeta_lin={zeta_lin:.3f} zeta_ang={zeta_ang:.3f}")
    if args.dtc_lambda:
        # D = 2*zeta*omega_n only means 1/s under Lambda; without it D is
        # N.s/m and N.m.s/rad and the cap does not apply at all.
        note += (f"  fn_lin={args.dtc_fn_lin:g} fn_ang={fn_ang:g}"
                 f"  D_lin={2.0 * zeta_lin * 2.0 * math.pi * args.dtc_fn_lin:.2f}"
                 f" D_ang={2.0 * zeta_ang * 2.0 * math.pi * fn_ang:.2f} 1/s"
                 f"  (cap {dmax:g} at --dtc-hz {args.dtc_hz:g})")
        # The number that decides whether the arm can move at all, and it is a
        # product of two flags that look independent. Under --no-follow-rebase-
        # on-live a held stick accumulates lead up to --max-lead-m, so the most
        # force the spring can ever make is Lambda * K * max_lead. Measured on
        # this rig Lambda_lin is ~6 kg and breakaway is ~18 N at the tool, and
        # dtc_20260903_131641.csv (fn 2.0 + --max-lead-m 0.02 -> 19.6 N) stalled
        # on 66.3% of spring-on ticks with the operator holding the stick: a
        # 1.1x margin over friction is not a margin. Want >= 2x.
        f_max = LAMBDA_LIN_NOMINAL_KG * (2.0 * math.pi * args.dtc_fn_lin) ** 2 \
            * args.max_lead_m
        warn = "" if f_max >= 2.0 * BREAKAWAY_LIN_NOMINAL_N else (
            f"  \033[93m<{2.0 * BREAKAWAY_LIN_NOMINAL_N:.0f}N = 2x breakaway; "
            f"raise --max-lead-m or --dtc-fn-lin\033[0m")
        note += (f"\n  stick force ceiling ~{f_max:.0f} N "
                 f"(Lambda~{LAMBDA_LIN_NOMINAL_KG:g}kg * K_lin * --max-lead-m "
                 f"{args.max_lead_m:g}), breakaway ~{BREAKAWAY_LIN_NOMINAL_N:g} N"
                 + warn)
    if getattr(args, "target_force", None) is not None:
        fd = abs(float(args.target_force))
        pm = resolve_press_max_n(args)
        kp = resolve_press_kp(args)
        del kp
        note += (f"\n  press: approach {args.press_approach_n:g} N (the pose "
                 f"spring approaches), hold Fd + {args.press_kp_hold:g}*(Fd-Fs), "
                 f"ceiling {pm:.0f} N for --target-force {fd:g} N")
    print(note, flush=True)
    return dict(
        wrench_law=True,
        k_lin=args.dtc_k_lin,
        k_ang=args.dtc_k_ang,
        damping_ratio=zeta_lin,
        damping_ratio_ang=zeta_ang,
        err_clip_lin=args.dtc_err_clip_lin,
        err_clip_rot=args.dtc_err_clip_rot,
        use_lambda=args.dtc_lambda,
        fn_lin=args.dtc_fn_lin,
        fn_ang=fn_ang,
        rearm_wrench_s=args.dtc_rearm_wrench_s,
    )


def _wrench_gain_note(args) -> str:
    """What the gains were *asked* for, for the log's `#` header.

    The header proper reads the live K and D, which is right and is blind to
    exactly one thing: the D ceiling. ``_cap_zeta`` lowers zeta silently apart
    from one line of yellow in scrollback, and scrollback is not evidence --
    dtc_20260830_135043 (D_lin 15.08) and dtc_20260830_135527 (D_lin 12.00)
    were a --dtc-damping-ratio 0.3 -> 0.4 experiment that came out *less*
    damped than it started, and nothing in either log said which cap was in
    force. So the request and the ceiling go in the record next to the result.
    """
    if getattr(args, "contact_backend", None) != "dtc" or args.dtc_law != "wrench":
        return ""
    zeta_lin = (WRENCH_ZETA_LIN if args.dtc_damping_ratio is None
                else args.dtc_damping_ratio)
    zeta_ang = (WRENCH_ZETA_ANG if args.dtc_damping_ratio_ang is None
                else args.dtc_damping_ratio_ang)
    bits = [f"zeta_req_ang={float(zeta_ang):g}",
            f"zeta_req_lin={float(zeta_lin):g}",
            f"fn_lin={float(args.dtc_fn_lin):g}",
            f"fn_ang={resolve_wrench_fn_ang(args.dtc_lambda, args.dtc_fn_lin, args.dtc_fn_ang):g}"]
    if args.dtc_lambda:
        bits.append(
            f"dmax={resolve_wrench_damping_max(args.dtc_hz, args.dtc_damping_max):g}")
    return "  ".join(bits)


def _hold_field_orientation(args) -> bool:
    """Whether the idle hold keeps the commanded field orientation.

    Defaults on exactly where the wrench law runs. Under speedJ the pose spring
    is a *velocity* command chased by a position servo, so a persistent angular
    error with hands off the stick servos the wrist continuously -- and that
    path has to stay bit-identical for the C153 A/B anyway.
    """
    if getattr(args, "hold_field_orientation", None) is not None:
        return bool(args.hold_field_orientation)
    return (getattr(args, "contact_backend", None) == "dtc"
            and args.dtc_law == "wrench")


def dtc_compliance_hold_mode(args) -> bool:
    """Push-return spring gains: idle re-base off so a push builds error."""
    return bool(getattr(args, "dtc_compliance_hold", False))


def push_test_line(controller, st) -> str:
    """The bench push test's 1 Hz console line.

    ``|W_ang|`` is here on purpose and is the reading that keeps this run
    honest: it is ~0 by construction (no commanded rotation), so a push test
    that goes well says nothing about --dtc-damping-ratio-ang. ``tau_peak`` is
    a single sample, so read the --dtc-log for anything between prints.
    """
    err_lin = float(np.linalg.norm(
        np.asarray(getattr(controller, "_last_err_lin", np.zeros(3)),
                   dtype=float).reshape(3))) * 1000.0
    return (f"dtc push test: |e_lin|={err_lin:.1f}mm "
            f"|W_lin|={st.get('w_lin', 0.0):.1f}N "
            f"|W_ang|={st.get('w_ang', 0.0):.2f}Nm "
            f"tau_peak={st.get('tau_peak', 0.0):.2f}Nm "
            f"hz={st.get('hz_meas', 0.0):.0f} "
            f"railosc={st.get('n_rail_osc', 0)}")


def dtc_push_test_mode(args) -> bool:
    """Bench push test: compliance-hold gains with *nothing else running*.

    The A-side of the A/B. No startup align, no input device, no field-follow,
    so the commanded pose is the handoff pose and ``set_hold_rotation`` is
    never called -- ``_hold_rotation`` stays None and ``hold_pose_at`` snaps
    target := live pose, which means **the angular error is identically zero**.
    That is the whole point and it is also the trap: a gain the push test does
    not exercise is not validated by the push test going well. Read the 1 Hz
    line's ``|W_ang|`` -- if it is 0.00, ``--dtc-damping-ratio-ang`` did not
    participate in whatever you just concluded.
    """
    return bool(getattr(args, "dtc_push_test", False))


def follow_rebase_on_idle(args, *, on_dtc) -> bool:
    """Whether a quiet stick re-bases the pose target onto the live EE.

    ``--dtc-hold-at-tick`` is a *DTC wrench-law* gain -- it is the knob
    ``--dtc-compliance-hold`` turns off so a push builds spring error instead of
    being erased at 200 Hz. It must not reach the speedJ segment. Two reasons,
    both load-bearing:

    - speedJ has to stay bit-identical or the C153 A/B means nothing.
    - the speedJ pose spring has no ``_input_active`` gate at all
      (``controller.py``), so ``hold_live_pose`` on a quiet stick is the *only*
      thing keeping a hands-off arm from driving toward a stale target. Turning
      it off there is arm sag, not compliance.

    ``on_dtc`` is "the DTC backend is the live actuator", i.e. the handoff has
    happened -- not "the run was launched with ``--contact-backend dtc``", which
    is still true throughout the speedJ startup align.
    """
    if bool(getattr(args, "dtc_hold_on_release", False)) and bool(on_dtc):
        # The release snap owns the idle target now, and it must be the *only*
        # thing writing it: field-follow's quiet branch re-bases at the main
        # loop's 50 Hz, which would re-introduce the chase this flag removes.
        return False
    return bool(args.dtc_hold_at_tick) or not bool(on_dtc)


def defer_dtc_until_align_done(*, dtc_backend, align, args=None) -> bool:
    """True: run speedJ through startup align, then hand off to DTC once.

    ``--dtc-compliance-hold`` uses the same align -> DTC path as teleop; only
    the bench push test skips it, because it holds wherever the arm already is.
    """
    if dtc_backend is None:
        return False
    if args is not None and dtc_push_test_mode(args):
        return False
    return align is not None and not align.finished


def route_spacemouse_twist(twist, *, sim_probe, on_hardware):
    """Viser probe consumes the stick; hardware only sees it when there is no probe."""
    if sim_probe is not None:
        if stick_is_idle(twist):
            sim_probe.clear_pending()
        else:
            sim_probe.add_body_linear(
                np.asarray(twist.get_linear(), dtype=float).reshape(3))
        return True
    on_hardware(twist)
    return False


def apply_terminal_key(key, *, align, sim_teleop):
    """Enter starts hardware align. ``t`` toggles viser-only stick while waiting."""
    if align is None or align.finished:
        return None
    waiting = align.phase in ("idle", "wait") and not align._triggered
    if key in ("t", "T"):
        if not waiting:
            return None
        sim_teleop["on"] = not sim_teleop["on"]
        return "sim_on" if sim_teleop["on"] else "sim_off"
    if key in ("\n", "\r"):
        if align.phase not in ("idle", "wait"):
            return None
        sim_teleop["on"] = False
        align.trigger()
        return "align"
    return None


def drive_startup_align(align, controller, now, T_flange, R_field, T_ee,
                        T_flange_display=None):
    """Track a slerped flange target. Hold and freeze when the one-shot finishes.

    ``T_flange`` / ``T_ee`` must be in the controller world (hardware q, rail at
    0). ``T_flange_display`` is the vention-overlaid flange the orange ghost uses.
    """
    was_done = align.finished
    cmd = align.step(
        now, T_flange, R_field, T_ee, T_flange_display=T_flange_display)
    if cmd.T_ee is not None:
        controller.get_objective().set_target(motor_from_matrix(cmd.T_ee))
        controller.set_input_active(True)
    elif cmd.finished and not was_done:
        controller.set_input_active(False)
        controller.hold_live_pose()
    return cmd


def resolve_press_kp(args):
    """Proportional trim on the press, N per N of force error.

    **This sets the approach authority, and 0.5 could not approach at all.**
    The law is ``F = Fd + kp*(Fd - Fs)``, so with the tool clear of the part
    (``Fs ~ 0``) the most it will ever command is ``Fd * (1 + kp)``. At Fd 5 and
    kp 0.5 that is **7.5 N against ~18 N of breakaway**, so the arm never moves,
    Fs never changes, and the loop sits there forever -- measured exactly that
    on contact_20260903_135326.csv: 3832 rows, ``F_press`` median **7.33 N**
    (= 5 + 0.5*(5 - 0.34), the law working perfectly), ``Fs`` median +0.34, and
    the tool hovering. A *velocity* command escapes this because velocity
    integrates into position; a *force* command does not, so an under-sized
    command is a permanent standoff rather than a slow approach.

    **Raising kp is free in steady state.** In contact the surface reacts what
    the arm pushes, so ``Fs -> F`` and ``F = Fd + kp*(Fd - F)`` gives
    ``F = Fd`` for *any* kp. kp buys approach authority and changes nothing at
    the target. The default is therefore sized so the free-space command clears
    breakaway: ``Fd*(1 + kp) >= Fd + breakaway`` i.e. ``kp >= breakaway/Fd``.

    Deliberately still not an integrator: an I term on a stiction nonlinearity
    is a stick-slip amplifier, and this gets the same approach authority from a
    term that vanishes at the target instead of winding up at it.
    """
    if args.press_kp is not None:
        return float(args.press_kp)
    # Breakaway sizing is retired: the pose spring does the approach now, so
    # the press never has to start the arm moving. Kept as a flag because a rig
    # without spring authority on the normal would still need it.
    #
    # ...and --press-auto-approach IS that rig. Measured: with the spring given
    # the approach, contact_20260905_195152.csv drove the carrot to the lead
    # clamp (err_z -33 mm, ~70 N commanded) and the arm did not follow it in --
    # |qdot| 0.004 rad/s, Fs never past 4.5 N. So under auto-approach the press
    # is again the thing that has to start the arm moving, and the sizing that
    # was retired with the hand-off comes back with it: the free-space command
    # is Fd*(1 + kp), so kp >= breakaway/Fd is what makes it clear breakaway at
    # all. At kp 0.5 and Fd 10 that command is 15 N -- and 15 N is measured
    # under breakaway in this direction (dtc_20260903_150932.csv held 23.0 N
    # along the normal for 43 s with |qdot| 0.0000 and Fs under 0.73 N).
    if getattr(args, "press_auto_approach", False):
        fd = abs(float(args.target_force))
        if fd > 0.0:
            return max(DEFAULT_PRESS_KP, BREAKAWAY_LIN_NOMINAL_N / fd)
    return DEFAULT_PRESS_KP


def resolve_press_max_n(args):
    """Ceiling on the commanded press, in newtons.

    The library default is ``3 * Fd``, which is right for a low-friction rig and
    **wrong here**. Entering contact frees the normal axis of pose spring
    (``free_along_normal``), so ``F_press`` is the *only* force along it -- and
    it must first overcome the arm's own breakaway friction before any of it
    reaches the part. Measured on this rig breakaway is ~18 N at the tool, so at
    ``--target-force 5`` a 15 N ceiling cannot move the arm at all: the tool
    hovers over the surface and Fs never rises, which is exactly what the first
    contact attempt did.

    So the floor is ``breakaway + 2*Fd`` -- enough to break away and still have
    Fd left over, with Fd of margin for the trim. Raising this is not the safety
    risk it looks like: the P trim can only reach the ceiling if Fs reads far
    *negative*, and the wrong-sign guard zeroes the press at
    ``Fs <= min(-2, -0.5*Fd)`` long before that.
    """
    if args.press_max_n is not None:
        return float(args.press_max_n)
    fd = abs(float(args.target_force))
    return max(3.0 * fd, BREAKAWAY_LIN_NOMINAL_N + 2.0 * fd)


def _contact_log_path(spec, save_dir="speedj_logs", prefix="contact"):
    """Resolve --contact-log / --dtc-log. 'none' disables; 'auto' timestamps."""
    if spec in (None, "", "none", "off"):
        return None
    if spec != "auto":
        return spec
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(pathlib.Path(save_dir).resolve() / f"{prefix}_{stamp}.csv")


def enter_speedj_contact(*, mode, contact_controller, approach_controller, interface=None):
    """Keep the pose-spring speedJ plant. Overlay DualLoop Z. Do not swap to line admittance."""
    del interface  # caller may pass it; we must not interface.control(contact)
    mode["active"] = "contact"
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    approach_controller.set_tool_z_velocity(0.0)
    approach_controller.set_z_compliant(True)
    contact_controller.set_on_vz(approach_controller.set_tool_z_velocity)
    contact_controller.start()


def use_wrench_press(args) -> bool:
    """True when the press must be a force (set_press_wrench), not a velocity.

    The wrench law never reads ``set_tool_z_velocity`` -- that lives in
    ``compute_control_command``, which it bypasses -- so entering contact with
    the velocity overlay under ``--dtc-law wrench`` would zero Kz, free the
    normal and then never press: the tool hovers and the force log reads a
    healthy zero.
    """
    return (getattr(args, "contact_backend", None) == "dtc"
            and getattr(args, "dtc_law", None) == "wrench")


def enter_wrench_contact(*, mode, contact_controller, approach_controller,
                         interface=None):
    """Contact under ``--dtc-law wrench``: the press is a force, not a velocity.

    Identical in shape to ``enter_speedj_contact`` -- same pose spring, same
    ``z_compliant`` + ``free_along_normal`` projection, same DualLoop force
    thread -- with ``set_on_press_wrench`` in place of ``set_on_vz``. The
    velocity overlay is explicitly zeroed rather than merely unused, because
    ``compute_control_command`` would still read it if the backend were ever
    handed back to speedJ mid-session.
    """
    del interface
    mode["active"] = "contact"
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    approach_controller.set_tool_z_velocity(0.0)
    approach_controller.set_press_wrench(None)
    approach_controller.set_z_compliant(True)
    contact_controller.set_on_press_wrench(approach_controller.set_press_wrench)
    # The spring keeps the normal until the press has arrived: the press is
    # force-limited and the gap is friction-limited.
    approach_controller.set_press_engaged(False)
    contact_controller.set_on_press_engaged(approach_controller.set_press_engaged)
    contact_controller.start()


def exit_wrench_contact(*, mode, contact_controller, approach_controller,
                        interface=None):
    """Drop the press force and freeze on the same pose spring."""
    del interface
    contact_controller.stop()
    contact_controller.set_on_press_wrench(None)
    contact_controller.set_on_press_engaged(None)
    approach_controller.set_press_wrench(None)
    approach_controller.set_press_engaged(True)
    approach_controller.set_z_compliant(False)
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    mode["active"] = "approach"


def exit_speedj_contact(*, mode, contact_controller, approach_controller, interface=None):
    """Drop Z overlay and freeze on the same pose-spring plant."""
    del interface
    contact_controller.stop()
    contact_controller.set_on_vz(None)
    approach_controller.set_tool_z_velocity(0.0)
    approach_controller.set_z_compliant(False)
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    mode["active"] = "approach"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task-space", choices=["left_ee", "right_ee"], default="right_ee")
    parser.add_argument(
        "--device", choices=["keyboard", "spacemouse"], default="spacemouse")
    parser.add_argument(
        "--spacemouse-path",
        help="hidraw path for the SpaceMouse (e.g. /dev/hidraw5). "
        "Default: by-id symlink, else newest hidraw when several are connected.")
    parser.add_argument(
        "--spacemouse-index", type=int,
        help="index into sorted connected hidraw paths (0=first). "
        "Ignored when --spacemouse-path is set.")
    parser.add_argument(
        "--check-spacemouse", action="store_true",
        help="list connected SpaceMice, open the selected one, poll 3s, and exit")
    parser.add_argument(
        "--hardware-config", default=DEFAULT_HARDWARE_CONFIG,
        help="YAML file mapping control groups to driver configs (Configuration.hardware_config)")
    parser.add_argument(
        "--no-bring-up", dest="bring_up", action="store_false",
        help="skip the dashboard power-on/brake-release sequence (already-running real hardware)")
    parser.add_argument(
        "--object", default=DEFAULT_OBJECT,
        help="mesh or point cloud file to place on the table (see teleoperation.py's module docstring)")
    parser.add_argument(
        "--normal-orientation", type=int, choices=[1, -1], default=1,
        help="sign correction for point-cloud objects' PCA-estimated normals (see teleoperation.py)")
    parser.add_argument(
        "--target-force", type=float, default=15.0,
        help="desired contact force (N) along tool Z in contact mode")
    parser.add_argument(
        "--force-sign", type=float, default=1.0,
        help="press sign along tool Z (default +1, verified on hardware: press "
             "is along +tool Z). Fixed knob; never latched from Fz polarity, and "
             "a field normal refines the press axis without flipping this sign.",
        )
    parser.add_argument(
        "--no-zero-ft", dest="zero_ft_sensor", action="store_false",
        help="do not call zero_ftsensor() on each force_mode contact entry "
             "(ignored for speedj). Use when already touching so preload is not zeroed.",
        )
    parser.add_argument(
        "--mode-button", type=int, default=1,
        help="spacemouse button index that toggles between approach and contact "
             "mode (with --device keyboard, index 1 is 'v' and index 0 is 't')")
    parser.add_argument(
        "--contact-backend", choices=["speedj", "force_mode", "dtc"], default="dtc",
        help="contact implementation ('speedj' = pose spring + Z force overlay). "
             "'force_mode' is UR hybrid force/position, not admittance. 'dtc' runs "
             "startup align under speedJ, then hands RTDE to Direct Torque for "
             "all teleop (free space + contact). Contact is the same pose spring "
             "+ Z overlay; only the actuator changes at that one handoff.")
    # --- dtc backend. The law above the seam is unchanged; these only set how
    # hard it is allowed to push and how often the host refreshes the on-robot
    # tau latch. Start tau_max low: it is the only thing left bounding a grab
    # now that the UR's own path supervisor is out of the loop.
    parser.add_argument(
        "--dtc-law", choices=["velocity", "wrench"], default="wrench",
        help="dtc control law. 'velocity' is the first port: it keeps the whole "
             "speedJ velocity pipeline (inertia, dt, DLS inverse, joint clamps) "
             "and closes tau = kv*(qdot_cmd - qdot_meas) under it. Those caps "
             "exist to protect a position servo and are what starve the torque "
             "law -- they turn a 4 mm error into ~0.02 rad/s into ~1 N.m, "
             "against a UR5e breakaway of several N.m. 'wrench' skips all of "
             "it and sends tau = clip(J.T @ W, +/-tau_max) straight off the "
             "pose spring, so lateral force is set on purpose as K*e in "
             "newtons. --dtc-kv/--dtc-ki are ignored under 'wrench'; the gains "
             "that matter are --dtc-k-lin/--dtc-k-ang (default %(default)s)")
    parser.add_argument(
        "--dtc-k-lin", type=float, default=3000.0, metavar="N_PER_M",
        help="--dtc-law wrench: linear stiffness (N/m) on blades 3..5. This is "
             "where lateral authority comes from -- at field-follow's 5 mm "
             "lead, 3000 N/m is ~15 N of slide. Prefer raising this over "
             "--max-lead-m, which is the path contract that has to stay "
             "comparable with speedj for the A/B -- but --max-lead-m is the "
             "hard ceiling on the error this multiplies, so at the 5 mm "
             "default no k-lin below ~8000 reaches 40 N (default %(default)s)")
    parser.add_argument(
        "--dtc-k-ang", type=float, default=50.0, metavar="NM_PER_RAD",
        help="--dtc-law wrench: angular stiffness (N.m/rad) on blades 0..2. A "
             "different unit from --dtc-k-lin, hence the 20:1 split the "
             "reference UR OSC loop uses; a flat 3000 here is violent "
             "(default %(default)s)")
    parser.add_argument(
        "--dtc-damping-ratio", type=float, default=None, metavar="ZETA",
        help="--dtc-law wrench: LINEAR damping as D = 2*sqrt(K_lin)*zeta "
             f"(default {WRENCH_ZETA_LIN}). Under --dtc-lambda M is 1, so this "
             "reads as badly under-damped and is not: raising it to 0.7 put "
             "the arm into a 24.5 Hz limit cycle with the stick at rest "
             "(dtc_20260830_005006.csv). zeta here is bounded by the loop "
             "delay against a structural mode, not by overshoot -- see "
             "resolve_wrench_zeta and --dtc-damping-max")
    parser.add_argument(
        "--dtc-damping-ratio-ang", type=float, default=None, metavar="ZETA",
        help="--dtc-law wrench: ANGULAR damping as D = 2*sqrt(K_ang)*zeta "
             f"(default {WRENCH_ZETA_ANG}, raised from 0.10 2026-08-31 -- it "
             # argparse %-expands help strings, so a literal percent must be
             # doubled or --help dies with "unsupported format character".
             "cut the rig's ~15 Hz structural-mode ring from 19.4%% to "
             "2.5-3.3%% of q-dot power with no regression, see WRENCH_ZETA_ANG). "
             "Kept separate from "
             "--dtc-damping-ratio even under --dtc-lambda, where Lambda does "
             "remove the apparent-M difference the split was for: the wrist is "
             "the joint that rails first when the damper turns negative at the "
             "rig's 24.5 Hz mode (j4 on its cap for 18.9%% of ticks at 0.7), so "
             "this half keeps its own, smaller, measured-stable number")
    parser.add_argument(
        "--dtc-lambda-max-lin", type=float, default=LAMBDA_MAX_LIN_KG,
        metavar="KG",
        help="--dtc-lambda only: ceiling on Lambda's linear eigenvalues, kg "
             f"(default {LAMBDA_MAX_LIN_KG:g}). Lambda is roughly m/sigma^2, "
             "so it is LARGE exactly along a direction the arm has poor "
             "leverage in -- which makes it the term that compensates a bad "
             "moment arm, and this ceiling the thing that stops it. At the "
             "pose of dtc_20260904_105747.csv the pre-clamp eigenvalues are "
             "3.7 / 5.1 / 42.2 kg, so 15 binds by ~3x in the direction that "
             "dead-zoned and is inert in the other two. Raise it to test that. "
             "It is still a real bound: Lambda multiplies the commanded force "
             "directly, so the force ceiling rises with it -- the 200 kg "
             "version of this clamp is what put the arm into an uncommanded "
             "high-speed motion.")
    parser.add_argument(
        "--dtc-damping-max", type=float, default=None,
        metavar="PER_S",
        help="--dtc-law wrench + --dtc-lambda: ceiling on the damping "
             "coefficient D = 2*zeta*omega_n in 1/s, applied to both blades "
             "after --dtc-damping-ratio* and --dtc-fn-* are resolved. This is "
             "the knob that actually bounds the damper, because D grows with "
             "fn as well as zeta -- a --dtc-fn-lin sweep at fixed zeta walks "
             "into the same limit cycle, and this takes zeta down instead of "
             "letting it. It is NOT the rigid-body D*T ~ 1 bound, which said "
             "0.21 was safe on the run that oscillated. 0 disables the cap. "
             "Default 12.0 at every rate. It briefly scaled with --dtc-hz and "
             "that put D_lin at 19.1, which limit-cycled at 32 Hz -- f_90 for "
             "this loop -- and swung the arm into the workstation: raising the "
             "rate sweeps f_90 up into structure that was out of reach before, "
             "so a higher rate is not a licence for more D. Above 12 wants a "
             "short run and the rail section of analyze_dtc_log.py first -- "
             "see resolve_wrench_damping_max")
    parser.add_argument(
        "--dtc-rail-osc-flips", type=int, default=DEFAULT_RAIL_OSC_FLIPS,
        metavar="N",
        help="--contact-backend dtc: float tau when any joint alternates "
             "between +tau_max and -tau_max this many times in "
             f"{RAIL_OSC_WINDOW_S:g} s. tau_max is the ONLY bound on a limit "
             "cycle's amplitude -- err_clip bounds the spring half of the "
             "wrench and the damping half -Lambda*D*v has no clip at all -- so "
             "an oscillating loop throws the arm as hard as the cap allows, "
             "50 N.m at the shoulder here. Rail-to-rail SIGN ALTERNATION is "
             "what separates a limit cycle from an honest stall, which sits on "
             "one sign and must never be cut. Healthy runs on this rig top out "
             "at 9 flips and only on j3/j4; the run that hit the workstation "
             "read 20/11/21/18 on j1-j4 at once. 0 disables (do not).")
    parser.add_argument(
        "--dtc-rail-osc-hold-s", type=float, default=DEFAULT_RAIL_OSC_HOLD_S,
        metavar="S",
        help="--contact-backend dtc: how long tau stays floated after a "
             "--dtc-rail-osc-flips trip. The law, the joint hold and the "
             "re-arm blend are all reset with it, so resuming cannot hand back "
             "the wrench that was oscillating.")
    parser.add_argument(
        "--no-dtc-hold-at-tick", dest="dtc_hold_at_tick",
        action="store_false", default=True,
        help="--dtc-law wrench: re-snap the idle pose target in the dtc loop "
             "rather than at --follow-hz. The snap period is a control gain -- "
             "target = pose(t-T) adds K*T/2 of delayed damping, 75 N.s/m at "
             "20 Hz -- so this off restores the pre-fix behavior")
    parser.add_argument(
        "--dtc-hold-on-release", action="store_true",
        help="--dtc-law wrench: snap the idle pose target ONCE on the stick's "
             "1->0 edge instead of every idle tick. The per-tick snap makes the "
             "target chase the arm, so a push on a released arm builds no pose "
             "error and meets no spring at all -- zero idle stiffness, not a "
             "friction dead band. This keeps 'let go = stop' (the carrot is "
             "discarded at release) while leaving the spring live, and also "
             "stops field-follow re-basing on idle. Gives up the K*T/2 of "
             "delayed damping the per-tick snap contributed")
    parser.add_argument(
        "--dtc-err-clip-lin", type=float, default=0.05, metavar="M",
        help="--dtc-law wrench: pose error clip (m) before it becomes force. "
             "This is the shove cap, replacing max_step_m. Without --dtc-lambda "
             "the bound in newtons is --dtc-k-lin times this; with it, the "
             "bound is (2*pi*--dtc-fn-lin)^2 * this * apparent mass -- at the "
             "defaults ~355 1/s^2 * 0.05 m * ~4 kg = ~71 N, which is the peak "
             "measured on hardware (default %(default)s)")
    parser.add_argument(
        "--dtc-rearm-wrench-s", type=float, default=DEFAULT_REARM_WRENCH_S,
        metavar="S",
        help="--dtc-law wrench: blend the spring's first active tick out of "
             "the idle hold wrench over this many seconds. The wrench-law "
             "counterpart to --rearm-slerp-s, which only ramps the commanded "
             "rotation. Armed by the input_active 0->1 edge only, so it is "
             "inactive during ordinary sliding and adds no phase lag there; 0 "
             "restores the one-tick step (default %(default)s)")
    parser.add_argument(
        "--dtc-lambda", dest="dtc_lambda",
        action=argparse.BooleanOptionalAction, default=True,
        help="--dtc-law wrench: inertia shaping. tau = J.T @ (Lambda @ a) with "
             "Lambda = (J M^-1 J.T)^-1, instead of tau = J.T @ W. Without it "
             "one wrench is handed to joints whose measured mobility spans 78x "
             "on this arm (examples/analyze_dtc_mobility.py), so the wrist "
             "takes the motion and the shoulder takes the force -- which is why "
             "align (a rotation, done by the mobile joints) worked while "
             "tracking (a translation, done by the stiff ones) did not. Lambda "
             "makes the EE a unit mass in every direction, so a gain is correct "
             "at more than one pose. Changes what K means, hence opt-in: "
             "--dtc-k-lin/--dtc-k-ang are ignored and --dtc-fn-lin/--dtc-fn-ang "
             "take over (default: on)")
    parser.add_argument(
        "--dtc-fn-lin", type=float, default=3.0, metavar="HZ",
        help="--dtc-lambda: linear closed-loop natural frequency (Hz). K = "
             "(2*pi*f)^2 in 1/s^2. Must sit well under the control rate; this "
             "loop holds ~124 Hz and limit-cycled at 5.9 Hz when omega_n was "
             "7.1 Hz. This is also the lever on *tracking lag*, which is not a "
             "gain-scheduling artifact but a-over-K: the arm sits err = "
             "a_needed/K behind the carrot, so the 10.5 mm median error "
             "measured at 3.0 halves at 4.24 and doubles at 2.1. Confirmed "
             "both ways on hardware -- 1.5 was reported worse (more lag), 3.0 "
             "better. Sweep it up, not down. Note raising it also raises "
             "D = 2*zeta*omega_n, which is the coefficient that limit-cycled "
             "this rig at 24.5 Hz; --dtc-damping-max holds D fixed and lowers "
             "zeta as fn rises, so the sweep is safe but gets progressively "
             "less damped -- watch the oscillation section of "
             "analyze_dtc_log.py, not just |W_lin| (default %(default)s)")
    parser.add_argument(
        "--dtc-fn-ang", type=float, default=None, metavar="HZ",
        help="--dtc-lambda: angular natural frequency (Hz). Lambda already "
             "carries the two blades' different apparent M, so the same number "
             "as the linear one is correct -- hence this now DEFAULTS to "
             "--dtc-fn-lin under --dtc-lambda (see resolve_wrench_fn_ang) and "
             "to 1.5 without it. It was detuned to 0.3 because the angular "
             "blade railed the wrists on a re-arm, but that was a *step* with "
             "its own fix in --rearm-slerp-s, and detuning a steady-state loop "
             "to kill a transient is the wrong lever: at 0.3 the angular "
             "command was pinned at its clip on every tick (K_ang = "
             "(2*pi*0.3)^2 = 3.55 1/s^2 against K_lin = 355, so 0.2 rad gives "
             "0.71 rad/s^2 and needs ~1.4 s to close the p95 error while the "
             "stick moves the carrot in tens of ms), and tool Z sat 11.7 deg "
             "off the field normal at the median. Hardware then confirmed the "
             "tie: at 1.5 (dtc_20260830_002254.csv) tool Z sat 4.17 deg / "
             "22.75 deg p95 off the normal, and at 3.0 = fn_lin "
             "(dtc_20260830_002759.csv) 1.33 / 5.28 deg. Stop raising it when "
             "|W_ang| rings or dt_max_ms climbs")
    parser.add_argument(
        "--dtc-err-clip-rot", type=float, default=0.2, metavar="RAD",
        help="--dtc-law wrench: angular pose error clip (rad) "
             "(default %(default)s)")
    parser.add_argument(
        "--dtc-joint-hold-kp", type=float, default=DEFAULT_JOINT_HOLD_KP,
        metavar="NM_PER_RAD",
        help="--dtc-law wrench: joint-space spring on the joints tau = J.T @ W "
             "structurally cannot reach -- in practice wrist 3. Its jacobian "
             "column is pure tool-Z angular (the TCP is on that axis: at "
             "|W_lin| = 98.8 N the measured |tau5| was 4e-4 N.m, a moment arm "
             "under 4 um) and the field constrains only the *direction* of tool "
             "Z, never the roll about it -- so the wrench is orthogonal to that "
             "column and tau5 was 0 on all 3056 ticks of "
             "dtc_20260829_233332.csv and all 3545 of the run before it. Under "
             "speedJ the UR position-servoed that joint; under torque nothing "
             "does and it is held by friction alone. The target follows the "
             "live angle while the stick is quiet and wherever the wrench does "
             "own the joint, so this cannot fight a commanded roll. 0 leaves "
             "only the damper (default %(default)s)")
    parser.add_argument(
        "--dtc-joint-hold-kd", type=float, default=DEFAULT_JOINT_HOLD_KD,
        metavar="NM_PER_RADPS",
        help="damping half of --dtc-joint-hold-kp. Needs no target, so it can "
             "never drive the arm anywhere -- it only stops a free joint "
             "running away (default %(default)s)")
    parser.add_argument(
        "--dtc-joint-hold-joints", type=int, nargs="*", default=None,
        metavar="J",
        help="control-group slots --dtc-joint-hold-kp/-kd act on "
             "(default: the last driven joint, i.e. wrist 3). Empty disables. "
             "Their torque shows up in the per-joint tau columns of --dtc-log "
             "like any other; stats() reports n_joint_hold")
    parser.add_argument(
        "--dtc-stiction-assist-nm", type=float,
        default=DEFAULT_STICTION_ASSIST_NM, metavar="NM",
        help="--dtc-law wrench: bounded feedforward push, in the direction the "
             "task torque already points, on a joint from "
             "--dtc-joint-hold-joints that is commanded but not moving. Wrist "
             "3 sat at the Lambda floor's full 1.42 N.m ceiling on three runs "
             "and travelled 0.02-0.49 deg end to end -- breakaway there is "
             "above anything the floor can command -- so the roll error never "
             "closed and leaked onto the other angular blades through Lambda's "
             "cross terms. Friction is the deficiency, so the fix is "
             "feedforward: this moves no pole, where raising "
             "LAMBDA_MIN_ANG_KGM2 to reach breakaway would inflate that axis's "
             "own loop to ~15 Hz, onto this rig's structural mode. It ramps in "
             "over 0.25 s, releases the moment the joint breaks free, and "
             "0 disables it (default %(default)s)")
    parser.add_argument(
        "--dtc-stiction-max-nm", type=float, default=DEFAULT_STICTION_MAX_NM,
        metavar="NM",
        help="ceiling on task torque + assist for an assisted joint. This is "
             "what keeps a friction compensator from becoming an authority: "
             "wrist 3's own --dtc-tau-max is 30 N.m (default %(default)s)")
    parser.add_argument(
        "--dtc-friction-comp", dest="dtc_friction_comp",
        action=argparse.BooleanOptionalAction, default=True,
        help="ask the UR for its own joint friction feedforward. The reference "
             "UR OSC loop runs with it off; we default it on because breakaway "
             "is the thing that stalled the velocity law. Under --dtc-law "
             "wrench the spring already makes real breakaway force, so this is "
             "worth A/B-ing (default: on)")
    parser.add_argument(
        "--dtc-kv", type=float, nargs="+", default=list(DEFAULT_KV),
        metavar="NM_PER_RADPS",
        help="--dtc-law velocity only: joint-velocity gain, scalar or 6 values "
             "(default %(default)s)")
    parser.add_argument(
        "--dtc-ki", type=float, nargs="+", default=list(DEFAULT_KI),
        metavar="NM_PER_RAD",
        help="dtc integral gain on velocity error, scalar or 6 values. Off by "
             "default. kv alone cannot break static friction here -- the pose "
             "spring commands ~0.02 rad/s, so kv=60 yields only ~1.5-2.6 N.m "
             "against a UR5e breakaway of several N.m, and raising kv shakes "
             "the arm instead of moving it. Try 800 800 800 250 250 250 "
             "(default %(default)s)")
    parser.add_argument(
        "--dtc-i-max", type=float, nargs="+", default=list(DEFAULT_I_MAX),
        metavar="NM",
        help="cap on the --dtc-ki contribution, scalar or 6 values. Keep it "
             "well under --dtc-tau-max so a stuck joint pushes hard but "
             "bounded (default %(default)s)")
    parser.add_argument(
        "--dtc-tau-max", type=float, nargs="+", default=list(WRENCH_TAU_MAX),
        metavar="NM",
        help="dtc per-joint torque cap, scalar or 6 values. This cap IS the "
             "compliance (default %(default)s)")
    parser.add_argument(
        "--dtc-hz", type=float, default=DEFAULT_DTC_HZ,
        help="dtc host control-law rate; the on-robot latch applies tau at the "
             "UR's own 500 Hz regardless (default %(default)s)")
    parser.add_argument(
        "--dtc-stale-tau-ms", type=float, default=None, metavar="MS",
        help="dtc publish-side staleness bound: past this age since the last "
             "control tick the publisher decays tau towards zero (full float "
             "at --dtc-hz's stale_after_s). Must sit ABOVE the period the loop "
             "actually holds -- read dtmax in the field-follow line. Default is "
             "max(4/--dtc-hz, 40 ms).")
    parser.add_argument(
        "--dtc-log", default="auto", metavar="PATH",
        help="per-tick CSV of the dtc seam (phase, safety mode, wrench, "
             "spacemouse raw twist sm0..sm5 = [ang, lin], tau, q, qdot, "
             "per-joint clip flags). 'auto' writes next to the speedj logs, "
             "'none' disables. Streamed to disk, so a protective-stopped run "
             "keeps the record that explains it.")
    parser.add_argument(
        "--dtc-payload-mass", type=float, default=None,
        help="override the pendant's installation payload for dtc (kg). Default "
             "is to defer to the pendant, which is calibrated on this rig; "
             "tau=0 only means 'float' if gravity compensation is right, so the "
             "--dtc-hold-zero banner always says which value is in force.")
    parser.add_argument(
        "--dtc-payload-cog", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="centre of gravity for --dtc-payload-mass, in the tool frame (m)")
    parser.add_argument(
        "--dtc-hold-zero", action="store_true",
        help="dtc bring-up gate: upload the latch script, verify the directTorque "
             "rate and stream tau=0 so the arm floats, with viser fully live. The "
             "control law still runs and the EE frame must track the real arm; "
             "nothing actuates.")
    parser.add_argument(
        "--dtc-compliance-hold", action="store_true",
        help="DTC wrench-law push-return mode on top of normal teleop: "
             "--dtc-lambda, joint hold off, a raised tau_max on the big joints, "
             "and hold_at_tick=false so external pushes return to the last "
             "commanded pose when the stick is quiet. Enter -> align -> "
             "SpaceMouse field-follow unchanged; the idle re-base is only "
             "disabled once DTC is the live actuator. Requires "
             "--contact-backend dtc and --dtc-law wrench (the defaults).")
    parser.add_argument(
        "--dtc-push-test", action="store_true",
        help="Bench push test: --dtc-compliance-hold gains with no startup "
             "align, no input device and no field-follow. The arm holds the "
             "pose it is already in; push it and it should return. Implies "
             "--dtc-compliance-hold. Note the angular blade is NOT exercised "
             "(no commanded rotation => zero angular error), so this validates "
             "the linear spring only -- watch |W_ang| in the 1 Hz line. Pair "
             "with --no-viz for a headless run.")
    # --- contact lift-off fixes, independently switchable so a hardware run can
    # bisect which one actually matters. All three off reproduces the previous
    # behavior exactly: --press-loop legacy --no-free-along-normal
    # --no-snap-follow-to-surface
    parser.add_argument(
        "--press-loop", choices=["dtc", "legacy"], default="dtc",
        help="speedj press seek law (default: dtc = the in-contact-dtc "
             "ForcePIDController, with LPF/deadband on the measurement plus "
             "oscillation, spike and overshoot handling). 'legacy' is the bare "
             "clip(kp*(Fd-Fs)) PID on fz alone.")
    parser.add_argument(
        "--no-free-along-normal", dest="free_along_normal",
        action="store_false", default=True,
        help="do not project the pose spring's velocity off the surface normal "
             "in contact (stiffness[5,5]=0 alone, the previous behavior)")
    parser.add_argument(
        "--no-snap-follow-to-surface", dest="snap_follow_to_surface",
        action="store_false", default=True,
        help="let field-follow keep discarding vf.project()'s position, so the "
             "commanded point never re-snaps to the mesh (previous behavior)")
    parser.add_argument(
        "--ang-stiffness", type=float, default=None, metavar="N_M",
        help="pose-spring stiffness on the three angular axes; default is the "
             "same 2000 as every other axis. In contact the stick writes xyz "
             "only, so this alone decides how fast the tool turns toward the "
             "field: 2000 with the 0.02 rad --max-dR cap is 2.3 deg/s, which "
             "outruns a softened --xy-stiffness slide. Lower it to keep the "
             "two in proportion.")
    parser.add_argument(
        "--xy-stiffness", type=float, default=None, metavar="N_M",
        help="pose-spring stiffness on the lateral (tool X/Y) axes only; "
             "default is the same 2000 N/m as every other axis. Lowering it "
             "softens the shove into a tool held by friction, which is what "
             "the UR reports as 'could not follow path'. Max lateral speed "
             "scales with it: 8 mm/s at 2000, 2 mm/s at 500.")
    parser.add_argument(
        "--dls-damping", type=float, default=0.05, metavar="LAMBDA",
        help="damped-least-squares lambda for the jacobian inverse. Was a "
             "hardcoded 0.001, which peaks the inverse gain at 500x near a "
             "singularity and windmills one joint into a C153 position-"
             "deviation stop. 0.001 restores the old behavior.")
    parser.add_argument(
        "--no-per-joint-velocity", dest="per_joint_velocity",
        action="store_false", default=True,
        help="clamp only norm(qdot), letting a single joint carry the whole "
             "velocity budget (the previous behavior)")
    parser.add_argument(
        "--press-v-max", type=float, default=0.008, metavar="M_S",
        help="ceiling on the along-normal seek speed. Raise this only after "
             "the guards are letting the loop run -- check `spike %%` and "
             "`v_z == 0` in analyze_contact_log.py first, or you are just "
             "raising the speed of an impact.")
    parser.add_argument(
        "--approach-v", type=float, default=0.005, metavar="M_S",
        help="soft feed speed while |Fs| is below contact_threshold_N")
    parser.add_argument(
        "--spike-rate", type=float, default=None, metavar="N_S",
        help="override the dtc spike guard's rate threshold (default 250 N/s, "
             "set from this rig's measured tick-to-tick force spread)")
    parser.add_argument(
        "--v-slew", type=float, default=0.4, metavar="M_S2",
        help="press-velocity slew limit (m/s^2). At the old 0.05 a reversal "
             "took 0.32s, so the loop kept pressing ~1.3mm after it had "
             "decided to back off. Joint-space ramp is separately bounded by "
             "max_qddot.")
    parser.add_argument(
        "--contact-log", default="auto", metavar="PATH",
        help="per-tick CSV of the force loop (Fs, Fd, v_z, tare, dtc flags). "
             "'auto' writes next to the speedj logs, 'none' disables. Streamed "
             "to disk, so an aborted or protective-stopped run keeps its data.")
    parser.add_argument(
        "--no-force-tare", dest="force_tare", action="store_false", default=True,
        help="do not sample the free-space force offset when contact first "
             "latches. UR ActualTCPForce is not gravity compensated unless the "
             "payload is set, and the speedj backend never issues "
             "zero_ftsensor(), so without the tare the dtc loop trips "
             "wrong_sign/mag_overload on the tool weight and retracts instead "
             "of approaching")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    parser.add_argument(
        "--startup-align", dest="startup_align", action="store_true", default=True,
        help="after viser is up, slerp the live flange onto the field (one-shot)")
    parser.add_argument(
        "--no-startup-align", dest="startup_align", action="store_false",
        help="skip the one-shot field orientation align")
    parser.add_argument(
        "--press-kp", type=float, default=None, metavar="N_PER_N",
        help="--dtc-law wrench contact: proportional trim on the press, in N "
             "per N of force error. The commanded press is "
             "Fd + press_kp*(Fd - Fs), so 0 is pure feedforward (command Fd, "
             "get Fd -- which under a wrench law is already the right steady "
             "state) and this only covers sensor bias and a surface that "
             "moves. It is also the *approach* authority: with the tool clear "
             "of the part the command is Fd*(1+kp), which must clear breakaway "
             "(~18 N here) or the arm never moves at all. Default is "
             "max(0.5, breakaway/Fd), so Fd*(1+kp) = Fd + breakaway. Free in "
             "steady state -- in contact Fs -> F and F = Fd for any kp. "
             "Deliberately no integral term: an I on a stiction nonlinearity is "
             "a stick-slip amplifier.")
    parser.add_argument(
        "--press-auto-approach", action="store_true",
        help="--dtc-law wrench contact: close the Fd-Fs gap from the moment "
             "button 0 is pressed, instead of waiting for the operator to "
             "drive the tool into the surface with the pose spring. The press "
             "commands Fd + --press-kp*(Fd-Fs) immediately, so with the tool "
             "clear of the part it pushes Fd*(1+kp) N along the contact normal "
             "until the surface reacts. Off (the default) commands "
             "--press-approach-n (0) until Fs reaches --press-engage-frac*Fd. "
             "KNOWN RISK: a pure force at the TCP torques the wrist rather "
             "than translating the arm -- dtc_20260903_151838.csv put 23.2 N "
             "along the normal and moved j3 through 16.7 deg while j0/j4/j5 "
             "moved 0.0 -- so watch n_deg in the contact log and keep "
             "--press-max-n sane. The angular blade that resists exactly that "
             "rotation is much stiffer now (fn_ang 3.0 / zeta_ang 0.25 against "
             "the 2.0 / 0.15 of that run), which is what makes this worth "
             "re-testing")
    parser.add_argument(
        "--press-engage-frac", type=float, default=0.5, metavar="FRAC",
        help="fraction of --target-force that Fs must reach before contact "
             "mode hands the surface normal from the pose spring to the force "
             "loop. Until then the press commands --press-approach-n (0 by "
             "default) and the spring still owns the normal, so pressing "
             "button 0 while hovering does NOT start a press -- the operator "
             "drives the tool into the surface with the spring and the "
             "hand-off happens on measured force. 0.5 (the default) hands over "
             "as soon as the tool is touching; 1.0 waits until the target "
             "force is actually reached, which is what you want if a "
             "half-seated tool must not switch to tangential-only. Clamped to "
             "[0, 1] (default %(default)s)")
    parser.add_argument(
        "--press-release-frac", type=float, default=0.15, metavar="FRAC",
        help="fraction of --target-force below which contact mode hands the "
             "normal back to the pose spring. The gap to --press-engage-frac "
             "is deliberate hysteresis: Fs noise is ~0.3 N and chattering "
             "between the two regimes at 200 Hz would be a square wave into "
             "the surface. Clamped to [0, --press-engage-frac] "
             "(default %(default)s)")
    parser.add_argument(
        "--press-kp-hold", type=float, default=DEFAULT_PRESS_KP,
        metavar="N_PER_N",
        help="proportional trim once the tool is actually touching (Fs above "
             "half of --target-force). --press-kp is sized to break static "
             "friction and is far too strong to hold with: the same command "
             "that gets the tool moving keeps accelerating it into the part. "
             "The loop latches between the two on Fs with hysteresis "
             "(default %(default)s)")
    parser.add_argument(
        "--press-approach-n", type=float, default=0.0, metavar="N",
        help="press force commanded before the tool has touched (default 0: the "
             "pose spring does the whole approach). Non-zero re-creates the "
             "failure it was added to fix -- a pure force at the TCP is turned "
             "into wrist torque by J.T, and the wrist is the compliant "
             "direction, so the tool rotates out of orthogonality instead of "
             "the arm translating.")
    parser.add_argument(
        "--press-max-n", type=float, default=None, metavar="N",
        help="--dtc-law wrench contact: hard ceiling on the commanded press "
             "(default 3x --target-force). Under --dtc-compliance-hold the "
             "shoulder tau_max is 140 N.m, so a runaway trim would be "
             "delivered in full -- unlike the velocity overlay, which was "
             "bounded by --press-v-max and a position servo that could stall.")
    parser.add_argument(
        "--press-slew", type=float, default=200.0, metavar="N_PER_S",
        help="--dtc-law wrench contact: rate limit on the commanded press "
             "(N/s), the counterpart of --v-slew. A step in commanded force is "
             "a step in tau and this rig is lightly damped at ~15 Hz; 200 N/s "
             "reaches a 10 N target in 50 ms (default %(default)s)")
    parser.add_argument(
        "--follow-target-resend", dest="follow_target_resend",
        action=argparse.BooleanOptionalAction, default=True,
        help="Re-write the pose target to the objective on every field-follow "
             "call even when the stick is quiet and the value has not changed "
             "(default: on, which is how every log so far was recorded). That "
             "write happens at the main loop's fixed 50 Hz -- NOT --follow-hz, "
             "which gates nothing on an untouched stick -- and from a different "
             "thread than the 200 Hz control loop that reads it. The value is "
             "provably constant while quiet (map_target is pose-invariant), so "
             "--no-follow-target-resend is semantically a no-op and is the "
             "bisection for whether that cross-thread write is behind the 30 Hz "
             "burst on dtc_20260902_171136.csv.")
    parser.add_argument(
        "--stick-linear-scale", type=float, default=STICK_LINEAR_SCALE,
        metavar="M", help="metres of carrot advance per SpaceMouse sample at "
             "full deflection. The poll thread runs at 50 Hz, so this times 50 "
             "is the top carrot speed: %(default)s gives 75 mm/s. This is the "
             "sensitivity knob -- halve it for finer control, and note it is "
             "NOT the same as --max-lead-m, which bounds how far the carrot may "
             "get ahead of the arm (and so bounds the force and the lurch when "
             "friction lets go) (default %(default)s)")
    parser.add_argument(
        "--stick-angular-scale", type=float, default=STICK_ANGULAR_SCALE,
        metavar="RAD", help="radians of commanded rotation per SpaceMouse "
             "sample at full deflection (default %(default)s)")
    parser.add_argument(
        "--follow-max-pending-m", type=float, default=0.0, metavar="M",
        help="Bound (m) on the undrained SpaceMouse backlog in FieldFollow. "
             "0 (default) keeps the historical unbounded queue. The stick feeds "
             "add_body_linear at 50 Hz (up to 130 mm/s on a 3-axis diagonal) "
             "and the follow tick drains at most --follow-hz * 4 mm, which the "
             "50 ms gate against a ~50 Hz caller makes 66.7 mm/s -- so a held "
             "stick queues the surplus and the command falls behind the "
             "operator for as long as the push lasts (2.4 s of lag after a 5 s "
             "two-axis traverse). Read `pending=` in the spacemouse console "
             "line. Try 0.02 (= --max-lead-m): the arm stops lagging, at the "
             "cost of a fast stick no longer banking distance it can spend "
             "later. Not a fix for ringing -- the damping term never sees this "
             "queue.")
    parser.add_argument(
        "--loop-hz", type=float, default=50.0, metavar="HZ",
        help="rate of the main teleop loop that drives field-follow and the "
             "idle hold. Was hard-coded at 50. It matters because "
             "--follow-linear-approach can only charge its constant-velocity "
             "ramp on calls that do NOT earn a new commit: at --follow-hz 50 "
             "against a 50 Hz caller every call commits, the ramp branch never "
             "runs, and the flag is inert. Raise this above --follow-hz to get "
             "(loop_hz / follow_hz) ramp slices per commit, turning the "
             "carrot's staircase into a ramp. Costs CPU on the main thread "
             "only -- the control loop is a separate thread at --dtc-hz "
             "(default %(default)s)")
    parser.add_argument(
        "--follow-hz", type=float, default=20.0,
        help="after align, project() rate for stick field-follow (15–50 Hz). "
             "Not the RTDE/speedJ rate.")
    parser.add_argument(
        "--follow-rebase-on-live", dest="follow_rebase_on_live",
        action=argparse.BooleanOptionalAction, default=None,
        help="where each field-follow stick step starts from. On (the speedJ "
             "default) re-bases the carrot on the live TCP every tick, so the "
             "command is one drained step ahead -- a velocity command, which "
             "is what the position servo underneath speedJ wants. Under "
             "--dtc-law wrench that same carrot is force = --dtc-k-lin times "
             "one step: force proportional to stick *speed*, hard-capped at "
             "k_lin * max_step_m (12 N at the defaults) and zero when you hold "
             "the stick still against a stalled arm, which no gain can fix -- "
             "raising k_lin scales free-space speed by the same factor. Off "
             "accumulates instead, so a held stick builds error up to "
             "--max-lead-m and the spring makes a real sustained push that "
             "decays on its own as the arm catches up. Defaults off under "
             "--dtc-law wrench, on everywhere else")
    parser.add_argument(
        "--hold-field-orientation", dest="hold_field_orientation",
        action=argparse.BooleanOptionalAction, default=None,
        help="keep the commanded field orientation while the stick is quiet, "
             "re-basing only the *position* onto the live TCP. Off, both idle "
             "holds snap target := live pose, which erases the angular error "
             "too: measured on dtc_20260829_214019.csv the wrench was exactly "
             "zero on 42%% of ticks and the wrists moved 0.06 deg over the run "
             "while j0..j2 moved 8 deg -- quiet erased the error, the next "
             "stick sample handed back the full field pose in one tick, and the "
             "wrist tau cap threw the correction away, so the tool never "
             "converged onto the field. On, the wrist keeps working toward the "
             "field between stick motions and there is no step left to make. "
             "Note this means the arm turns with hands off the stick. Defaults "
             "on under --dtc-law wrench, off everywhere else (speedJ must stay "
             "bit-identical for the C153 A/B)")
    parser.add_argument(
        "--rearm-slerp-s", type=float, default=0.5, metavar="S",
        help="time constant for the commanded orientation's approach to the "
             "field goal. 0 snaps, which put |W_ang| 0 -> 4.25 N.m and tau on "
             "the 50 N.m rail in one tick. This is a rate on the command's own "
             "history, not a cap against live R -- the live-based cap is what "
             "pinned R_err at ~1.1 deg forever and was removed in 2eceb0f. "
             "Ignored without --hold-field-orientation, where the command is "
             "re-based on live every quiet tick and a ramp could only lag "
             "(default %(default)s)")
    parser.add_argument(
        "--follow-linear-approach", dest="follow_linear_approach",
        action=argparse.BooleanOptionalAction, default=False,
        help="ramp the linear carrot to its goal at a constant velocity "
             "spread over the gap to the next --follow-hz commit, charged "
             "every control tick, instead of snapping the full step "
             "(max_step_m) at the commit. Off (default) is the pre-fix "
             "behavior -- rotation already gets this treatment via "
             "--rearm-slerp-s, position never did, and a snapped step every "
             "--follow-hz is a periodic kick into the control loop (HANDOFF: "
             "\"the carrot is a 20 Hz staircase into a 124 Hz loop\"). "
             "Deliberately a constant-velocity ramp, not an exponential "
             "settle: the goal moves again before an exponential catches up, "
             "which only adds tracking lag. Untested on hardware -- bring up "
             "under --dtc-hold-zero first, then bare defaults, and read the "
             "oscillation section of analyze_dtc_log.py")
    parser.add_argument(
        "--follow-max-dr-rad", type=float, default=0.02, metavar="RAD",
        help="per-follow-tick cap on how far the field goal itself may turn, "
             "so one project() flip at a mesh edge cannot bank a large angular "
             "error for the next re-entry. Field->field only; 0 disables "
             "(default %(default)s). DO NOT TIGHTEN THIS to fight a "
             "curvature-localized ring -- tried on this rig (0.02 -> 0.01,"
             " dtc_20260831_183327.csv) and it produced a real self-sustaining"
             " limit cycle that was not there at 0.02: j3 reversed at "
             "+/-tau_max 12 times in 0.3 s at t=52.62 with the stick quiet "
             "and |e_lin| at zero, and the release-response step test found "
             "1 of 13 releases that grew instead of decaying, at 32.7 Hz "
             "against this loop's own f_90 = 1/(4T) = 33 Hz -- the exact "
             "delayed-damper signature documented on WRENCH_DAMPING_MAX_1_S, "
             "just reached through this rate cap instead of through D. A "
             "tightened rate limiter saturates more often and a saturating "
             "nonlinearity in the forward path adds phase lag of its own; "
             "that is plausibly what walked this loop's effective delay up "
             "to the f_90 line at the *same* D_lin (11.31) that ran clean "
             "with this cap at 0.02 (dtc_20260831_175638/180840.csv). "
             "RailOscillationGuard caught it here -- do not treat that as "
             "permission to repeat this, the guard is the last line, not "
             "the fix. A curvature-localized tracking gap is a --dtc-fn-lin"
             "/--dtc-damping-ratio question, not this one, and per the same "
             "run's own oscillation section: lower D, do not raise "
             "--dtc-damping-max, fn will not touch a 32 Hz runaway")
    parser.add_argument(
        "--max-lead-m", type=float, default=0.05, metavar="M",
        help="how far the field-follow carrot may run ahead of the live TCP. "
             "Under speedJ this is a path contract and 5 mm is right -- pass "
             "--max-lead-m 0.005 when running --contact-backend speedj. Under "
             "--dtc-law wrench it is also the *force* ceiling, because the "
             "pose error that becomes force cannot exceed it: 3000 N/m x 5 mm "
             "= 15 N, well under UR5e breakaway, and --dtc-err-clip-lin can "
             "never bind above it. It is a ceiling twice over -- _clamp_lead "
             "trims the carrot to it, and past it the catch-up branch rebases "
             "the carrot onto the live TCP, which collapses the error to zero "
             "exactly when the loop was closest to breaking stiction "
             "(default %(default)s)")
    parser.add_argument(
        "--align-delay", type=float, default=None,
        help="seconds to wait after viser starts before the align slerp "
             "(default: wait for Enter; orange ghost shows the snap first)")
    parser.add_argument(
        "--align-duration", type=float, default=4.0,
        help="seconds to slerp live orientation onto the field (no snap)")
    parser.add_argument(
        "--vention-height", type=float, default=DEFAULT_VENTION_HEIGHT,
        help="physical vention rail height in meters for both arms (hardware does "
             "not sense these joints; used for visualization FK).")
    parser.add_argument(
        "--left-vention-height", type=float, default=None,
        help="override --vention-height for the left rail only")
    parser.add_argument(
        "--right-vention-height", type=float, default=None,
        help="override --vention-height for the right rail only")
    parser.add_argument(
        "--speedj-plot", dest="speedj_plot", action="store_true",
        help="open the live speedJ qdot / qddot / dt / spectrum plot. OFF by "
             "default: it traces from inside the control loop and renders in "
             "another process, and the main teleop loop now runs at --loop-hz "
             "(200), so it is real compute on the thread that feeds the "
             "carrot. analyze_dtc_log.py reads the same data off the CSV "
             "afterwards, with more of it")
    parser.add_argument(
        "--no-speedj-plot", dest="speedj_plot", action="store_false",
        help="explicitly disable the live plot (already the default)")
    parser.add_argument(
        "--plot-size", type=float, nargs=2, metavar=("W", "H"),
        default=list(LIVE_FIGSIZE),
        help="live speedJ plot window size in inches (default %(default)s); it "
             "shares the screen with viser")
    parser.set_defaults(speedj_plot=False)

    # --- deterministic path experiment (no SpaceMouse) ---
    parser.add_argument(
        "--path-mode", action="store_true",
        help="deterministic path experiment: no input device at all. Pick a "
             "start and an end point on the object in viser, preview the plan "
             "with the orange ghost, then press Enter (or the EXECUTE button) "
             "and the arm walks approach -> plunge -> traverse -> retract. The "
             "field-follow carrot, the projection and the DTC wrench law are "
             "the same code the SpaceMouse drives; only the source of the steps "
             "changes, which is what isolates controller latency from input "
             "latency. Implies --press-feedforward and a geometric press latch")
    parser.add_argument(
        "--path-speed", type=float, default=0.02, metavar="M_S",
        help="free-space carrot speed along the plan (default %(default)s m/s)")
    parser.add_argument(
        "--path-contact-speed", type=float, default=0.01, metavar="M_S",
        help="carrot speed during plunge and traverse (default %(default)s m/s)")
    parser.add_argument(
        "--path-standoff", type=float, default=0.03, metavar="M",
        help="height above the surface the approach and retract waypoints sit "
             "at, along the surface normal (default %(default)s m)")
    parser.add_argument(
        "--path-ds", type=float, default=0.005, metavar="M",
        help="traverse waypoint spacing along the surface (default %(default)s m)")
    parser.add_argument(
        "--path-depth", type=float, default=0.002, metavar="M",
        help="how far BELOW the scanned surface the plunge and traverse are "
             "COMMANDED (default %(default)s m). The mesh is a laser scan "
             "placed by calibration, so a pose commanded exactly on the model "
             "leaves the tool touching lightly or hovering. This is the knob "
             "for 'the tool does not go far enough in' BEFORE the press takes "
             "over -- after the latch engages, free_along_normal removes the "
             "normal from the pose spring and --target-force owns it alone, so "
             "depth does not add force in contact. d_surf goes negative by "
             "this much when the plan is being followed")
    parser.add_argument(
        "--path-touch-n", type=float, default=0.0, metavar="N",
        help="require this much measured Fs before the plunge hands over "
             "(0 = geometry only, the default). Above 0 it does two things: "
             "the plunge keeps descending until Fs reaches it (up to "
             "--path-max-seek, then aborts), and the press latch requires it "
             "as well as the distance. This is the one gate that closes the "
             "scan-vs-part gap, because geometry alone cannot tell 'touching' "
             "from 'the model is 3 mm off'. A threshold is a fair use of a "
             "noisy sensor; regulating on it is not")
    parser.add_argument(
        "--path-max-seek", type=float, default=0.01, metavar="M",
        help="how far past the planned depth --path-touch-n may seek before "
             "giving up and aborting the run (default %(default)s m)")
    parser.add_argument(
        "--path-engage-gap", type=float, default=0.002, metavar="M",
        help="press latch ON at this signed distance from the planned surface "
             "point, measured along its normal (default %(default)s m). This "
             "replaces the Fs latch: both terms come from FK and the plan, so "
             "it does not chatter on a noisy F/T reading near the target")
    parser.add_argument(
        "--path-release-gap", type=float, default=0.005, metavar="M",
        help="press latch OFF above this distance. Must exceed "
             "--path-engage-gap; the gap is the hysteresis (default %(default)s m)")
    parser.add_argument(
        "--press-feedforward", dest="press_feedforward", action="store_true",
        help="command a flat --target-force along the contact normal while the "
             "latch says the tool is on the part, instead of the "
             "Fd + kp*(Fd-Fs) trim. The trim is only worth having if the "
             "measurement is better than the bias it removes; getActualTCPForce "
             "on this rig carries several N of spread at a 15 N target, so the "
             "trim injects that noise as torque. Fs is still logged and is "
             "still the overload guard (--press-overload-n)")
    parser.add_argument(
        "--no-press-feedforward", dest="press_feedforward",
        action="store_false",
        help="keep the closed-loop trim even under --path-mode")
    parser.add_argument(
        "--press-overload-n", type=float, default=None, metavar="N",
        help="|Fs| at or above this commands 0 N and stays there until |Fs| "
             "falls under half of it. Magnitude, not signed, so an inverted "
             "--force-sign still trips it. Default 2x --target-force")
    parser.add_argument(
        "--tool-flip", action="store_true",
        help="start in tool orientation B: the whole field rolled 180 deg "
             "about tool Z, so the chamfer faces the other way down the "
             "stringer. 'F' toggles it live (free space only). Tool +Z, the "
             "surface normal and every distance the press latch measures are "
             "identical in A and B -- only the roll changes")
    parser.add_argument(
        "--path-contact-at", choices=("plunge", "execute"), default="plunge",
        help="when contact mode is entered. 'plunge' (default) keeps the whole "
             "approach to the start point in plain approach mode and switches "
             "at the standoff point, where the tool is still --path-standoff "
             "clear of the surface so the force tare is still taken in free "
             "space. 'execute' switches the moment the run starts, which is "
             "what the first version did -- the press commands 0 N either way, "
             "but the 200 Hz force thread shares the RTDE socket for the whole "
             "approach")
    parser.add_argument(
        "--path-hold-frac", type=float, default=PATH_LAG_HOLD_FRAC,
        metavar="FRAC",
        help="fraction of --max-lead-m at which the plan waits for the arm "
             "(default %(default)s). Read it as a FORCE cap, because that is "
             "what it is: the pose spring makes force out of lead, so holding "
             "at 0.6 of a 30 mm clamp delivered 13 N where 0.8 delivered 30 N "
             "on the same task. 0 disables the hold entirely and lets the lead "
             "clamp be the only bound")
    parser.add_argument(
        "--path-stall-abort-s", type=float, default=2.0, metavar="S",
        help="window over which a stall is judged (0 disables). A stalled pose "
             "spring is not harmless: it keeps commanding Lambda*K*err, so a "
             "jammed tool is ground into the part at full authority for as "
             "long as the plan runs (default %(default)s s)")
    parser.add_argument(
        "--path-stall-mm", type=float, default=1.0, metavar="MM",
        help="TCP travel below this over --path-stall-abort-s, while the arm "
             "is also behind the command, counts as stalled (default "
             "%(default)s mm). Displacement over a window, not speed: the "
             "motion is stick-slip, so an instantaneous speed reads zero most "
             "of the time even on an arm that is making progress. Measured "
             "idle noise floor on this rig is 0.18 mm/s p95")
    parser.add_argument(
        "--path-plot", dest="path_plot", action="store_true",
        help="after every executed path, render commanded vs actual joints and "
             "tool pose to a PNG next to the dtc log (examples/plot_path_run.py "
             "in a subprocess). On by default under --path-mode")
    parser.add_argument(
        "--no-path-plot", dest="path_plot", action="store_false",
        help="skip the post-run plot")
    parser.set_defaults(press_feedforward=None, path_plot=True)
    return parser


# Fraction of --max-lead-m at which the plan stops advancing and waits for the
# arm. Below 1.0 on purpose: the clamp itself is the hard stop, and a plan that
# only notices at the stop has already wound the command up against it.
#
# **This is a force cap, not just a pacing knob, and lowering it cost a working
# approach.** The lead IS the force: dtc_20260906_192810.csv approached the
# surface with |e| 22.3 mm and |W_lin| 30.1 N, and after this went 0.8 -> 0.6
# the same task ran at 18 mm and 13 N and could only creep. Default is back at
# the value that worked; --path-hold-frac overrides it.
PATH_LAG_HOLD_FRAC = 0.8


def path_mode_on(args) -> bool:
    return bool(getattr(args, "path_mode", False))


def resolve_press_feedforward(args) -> bool:
    """Feed-forward press: explicit flag wins, otherwise on under --path-mode."""
    ff = getattr(args, "press_feedforward", None)
    if ff is None:
        return path_mode_on(args)
    return bool(ff)


def configure_path_mode(args) -> None:
    """Validate --path-mode and apply its side effects."""
    if not path_mode_on(args):
        return
    if args.contact_backend != "dtc" or args.dtc_law != "wrench":
        raise SystemExit(
            "--path-mode requires --contact-backend dtc --dtc-law wrench "
            "(the defaults): the press is a force, not a velocity seek")
    if dtc_push_test_mode(args):
        raise SystemExit("--path-mode and --dtc-push-test are both input "
                         "sources; pick one")
    if args.path_release_gap < args.path_engage_gap:
        raise SystemExit("--path-release-gap must be >= --path-engage-gap "
                         "(the gap is the latch hysteresis)")
    print(
        f"path mode: no input device. Fd={args.target_force:.1f}N "
        f"{'feedforward' if resolve_press_feedforward(args) else 'closed loop'}, "
        f"latch = plan phase AND distance "
        f"(on <= {args.path_engage_gap * 1000:.0f}mm, "
        f"off > {args.path_release_gap * 1000:.0f}mm)"
        + (f" AND Fs >= {args.path_touch_n:.1f}N" if args.path_touch_n > 0
           else "")
        + f", depth {args.path_depth * 1000:.1f}mm, "
        f"speeds {args.path_speed * 1000:.0f}/{args.path_contact_speed * 1000:.0f} mm/s "
        f"free/contact.",
        flush=True,
    )


def configure_dtc_compliance_hold(args) -> None:
    """Validate and apply --dtc-compliance-hold / --dtc-push-test side effects."""
    if dtc_push_test_mode(args):
        args.dtc_compliance_hold = True
    if not dtc_compliance_hold_mode(args):
        return
    if args.contact_backend != "dtc":
        raise SystemExit("--dtc-compliance-hold requires --contact-backend dtc")
    if args.dtc_law != "wrench":
        raise SystemExit("--dtc-compliance-hold requires --dtc-law wrench")
    if args.dtc_hold_zero:
        raise SystemExit(
            "--dtc-compliance-hold and --dtc-hold-zero are mutually exclusive")
    if not args.dtc_lambda:
        # The bench push test is the one place --no-dtc-lambda has to be
        # reachable, because the guard's premise is exactly what it exists to
        # test. diffusion_policy's rtde_interpolation_controller.py runs this
        # arm with a plain 1000 N/m / 50 N.m/rad task spring, NO Lambda, and
        # shows no dead band -- while every gain this loop delivers is stiffer
        # (1928 N/m and 138 N.m/rad measured at rest over 18 loaded rests). So
        # "without Lambda the wrench maps unevenly" is a claim with a
        # counterexample, and refusing the A/B leaves the dead band unbisected.
        #
        # Only under --dtc-push-test: no field, no align, no device, the target
        # snapped to the live pose, and an operator standing at the arm. Under
        # --dtc-compliance-hold on live teleop it stays a hard error, because
        # there the field commands orientation continuously and an uneven map
        # is a real hazard rather than a bench observation.
        if not dtc_push_test_mode(args):
            raise SystemExit(
                "--dtc-compliance-hold requires --dtc-lambda (inertia shaping). "
                "Without Lambda the same task wrench maps unevenly across joints "
                "(wrist moves, shoulder takes force); see analyze_dtc_mobility.py."
                "  --dtc-push-test allows it, for the reference-controller A/B.")
        print("dtc: --no-dtc-lambda under --dtc-push-test -- inertia shaping OFF. "
              "K and D are now N/m and N.s/m (and N.m/rad, N.m.s/rad), NOT 1/s^2 "
              "and 1/s, so --dtc-fn-* do not apply and --dtc-damping-max does "
              "not cap anything. Set --dtc-k-lin/--dtc-k-ang explicitly.",
              flush=True)
    else:
        args.dtc_lambda = True
    # fn_ang defaults to fn_lin under Lambda via resolve_wrench_fn_ang when unset.
    # Joint-space hold adds extra stiffness on selected joints (default wrist 3)
    # on top of the task wrench -- it defeats an equal-compliance push test.
    args.dtc_joint_hold_kp = 0.0
    args.dtc_joint_hold_kd = 0.0
    if list(args.dtc_tau_max) == list(WRENCH_TAU_MAX):
        args.dtc_tau_max = list(COMPLIANCE_HOLD_TAU_MAX)
    if args.dtc_damping_ratio is None:
        args.dtc_damping_ratio = COMPLIANCE_HOLD_ZETA_LIN
    if args.dtc_damping_ratio_ang is None:
        args.dtc_damping_ratio_ang = COMPLIANCE_HOLD_ZETA_ANG
    # Teleop idle hold re-bases the spring target onto the live pose every
    # tick so field-follow does not fight a stale carrot. For a push test the
    # target must stay at the handoff pose instead, or releasing the arm leaves
    # error ~= 0 and |W| ~= 0 with no return motion.
    args.dtc_hold_at_tick = False
    # Compliance hold is the *stronger* policy: the target never moves after
    # handoff, so a push always meets the spring. --dtc-hold-on-release would
    # re-introduce a snap this mode exists to remove. Say so rather than
    # silently overriding -- this repo has been bitten three times by a cap
    # that quietly changed a gain and left only scrollback behind.
    if args.dtc_hold_on_release:
        print("\033[93mdtc: --dtc-hold-on-release ignored under "
              "--dtc-compliance-hold -- that mode already freezes the target "
              "at handoff and never re-snaps, which is strictly stronger.\033[0m",
              flush=True)
    args.dtc_hold_on_release = False
    if dtc_push_test_mode(args):
        # Hold wherever the arm already is. Align would drive it somewhere
        # first, and its speedJ leg is not what this run is measuring.
        args.startup_align = False


def install_termination_guards():
    """Make every ordinary way of ending this script run the teardown.

    Under ``--contact-backend dtc`` the arm is held by *our* torque stream and
    nothing else: tau = 0 means float, because gravity is compensated
    on-controller. So a process that stops running without tearing down leaves a
    back-drivable arm -- observed on hardware as "I pushed it and the elbow
    folded after the script had stopped".

    - **SIGTSTP (Ctrl-Z) is the dangerous one and is refused outright.** It
      suspends the process without running any handler or ``finally``, so the
      tau streamer stops republishing, the on-robot latch decays tau to zero
      after ``--dtc-stale-tau-ms``, and the arm floats with the script apparently
      "stopped". There is no safe way to suspend this program; Ctrl-C instead.
    - **SIGTERM / SIGHUP** (kill, terminal closed) raise KeyboardInterrupt so
      the existing ``finally`` runs -- zero tau, servoJ hold, stopScript,
      disconnect.

    SIGKILL and a hard crash cannot be caught. The on-robot latch's stale decay
    is the backstop there, and it lands on float, so the pendant's stop is the
    real last line of defence. Say so once at startup rather than implying the
    arm is safe.
    """
    def _refuse_suspend(signum, frame):
        del signum, frame
        print("\033[91mCtrl-Z ignored: suspending would leave the arm floating "
              "at tau=0 with no teardown. Use Ctrl-C.\033[0m", flush=True)

    def _terminate(signum, frame):
        del frame
        raise KeyboardInterrupt(f"signal {signum}")

    for name, handler in (("SIGTSTP", _refuse_suspend),
                          ("SIGTERM", _terminate),
                          ("SIGHUP", _terminate)):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Not the main thread, or the platform refuses it. Not fatal.
            pass


def main():
    args = build_arg_parser().parse_args()
    # Snapshot the terminal before anything can put it in cbreak, so the
    # teardown can restore it even if the key thread never got that far.
    try:
        tty_saved = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        tty_saved = None
    install_termination_guards()
    configure_dtc_compliance_hold(args)
    configure_path_mode(args)
    if args.check_spacemouse:
        paths = list_spacemouse_paths()
        if not paths:
            print("spacemouse: none connected", flush=True)
            raise SystemExit(1)
        print(f"spacemouse: connected: {format_spacemouse_paths()}", flush=True)
        path = resolve_spacemouse_path(
            args.spacemouse_path, device_index=args.spacemouse_index)
        print(f"spacemouse: opening {path} — move the cap for 3s…", flush=True)
        saw_motion, peak = verify_spacemouse_connection(path, wait_s=3.0)
        if saw_motion:
            print(f"spacemouse: OK (peak axis={peak:.3f})", flush=True)
            raise SystemExit(0)
        print(
            f"spacemouse: no motion detected (peak={peak:.3f}). "
            "Wrong device? Try --spacemouse-path or --spacemouse-index.",
            flush=True,
        )
        raise SystemExit(1)

    print(
        f"contact: backend={args.contact_backend}  "
        f"force_sign={args.force_sign:g}  "
        f"Fd={args.force_sign * args.target_force:+.1f}N",
        flush=True,
    )

    with open(args.hardware_config) as f:
        hardware_config = yaml.safe_load(f)

    if args.bring_up:
        for control_group, entry in hardware_config.items():
            _dashboard_bring_up(entry["robot_ip"])

    task_group = "left_ur5e" if args.task_space == "left_ee" else "right_ur5e"
    robot_ip = hardware_config[task_group]["robot_ip"]

    interface = ri.RobotInterface(ri.Configuration(MODEL, "hardware", args.hardware_config))
    interface.initialize()

    # Built before anything reads kinematics, because DTC's handoff calls
    # interface.shutdown() and that takes gafropy's System with it. The backend
    # loads its own driverless System from the model file (no driver, no RTDE
    # client), so viser, the chamfer visual and every FK below keep working
    # across the handoff. connect() -- the part that touches RTDE -- happens
    # later, in start_control_and_device.
    dtc_backend = None
    if args.contact_backend == "dtc":
        dtc_backend = DirectTorqueBackend(
            robot_ip=robot_ip,
            model_file=MODEL,
            task_space_name=args.task_space,
            kv=args.dtc_kv, tau_max=args.dtc_tau_max,
            ki=args.dtc_ki, i_max=args.dtc_i_max,
            law_kind=args.dtc_law,
            friction_comp=args.dtc_friction_comp,
            use_lambda=args.dtc_lambda,
            lambda_max_lin_kg=args.dtc_lambda_max_lin,
            hz=args.dtc_hz, hold_zero=args.dtc_hold_zero,
            hold_at_tick=args.dtc_hold_at_tick,
            hold_on_release=args.dtc_hold_on_release,
            joint_hold_kp=args.dtc_joint_hold_kp,
            joint_hold_kd=args.dtc_joint_hold_kd,
            joint_hold_joints=args.dtc_joint_hold_joints,
            stiction_assist_nm=args.dtc_stiction_assist_nm,
            stiction_max_nm=args.dtc_stiction_max_nm,
            stale_tau_s=(
                None if args.dtc_stale_tau_ms is None
                else args.dtc_stale_tau_ms / 1000.0),
            rail_osc_flips=args.dtc_rail_osc_flips,
            rail_osc_hold_s=args.dtc_rail_osc_hold_s,
            log_path=_contact_log_path(args.dtc_log, prefix="dtc"),
            gain_note_extra=_wrench_gain_note(args),
            payload_mass=args.dtc_payload_mass,
            payload_cog=args.dtc_payload_cog,
        )
    kin_system = (
        interface.get_system() if dtc_backend is None else dtc_backend.get_system())

    def on_dtc_now():
        """True once the DTC backend is the live actuator (post-handoff)."""
        return dtc_backend is not None and interface is dtc_backend

    left_vention = args.left_vention_height if args.left_vention_height is not None else args.vention_height
    right_vention = args.right_vention_height if args.right_vention_height is not None else args.vention_height
    print(f"hardware does not report vention rails; drawing them at "
          f"left={left_vention:.3f}m right={right_vention:.3f}m")

    def q_with_vention():
        # `interface` is rebound to the DTC backend at handoff; it duck-types
        # get_state() so this and every caller below need no branch.
        return apply_vention_heights(
            kin_system, interface.get_state().get_position(),
            left_vention, right_vention)

    # Control task_space must match the live RTDE owner. Using the driverless
    # DTC System here while interface.control() still runs gafropy made align
    # issue qdot against the wrong joints -- arm sat still. Switch to the DTC
    # task_space only inside enter_direct_torque.
    task_space = interface.get_task_space(args.task_space)
    if task_space is None:
        raise RuntimeError(f"task space '{args.task_space}' not found in {MODEL}")

    joint_idx = np.array(task_space.get_joint_indices())
    live_position = interface.get_state().get_position()[joint_idx]
    home_ee_motor = task_space.compute_ee_motor(live_position)

    viz_position = q_with_vention()[joint_idx]
    viz_ee_motor = task_space.compute_ee_motor(viz_position)
    viz_translator = viz_ee_motor.get_translator()
    home_xyz = np.array(
        [viz_translator.x(), viz_translator.y(), viz_translator.z()])
    rail_height = left_vention if args.task_space == "left_ee" else right_vention
    vention_offset = float(viz_position[0]) - rail_height
    table_reference_xyz = home_xyz - np.array([0.0, 0.0, vention_offset])

    object_points_raw = load_points(args.object)
    if is_laser_left_base_object(args.object):
        # Calibrated Keyence merge: left_base → vention → tag_0. Do not PCA-rest
        # or rewrite into right_base (that rotates the layup ~90°).
        placement = place_laser_object(args.object)
        object_surface = reconstruct_mesh_from_points(placement.points_vention)
        n_poisson = len(object_surface.vertices)
        object_surface = clip_mesh_to_scan_points(
            object_surface, placement.points_vention)
        orient_surface_toward(object_surface, np.array([0.0, 0.0, 1.0]))
        if args.normal_orientation < 0:
            object_surface.invert()
        print(
            "laser overlay: merged.ply left_base → vention "
            f"(+{placement.T_left_base_vention[:3, 3].round(3).tolist()} mount) "
            f"→ tag_0; T_tag0_rtde.t={placement.T_tag0_rtde[:3, 3].round(3).tolist()} "
            "(right_base). Cloud is not copied into the right arm. "
            f"scan surface {len(object_surface.vertices)}/{n_poisson} verts "
            "(Poisson underside dropped, normals toward +Z). "
            f"calibrated offsets applied: laser="
            f"{(1e3 * load_t_vention(DEFAULT_LASER_OFFSET)).round(1).tolist()}mm "
            f"object={(1e3 * load_t_vention(DEFAULT_TAG0_OFFSET)).round(1).tolist()}mm.",
            flush=True,
        )
    else:
        rotation = compute_resting_orientation(object_points_raw)
        translation = table_placement_translation(
            object_points_raw, rotation, table_reference_xyz)
        object_surface = load_object_mesh(
            args.object, rotation=rotation, translation=translation,
            normal_orientation=args.normal_orientation)

    key_direction = np.asarray(home_ee_motor.get_rotor().to_rotation_matrix())[:, 0]
    vf = OrientationField(object_surface, key_direction=key_direction)
    field_preview = FieldAlignPreview(vf)
    follow = FieldFollow(
        vf, hz=args.follow_hz, project_in_process=True, mesh=object_surface,
        max_lead_m=args.max_lead_m, max_step_m=0.004,
        max_pending_m=args.follow_max_pending_m,
        max_dR_rad=args.follow_max_dr_rad,
        # The ramp only means anything where the command persists: with
        # hold_field_orientation off the commanded R is collapsed onto live R
        # on every quiet/catch-up call, so a ramp would restart from live every
        # time and pin the command permanently behind the field -- which is the
        # ~1.1 deg lag 2eceb0f removed.
        rearm_slerp_s=(args.rearm_slerp_s
                       if _hold_field_orientation(args) else 0.0),
        hold_field_orientation=_hold_field_orientation(args),
        linear_approach=args.follow_linear_approach,
        snap_to_surface=args.snap_follow_to_surface,
        # speedJ keeps the re-based carrot: it is a velocity command and the
        # position servo underneath wants exactly that. The wrench law turns
        # the same carrot into force = k_lin * (one drained step), which is
        # force proportional to stick *speed* and capped at
        # k_lin * max_step_m = 12 N at the defaults -- measured max 14.4 N on
        # dtc_20260828_182701.csv with --max-lead-m 0.02, which never bound.
        rebase_on_live=(args.follow_rebase_on_live
                        if args.follow_rebase_on_live is not None
                        else not (args.contact_backend == "dtc"
                                  and args.dtc_law == "wrench")))
    follow_log_t = {"t": float("-inf")}
    stick = {
        "t": float("-inf"), "t_nz": float("-inf"),
        "lin": np.zeros(3), "n": 0, "nz": 0,
    }

    # SpeedJ plant for align (and for --contact-backend speedj). DTC wrench
    # gains are 1/s^2 under Lambda and are applied only in enter_direct_torque.
    approach_controller = _make_approach_controller(
        interface, task_space, home_ee_motor,
        free_along_normal=args.free_along_normal,
        dls_damping=args.dls_damping,
        per_joint_velocity=args.per_joint_velocity,
        xy_stiffness=args.xy_stiffness,
        ang_stiffness=args.ang_stiffness,
        **(_wrench_law_kwargs(args) if args.contact_backend != "dtc" else {}))

    # --- speedj/dtc contact: the same pose spring + Z overlay. Only the
    # actuation seam under it differs, so the contact controller, the force
    # reader and the contact CSV are built identically for both. ---
    contact_controller = None
    measured_fz_fn = None
    safety_mode_fn = None
    measured_force_fn = None
    measured_wrench_fn = None
    close_tcp_force = None
    if args.contact_backend in ("speedj", "dtc"):
        contact_stiffness = np.diag([400.0] * 5 + [0.0])
        z_damping = 200.0
        contact_damping = np.diag([40.0] * 5 + [z_damping])
        contact_config = ga.control.VelocityAdmittanceControllerConfig()
        contact_config.dt = 0.01
        contact_config.max_linear_speed = 0.005
        contact_config.max_angular_speed = 0.15
        contact_config.admittance = ga.control.AdmittanceConfig(
            inertia=1.0 * np.eye(6), damping=contact_damping, stiffness=contact_stiffness)
        try:
            read_tcp_force, close_tcp_force = connect_tcp_force_reader(robot_ip)
            measured_fz_fn = lambda: fz_from_tcp_force(read_tcp_force())
            # The dtc loop resolves force along the field normal, not tool Z, so
            # it needs all three linear components.
            measured_force_fn = lambda: force_from_tcp_force(read_tcp_force())
            # All six axes, raw, for the log only. The press only ever needed
            # the linear part, so the torques were never recorded -- and "was
            # the tool being twisted while it slid" is exactly what a traverse
            # raises. plot_path_run.py draws them.
            measured_wrench_fn = lambda: list(read_tcp_force())
            safety_mode_fn = getattr(read_tcp_force, "safety_mode", None)
            if dtc_backend is not None:
                # Losing speedJ loses the UR's path supervisor, so this gate is
                # the only thing left that stops torque during a protective stop.
                dtc_backend.set_safety_mode_fn(safety_mode_fn)
                # Share this one receive socket rather than opening a second on
                # 30004: most controllers allow a single RTDE client, and DTC
                # already needs the control client on 30003.
                dtc_backend.set_receive(getattr(read_tcp_force, "receive", None))
            print(f"contact FT: RTDE ActualTCPForce from {robot_ip}", flush=True)
        except Exception as exc:
            print(
                f"\033[93mWARNING: cannot open RTDE ActualTCPForce on {robot_ip} "
                f"({exc}). {args.contact_backend} contact will refuse without Fz, "
                "and DTC loses its safety-mode gate.\033[0m",
                flush=True,
            )
        contact_controller = ContactTeleoperationController(
            vf, contact_config, target_force=args.target_force,
            kp_force=0.001, ki_force=0.0003,
            v_approach=args.approach_v, max_v_hold=args.press_v_max,
            z_damping=z_damping,
            v_slew_mps2=args.v_slew, lost_s=1.0, sit_s=2.0,
            spike_dF_dt_N_s=args.spike_rate,
            safety_mode_fn=safety_mode_fn,
            force_sign=args.force_sign,
            press_loop=args.press_loop,
            force_tare=args.force_tare,
            press_kp_n_per_n=resolve_press_kp(args),
            press_kp_hold=args.press_kp_hold,
            press_auto_approach=args.press_auto_approach,
            press_engage_frac=args.press_engage_frac,
            press_release_frac=args.press_release_frac,
            press_approach_n=args.press_approach_n,
            press_max_n=resolve_press_max_n(args),
            press_slew_n_s=args.press_slew,
            press_feedforward=resolve_press_feedforward(args),
            press_overload_n=args.press_overload_n,
            measured_fz_fn=measured_fz_fn,
            measured_force_fn=measured_force_fn,
            measured_wrench_fn=measured_wrench_fn)
        contact_controller.set_contact_log(_contact_log_path(args.contact_log))
        contact_controller.set_conditioning_fn(approach_controller.conditioning)
        contact_controller.set_task_space(task_space)
        contact_controller.set_interface(interface)
        contact_controller.set_objective(
            ga.control.SingleArmLineControlObjective(
                home_ee_motor.transform_line(ga.Line.Z())))

    force_session = None
    if args.contact_backend == "force_mode":
        force_session = ForceModeContactSession(
            robot_ip,
            target_force=args.target_force,
            force_sign=args.force_sign,
            force_ramp_s=2.0,
            zero_ft_sensor=args.zero_ft_sensor,
        )

    mode = {"active": "approach"}
    handoff_busy = {"busy": False}
    pending_contact = {"on": False}
    device = None
    speedj_trace = None
    viz = None
    robot_viz = None
    eeviz = None
    sim_probe = {"p": None}
    sim_teleop = {"on": False}
    align = (
        StartupAlign(delay_s=args.align_delay, duration_s=args.align_duration)
        if args.startup_align else None)
    align_phase = {"name": None}

    def live_ee_motor():
        position = interface.get_state().get_position()[joint_idx]
        return task_space.compute_ee_motor(position)

    def chamfer_flange_from_q(q):
        fk = kin_system.compute_forward_kinematics(np.asarray(q, dtype=float))
        T_wrist = np.asarray(
            fk.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
        return chamfer_world_from_wrist3(T_wrist)

    def snap_target_flange(T_flange_viz):
        """Field orientation at the live TCP, with the tip held (same as sim ``t``)."""
        p_tcp = live_tcp_xyz_from_flange(T_flange_viz)
        R_field = vf.approach_basis_at(p_tcp)
        return R_field, ghost_flange_world(T_flange_viz, R_field, p_tcp, 1.0)

    def run_field_follow(T_flange_viz, T_ee):
        if follow.xyz is None:
            p_tcp = live_tcp_xyz_from_flange(T_flange_viz)
            follow.seed(p_tcp, vf.approach_basis_at(p_tcp))
        T_flange_hw = chamfer_flange_from_q(interface.get_state().get_position())
        T_ee_m = np.asarray(T_ee, dtype=float).reshape(4, 4)

        def map_target(motor):
            return motor_from_matrix(field_tcp_to_ee(
                matrix_from_motor(motor),
                T_flange_hw=T_flange_hw,
                T_flange_viz=T_flange_viz,
                T_ee_hw=T_ee_m))

        T_prop = np.eye(4)
        T_prop[:3, 3] = follow.xyz
        now = time.monotonic()
        lin_now = follow_linear(
            stick["lin"], ignore_z=mode["active"] == "contact")
        quiet = float(np.linalg.norm(lin_now)) <= 1e-12
        if path_mode_on(args):
            # There is no stick, so the stick is always "quiet" -- and the quiet
            # branch of drive_field_follow calls set_idle(clear_pending=True),
            # which would throw away the very steps the plan just queued. The
            # plan running IS the input being active.
            quiet = path_session is None or not path_session.running
        tick = drive_field_follow(
            follow, approach_controller, now, motor_from_matrix(T_prop),
            live_xyz=live_tcp_xyz_from_flange(T_flange_viz),
            live_R=T_flange_viz[:3, :3],
            live_ee_xyz=T_ee_m[:3, 3],
            map_target=map_target,
            stick_quiet=quiet,
            # Tangent-only follows the *press latch*, not the mode. Gating it
            # on the mode was a deadlock: entering contact stripped the normal
            # out of every stick step, so the operator could not drive into the
            # surface, so Fs never rose, so the press (which commands
            # --press-approach-n = 0 until Fs >= engage_frac*Fd) never engaged.
            # The tool hovered and pushing the stick down did nothing.
            # Same hand-off as free_along_normal: the spring and the stick own
            # the normal on the way in, the force loop owns it once there.
            tangent=(mode["active"] == "contact"
                     and approach_controller.press_engaged),
            rebase_on_idle=follow_rebase_on_idle(args, on_dtc=on_dtc_now()),
            resend_target=args.follow_target_resend)
        if not approach_controller._input_active and follow_rebase_on_idle(
                args, on_dtc=on_dtc_now()):
            approach_controller.hold_live_pose()
        # Publish the surface normal both loops need, in the *controller* world
        # (map_target has already taken the field pose through field_tcp_to_ee,
        # so its Z column and T_ee's rotation share a frame). The force thread
        # must not query vf itself -- project() is why follow runs in a helper
        # process in the first place.
        if follow.last_target is not None:
            T_cmd = matrix_from_motor(map_target(follow.last_target))
            n_base = T_cmd[:3, 2]
            R_tool = T_ee_m[:3, :3]
            approach_controller.set_contact_normal(n_base, R_tool)
            # Same frame, same tick: the idle hold keeps this orientation and
            # re-bases only the position, so the hold target and the follow
            # target are the same rotation by construction.
            if _hold_field_orientation(args):
                approach_controller.set_hold_rotation(T_cmd[:3, :3])
            if contact_controller is not None:
                contact_controller.set_contact_normal(n_base, R_tool)
        if (now - follow_log_t["t"]) >= 1.0:
            follow_log_t["t"] = now
            p_live = live_tcp_xyz_from_flange(T_flange_viz)
            p_cmd = follow.xyz if follow.xyz is not None else p_live
            R_cmd = T_flange_viz[:3, :3]
            if follow.last_target is not None:
                T_cmd = matrix_from_motor(follow.last_target)
                R_cmd = T_cmd[:3, :3]
            lin = np.asarray(stick["lin"], dtype=float).reshape(3)
            age_ms = (
                1e9 if not np.isfinite(stick["t"])
                else (now - float(stick["t"])) * 1000.0)
            # Under DTC the pose spring can look perfectly healthy while no
            # torque reaches the joints -- tau = kv * (qdot_cmd - qdot_meas) is
            # all the authority there is, and it lands below breakaway friction
            # long before anything above the seam complains. Print it here so a
            # stalled arm and a healthy one do not read identically.
            actuation = None
            if dtc_backend is not None:
                st = dtc_backend.stats()
                if st.get("law") == "wrench":
                    # w_ang and hz_meas are here because the console line
                    # without them read as healthy through the run that ended in
                    # an emergency stop: |W_lin| was 5.8 N while tau railed at
                    # 30 N.m, and the loop was at a third of its nominal rate,
                    # which is the delay the damping term is fed back through.
                    # `stale` here is the PUBLISH-side count off the streamer,
                    # not the backend's dead-state-read counter: the line used
                    # to print the latter as 0 through a run where the former
                    # was chopping tau 19 times per 5 s. Both are shown, and the
                    # clipped joint is named -- a railed wrist and a railed
                    # shoulder are different faults with the same scalar count.
                    pub = st.get("streamer") or {}
                    worst = st.get("clip_joint_worst", -1)
                    actuation = (
                        f"|W|={st.get('w_lin', 0.0):.1f}N "
                        f"|Wa|={st.get('w_ang', 0.0):.2f}Nm "
                        f"tau_peak={st.get('tau_peak', 0.0):.2f}Nm "
                        f"clip={st.get('n_tau_clipped', 0)}"
                        f"{'' if worst < 0 else f'@j{worst}'} "
                        # Times the loop cut itself out for a rail-to-rail
                        # limit cycle. Anything but 0 is a gain that has to
                        # come down before the next run -- see
                        # RailOscillationGuard.
                        f"railosc={st.get('n_rail_osc', 0)} "
                        f"hz={st.get('hz_meas', 0.0):.0f}/"
                        f"{st.get('hz', 0.0):.0f} "
                        f"dtmax={st.get('dt_max_ms', 0.0):.0f}ms "
                        f"pub_stale={pub.get('n_stale', 0)}"
                        f"/{pub.get('n_zeroed', 0)}float "
                        f"read_stale={st.get('n_stale', 0)} "
                        # Ticks the joint-space hold contributed anything on.
                        # Stuck at 0 means the flag never reached the backend --
                        # a typo would otherwise be invisible, because the joint
                        # it holds is one the task wrench never touches either.
                        f"jhold={st.get('n_joint_hold', 0)} "
                        f"assist={st.get('n_stiction', 0)} "
                        # Ticks Lambda's angular eigenvalue floor clamped on.
                        # A light axis -- tool roll -- is the one the shaped law
                        # starves, because the commanded moment scales with the
                        # apparent inertia and breakaway friction does not. See
                        # LAMBDA_MIN_ANG_KGM2. Stuck at 0 means the floor is
                        # inert and a stuck roll is not this.
                        f"lamfloor={st.get('n_lambda_ang_floor', 0)}")
                else:
                    actuation = (
                        f"tau_peak={st.get('tau_peak', 0.0):.2f}Nm "
                        f"i={st.get('i_peak', 0.0):.2f}Nm "
                        f"sat={st.get('n_saturated', 0)}")
            print(
                format_follow_lag(
                    p_live, p_cmd, T_flange_viz[:3, :3], R_cmd,
                    approach_controller._input_active,
                    project_ms=tick.project_ms, actuation=actuation),
                flush=True,
            )
            print(
                "spacemouse: "
                f"|lin|={1000.0 * float(np.linalg.norm(lin)):.2f}mm "
                f"lin=[{lin[0]:.5f},{lin[1]:.5f},{lin[2]:.5f}] "
                f"n={int(stick['n'])} nz={int(stick['nz'])} "
                f"age={age_ms:.0f}ms follow_idle={follow.idle} "
                f"quiet={quiet} pending={1000.0 * follow.pending_m:.2f}mm",
                flush=True,
            )
            stick["n"] = 0
            stick["nz"] = 0
        return tick

    def start_align_keys():
        if args.device == "keyboard":
            return

        def _run():
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while align is not None and not align.finished:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if not ready:
                        continue
                    on_align_key(sys.stdin.read(1))
            except Exception:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        threading.Thread(target=_run, daemon=True, name="align-keys").start()

    def on_align_key(key):
        """Return True when startup align claims the key.

        KeyboardTeleop offers 't'/Enter here first and falls through on a
        falsy return, which is how 't' reaches on_button(0) -- the
        approach<->contact toggle -- once align has finished with it.
        """
        msg = apply_terminal_key(key, align=align, sim_teleop=sim_teleop)
        if msg == "sim_on":
            print(
                "sim teleop ON: SpaceMouse moves the orange ghost. "
                "Press t to snap it back to live TCP. Enter aligns hardware.",
                flush=True,
            )
        elif msg == "sim_off":
            print(
                "sim teleop OFF: orange ghost is the live-TCP snap preview.",
                flush=True,
            )
        elif msg == "align":
            print("startup align: Enter received", flush=True)
        return msg is not None

    def run_startup_align(T_flange, T_ee):
        if align is None or mode["active"] != "approach":
            return None
        if align.phase == "idle":
            align.arm(time.monotonic())
            if args.align_delay is not None:
                print(
                    f"startup align: waiting {args.align_delay:.0f}s (watch viser), then "
                    f"{args.align_duration:.0f}s orientation slerp with the spring on",
                    flush=True,
                )
        R_field, _ = snap_target_flange(T_flange)
        T_flange_hw = chamfer_flange_from_q(interface.get_state().get_position())
        cmd = drive_startup_align(
            align, approach_controller, time.monotonic(), T_flange_hw, R_field, T_ee,
            T_flange_display=T_flange)
        if cmd.phase != align_phase["name"]:
            align_phase["name"] = cmd.phase
            print(f"startup align: {cmd.phase}", flush=True)
            if cmd.finished and sim_probe["p"] is not None:
                sim_probe["p"] = SimTcpProbe(live_tcp_xyz_from_flange(T_flange))
            if cmd.finished:
                p_tcp = live_tcp_xyz_from_flange(T_flange)
                follow.seed(p_tcp, vf.approach_basis_at(p_tcp))
                print(
                    f"field follow: live TCP + stick linear at {args.follow_hz:.0f} Hz → "
                    "slerped field R from live (xyz from stick). Quiet stick freezes.",
                    flush=True,
                )
                # speedJ align done → one handoff to DTC for all remaining teleop.
                if dtc_backend is not None and interface is not dtc_backend:
                    enter_direct_torque()
                if pending_contact["on"]:
                    pending_contact["on"] = False
                    print("startup align done: applying queued contact button", flush=True)
                    toggle_contact_mode()
        return cmd

    def rebind_after_gafropy_reopen(new_interface):
        nonlocal interface, task_space, joint_idx, approach_controller
        interface = new_interface
        task_space = interface.get_task_space(args.task_space)
        if task_space is None:
            raise RuntimeError(f"task space '{args.task_space}' missing after reopen")
        joint_idx = np.array(task_space.get_joint_indices())
        ee = live_ee_motor()
        # Same flags as the original build: a rebuilt controller that silently
        # reverted to library defaults would make a bisect meaningless, and the
        # conditioning callback the contact log reads would go stale.
        approach_controller = _make_approach_controller(
            interface, task_space, ee,
            free_along_normal=args.free_along_normal,
            dls_damping=args.dls_damping,
            per_joint_velocity=args.per_joint_velocity,
            xy_stiffness=args.xy_stiffness,
            ang_stiffness=args.ang_stiffness,
            **_wrench_law_kwargs(args))
        if contact_controller is not None:
            contact_controller.set_conditioning_fn(approach_controller.conditioning)
        if robot_viz is not None:
            robot_viz.update(q_with_vention())
        if eeviz is not None:
            eeviz.update(ee)

    def enter_force_mode_contact():
        nonlocal interface
        if handoff_busy["busy"]:
            return
        handoff_busy["busy"] = True
        try:
            print("-> contact handoff: settling speedJ, then force_mode...", flush=True)
            # Hold current pose (no vf.project snap) while still under gafropy.
            approach_controller.get_objective().set_target(live_ee_motor())
            time.sleep(0.35)
            shutdown_gafropy_for_handoff(interface, settle_s=0.35)
            force_session.enter(settle_s=0.2)
            mode["active"] = "contact"
            print(
                f"-> contact mode (force_mode): Fd={args.force_sign * args.target_force:+.1f}N "
                "tool Z; SpaceMouse XY slide only",
                flush=True,
            )
        except Exception as exc:
            print(f"\033[91mforce_mode handoff failed: {exc}\033[0m", flush=True)
            force_session.exit(settle_s=0.2)
            interface = reopen_gafropy(MODEL, args.hardware_config, robot_ip=robot_ip)
            rebind_after_gafropy_reopen(interface)
            interface.control(approach_controller, blocking=False)
            mode["active"] = "approach"
            print("-> approach mode restored after failed handoff", flush=True)
        finally:
            handoff_busy["busy"] = False

    def enter_approach_from_force_mode():
        nonlocal interface
        if handoff_busy["busy"]:
            return
        handoff_busy["busy"] = True
        try:
            print("-> approach handoff: ending force_mode, restoring speedJ...", flush=True)
            force_session.exit(settle_s=0.35)
            interface = reopen_gafropy(MODEL, args.hardware_config, robot_ip=robot_ip)
            rebind_after_gafropy_reopen(interface)
            # Live pose target — no snap to a stale pre-contact objective.
            approach_controller.set_input_active(False)
            approach_controller.hold_live_pose()
            interface.control(approach_controller, blocking=False)
            mode["active"] = "approach"
            print("-> approach mode: holding live pose", flush=True)
        except Exception as exc:
            print(f"\033[91mapproach handoff failed: {exc}\033[0m", flush=True)
        finally:
            handoff_busy["busy"] = False

    def _hardware_twist(twist):
        # Hands-off / rest bias: freeze — zero speedJ (approach) and clear speedL (contact).
        if stick_is_idle(twist):
            if mode["active"] == "approach":
                approach_controller.set_input_active(False)
                if follow_rebase_on_idle(args, on_dtc=on_dtc_now()):
                    approach_controller.hold_live_pose()
            elif mode["active"] == "contact":
                # hardware_free_teleop: zero twist still update_command so the
                # pose spring tracks live EE. DualLoop keeps tool Z.
                approach_controller.update_command(twist)
            if args.contact_backend == "force_mode" and force_session is not None and force_session.active:
                force_session.clear_twist()
            return
        if mode["active"] == "approach":
            approach_controller.update_command(twist)
            return
        if args.contact_backend == "force_mode":
            if force_session is not None and force_session.active:
                # Spacemouse twist is body/tool frame; force session zeros tool Z.
                force_session.set_twist(twist.get_linear(), twist.get_angular())
            return
        # speedj contact: same pose-spring plant as hardware_free_teleop.
        approach_controller.update_command(twist)

    def on_twist(twist):
        lin = np.asarray(twist.get_linear(), dtype=float).reshape(3)
        ang = np.asarray(twist.get_angular(), dtype=float).reshape(3)
        now = time.monotonic()
        stick["t"] = now
        stick["lin"] = lin
        stick["n"] = int(stick["n"]) + 1
        approach_controller.set_stick_linear(lin)
        # Full Twist even when field-follow drops angular / contact zeros Z.
        approach_controller.set_raw_twist(np.concatenate([ang, lin]))
        if float(np.linalg.norm(lin)) > 1e-12:
            stick["t_nz"] = now
            stick["nz"] = int(stick["nz"]) + 1
        if handoff_busy["busy"]:
            return
        if align is not None and not align.finished:
            if sim_teleop["on"] and sim_probe["p"] is not None:
                route_spacemouse_twist(
                    twist, sim_probe=sim_probe["p"], on_hardware=lambda _: None)
            return
        if field_follow_in_mode(mode["active"], args.contact_backend):
            lin_f = follow_linear(lin, ignore_z=mode["active"] == "contact")
            if float(np.linalg.norm(lin_f)) > 1e-12:
                follow.add_body_linear(lin_f)
            return
        route_spacemouse_twist(
            twist, sim_probe=None, on_hardware=_hardware_twist)

    def _snap_follow_to_live():
        """Put field-follow's carrot on the live TCP, discarding any lead."""
        if follow is None:
            return
        try:
            T_flange = chamfer_flange_from_q(q_with_vention())
            follow.set_idle(live_tcp_xyz_from_flange(T_flange),
                            clear_pending=True)
        except Exception as exc:
            print(f"\033[93mcould not re-base the follow carrot on exit: "
                  f"{exc}\033[0m", flush=True)

    def toggle_contact_mode():
        if handoff_busy["busy"]:
            return
        if args.contact_backend == "force_mode":
            if mode["active"] == "approach":
                enter_force_mode_contact()
            else:
                enter_approach_from_force_mode()
            return
        assert contact_controller is not None
        if mode["active"] == "approach":
            if measured_fz_fn is None:
                print(
                    "\033[91mrefusing speedj contact: no ActualTCPForce reader\033[0m",
                    flush=True,
                )
                return
            mode["active"] = "contact"
            wrench_press = use_wrench_press(args)
            (enter_wrench_contact if wrench_press else enter_speedj_contact)(
                mode=mode,
                contact_controller=contact_controller,
                approach_controller=approach_controller,
                interface=interface,
            )
            how = (f"press = force, Fd + {resolve_press_kp(args):g}*(Fd-Fs) N "
                   f"capped at {contact_controller._press_max_n:.1f}N"
                   if wrench_press
                   else f"PID holds Fd \u22642mm/s")
            print(
                f"-> contact mode ({args.contact_backend}"
                f"{'/wrench' if wrench_press else ''}): latched until button "
                f"{args.mode_button}; field-follow tangent XY, "
                f"{how} to {args.target_force:.1f}N",
                flush=True,
            )
            return
        # exit_*_contact snaps the objective onto the live pose, but under
        # --dtc-compliance-hold nothing re-bases *field-follow's* carrot, and
        # drive_field_follow re-sends `follow.last_target` on the very next
        # call -- the carrot from before the press, which the arm has since
        # moved away from. That overwrites the snap within one main-loop tick
        # and the arm jumps to it. Re-base the carrot first so the two agree.
        _snap_follow_to_live()
        (exit_wrench_contact if use_wrench_press(args)
         else exit_speedj_contact)(
            mode=mode,
            contact_controller=contact_controller,
            approach_controller=approach_controller,
            interface=interface,
        )
        print("-> approach mode: holding live pose", flush=True)

    def on_button(index):
        # Handoff is rising-edge button only (SpaceMouseTeleop); twists never toggle mode.
        print(f"spacemouse: button {index}", flush=True)
        if index != args.mode_button:
            return
        if align is not None and not align.finished:
            pending_contact["on"] = not pending_contact["on"]
            queued = "queued" if pending_contact["on"] else "cancelled"
            print(
                f"startup align running: contact {queued} for when align finishes",
                flush=True,
            )
            return
        toggle_contact_mode()

    # ------------------------------------------------------------------
    # Deterministic path (--path-mode). Same carrot, same wrench law; the
    # steps come from a plan instead of a stick, which is the whole point:
    # anything left over is the controller's own latency.
    # ------------------------------------------------------------------
    path_session = None
    path_t = {"last": None, "held": False, "held_since": None, "hist": []}
    stop_keys = threading.Event()

    def _path_execute_start():
        # Re-base the carrot on the live TCP. Between align finishing and
        # EXECUTE the carrot sits wherever the seed left it while the arm sags
        # or is nudged, and the first commit would then command that stale pose
        # -- a step of up to --max-lead-m in one tick. Snapping makes the first
        # commanded pose the live one, so the run opens at zero error. Same
        # rule as leaving contact, in the other direction: whoever snaps the
        # objective must re-base the carrot, or the carrot wins on the next tick.
        _snap_follow_to_live()
        if args.path_contact_at == "execute":
            _path_enter_contact()

    def _path_enter_contact():
        # Entered from FREE SPACE, deliberately: start() tares the F/T sensor
        # and a tare taken while touching absorbs the contact force into the
        # bias. At --path-contact-at plunge (the default) that free space is
        # the standoff point, --path-standoff above the surface, which is where
        # the approach ends and the descent begins -- so the tare is still
        # taken clear of the part, and the whole traverse to the start point
        # runs in plain approach mode. The press commands 0 N either way until
        # the latch engages; what entering early actually cost was the 200 Hz
        # force thread sharing the RTDE socket for the length of the approach.
        if mode["active"] != "approach":
            return
        if contact_controller is not None:
            contact_controller.set_press_external(False)
        toggle_contact_mode()
        if mode["active"] != "contact":
            # toggle_contact_mode refuses without a force reader, and a run
            # with no press is not the experiment -- do not walk the plan into
            # the part on the pose spring alone.
            print("\033[91mpath: contact mode refused; aborting the run\033[0m",
                  flush=True)
            if path_session is not None:
                path_session.request_abort()

    def _path_exit_contact():
        if contact_controller is not None:
            contact_controller.set_press_external(False)
        if mode["active"] == "contact":
            toggle_contact_mode()
        _spawn_path_plot()

    def _render_path_plot(delay_s=0.0):
        """Run plot_path_run.py on the newest logs. Never raises."""
        if not path_mode_on(args) or not args.path_plot:
            return
        script = pathlib.Path(__file__).resolve().parent / "plot_path_run.py"
        if delay_s > 0.0:
            time.sleep(delay_s)
        try:
            subprocess.run([sys.executable, str(script), "--no-show"],
                           check=False, timeout=180)
        except Exception as exc:
            print(f"\033[93mpath plot failed ({exc}); run "
                  f"'python examples/plot_path_run.py' by hand\033[0m",
                  flush=True)

    def _spawn_path_plot():
        """Render commanded-vs-actual for the run that just finished.

        A subprocess, deliberately: this is called from the loop that holds the
        robot, and matplotlib in-process is the one thing in this file that has
        no business sharing a thread with the arm. Delayed a second so the
        contact CSV -- closed by contact_controller.stop() in the toggle above
        -- and the line-buffered dtc log are both on disk first.
        """
        threading.Thread(target=lambda: _render_path_plot(delay_s=1.0),
                         daemon=True, name="path-plot").start()

    if path_mode_on(args):
        path_session = PathSession(
            vf=vf,
            mesh=object_surface,
            free_speed_m_s=args.path_speed,
            contact_speed_m_s=args.path_contact_speed,
            standoff_m=args.path_standoff,
            ds_m=args.path_ds,
            engage_gap_m=args.path_engage_gap,
            release_gap_m=args.path_release_gap,
            depth_m=args.path_depth,
            touch_n=args.path_touch_n,
            max_seek_m=args.path_max_seek,
            on_execute_start=_path_execute_start,
            on_execute_end=_path_exit_contact,
        )

    def tick_path(T_flange_viz):
        """Drive the follow carrot from the plan.

        Called on EVERY main-loop tick, including the ones that cannot move the
        arm -- during startup align, and under --dtc-hold-zero. Viser records
        button and key presses on its own threads and this is where they are
        consumed, so a tick skipped here is a button that does nothing. Only
        EXECUTE is gated; preview, abort and reset always run.
        """
        if path_session is None or follow is None:
            return
        now = time.monotonic()
        last = path_t["last"]
        path_t["last"] = now
        dt = 0.0 if last is None else min(0.1, now - last)
        tcp = live_tcp_xyz_from_flange(T_flange_viz)
        carrot = follow.xyz
        if carrot is None:
            carrot = tcp
        blocked = ""
        if align is not None and not align.finished:
            blocked = "startup align has not finished -- press Enter to align first"
        elif args.dtc_hold_zero:
            blocked = ("--dtc-hold-zero streams tau=0, so the arm cannot move. "
                       "Re-run without it to execute")
        fs = (None if contact_controller is None
              else contact_controller.last_fs)
        # Never outrun the arm. The carrot is clamped to --max-lead-m from the
        # live TCP, so a plan that keeps advancing into a stalled arm is
        # pushing against a stop: the clamp pulls the carrot back, the plan
        # pushes it out, and the commanded pose oscillates at the clamp -- with
        # nothing wrong with the arm at all. Holding instead makes the path
        # self-pacing: it advances exactly as fast as the arm can follow.
        lag = float(np.linalg.norm(np.asarray(carrot) - np.asarray(tcp)))
        hold = lag >= float(args.path_hold_frac) * float(args.max_lead_m)
        if hold != path_t["held"]:
            path_t["held"] = hold
            path_t["held_since"] = now if hold else None
            if hold and path_session.running:
                print(f"path: holding, the arm is {1e3 * lag:.1f} mm behind the "
                      f"command (clamp {1e3 * args.max_lead_m:.0f} mm)",
                      flush=True)
        # A stall is the arm NOT MOVING, not the arm being behind. Being
        # behind is the normal state of a spring dragging a friction-limited
        # arm: dtc_20260907_131940.csv sat at 20.9 mm of lag for its whole run
        # and still covered 17 mm in 2.8 s -- and an abort on lag alone killed
        # it twice. Measure displacement over the abort window instead. It has
        # to be a window, not a tick: the motion is stick-slip (median TCP
        # speed 0.14 mm/s, peaks over 40 mm/s), so any instantaneous speed
        # reads zero most of the time.
        hist = path_t["hist"]
        if path_session.running:
            hist.append((now, np.asarray(tcp, dtype=float).copy()))
        else:
            hist.clear()
        # Keep exactly one sample OLDER than the window. Trimming to "no older
        # than the window" and then asking for "at least the window" is a race
        # the test always loses -- that bug made the abort silent on
        # dtc_20260907_134616.csv, which then sat at 13.2 N for 5 s having
        # moved 0.16 mm.
        span = float(args.path_stall_abort_s)
        while len(hist) > 2 and (now - hist[1][0]) > span:
            hist.pop(0)
        moved = None
        if hist and span > 0.0 and (now - hist[0][0]) >= span:
            moved = float(np.linalg.norm(np.asarray(tcp) - hist[0][1]))
        if (hold and path_session.running and moved is not None
                and moved < 1e-3 * args.path_stall_mm):
            st = dtc_backend.stats() if dtc_backend is not None else {}
            print(f"\033[91mpath: the arm moved {1e3 * moved:.2f} mm in "
                  f"{span:.1f}s while {1e3 * lag:.1f} mm behind the command "
                  f"-- stalled against something. Aborting rather than pushing "
                  f"harder.\033[0m", flush=True)
            # What it was stalled AGAINST, in the two numbers that separate
            # "the spring is not asking for enough" from "the joint cannot
            # break away": the commanded force, and the worst joint torque.
            print(f"  at the stall: |W_lin|={st.get('w_lin', float('nan')):.1f}N  "
                  f"tau_peak={st.get('tau_peak', float('nan')):.2f}Nm  "
                  f"clip_worst=j{st.get('clip_joint_worst', -1)}",
                  flush=True)
            path_session.request_abort()
            hist.clear()
        step = path_session.tick(dt, carrot, tcp_xyz=tcp, fs=fs,
                                 allow_execute=not blocked,
                                 block_reason=blocked, hold=hold)
        if contact_controller is not None:
            contact_controller.set_press_latch_debug(
                distance_m=path_session.latch_distance_m,
                phase=path_session.phase)
            if mode["active"] == "contact":
                # One bool decides the press command, free_along_normal and
                # field-follow's tangent gate -- exactly as the Fs latch used
                # to, only now it is geometry.
                contact_controller.set_press_external(path_session.press_engaged)
        if step is None or not path_session.running:
            return
        if (args.path_contact_at == "plunge" and step.press_phase
                and mode["active"] == "approach"):
            # The plan has finished the approach and is about to descend. Still
            # --path-standoff above the surface, so the tare is clear of the
            # part.
            _path_enter_contact()
        d = np.asarray(step.delta_world, dtype=float).reshape(3)
        if float(np.linalg.norm(d)) <= 1e-12:
            return
        # add_body_linear takes a *body* step: FieldFollow.tick rotates it by
        # the field basis at the carrot before adding it, so undo exactly that
        # rotation here rather than assuming the two frames agree.
        R_field = np.asarray(vf.approach_basis_at(carrot), dtype=float).reshape(3, 3)
        follow.add_body_linear(R_field.T @ d)

    def start_path_loop():
        """Run the command side at --loop-hz on its own thread, under viser.

        gafropy's ``Visualizer.show()`` ticks its callbacks at 20 Hz
        (``time.sleep(0.05)``), so with the carrot fed from the viser ticker a
        20 mm/s path would advance in 1 mm steps -- the same staircase into the
        loop that ``--follow-linear-approach`` exists to smooth, reintroduced
        by the visualization. The viser ticker keeps the visuals; this owns the
        commands, exactly as the headless loop does. Nothing else drives them
        in path mode, so the two threads never write the objective together.
        """
        loop_dt = 1.0 / max(1.0, float(args.loop_hz))

        def _run():
            while not stop_keys.is_set():
                try:
                    q = q_with_vention()
                    T_flange = chamfer_flange_from_q(q)
                    T_ee = np.asarray(live_ee_motor().to_transformation_matrix())
                    tick_path(T_flange)
                    if align is not None and not align.finished:
                        run_startup_align(T_flange, T_ee)
                    elif field_follow_in_mode(mode["active"],
                                              args.contact_backend):
                        run_field_follow(T_flange, T_ee)
                except Exception as exc:
                    print(f"path loop tick skipped: {exc}", flush=True)
                time.sleep(loop_dt)

        threading.Thread(target=_run, daemon=True, name="path-loop").start()
        print(f"path mode: command loop at {args.loop_hz:.0f} Hz on its own "
              f"thread (viser ticks the visuals at 20 Hz).", flush=True)

    tool_roll = {"flip": bool(args.tool_flip), "label": None}

    def tool_roll_name(flip=None):
        flip = tool_roll["flip"] if flip is None else bool(flip)
        return "B (180 deg)" if flip else "A (0 deg)"

    def apply_tool_roll(flip, *, announce=True):
        """Set tool orientation A/B on the field, the helper and the display."""
        flip = bool(flip)
        tool_roll["flip"] = flip
        if follow is not None:
            follow.set_roll_flip(flip)
        else:
            setter = getattr(vf, "set_roll_flip", None)
            if callable(setter):
                setter(flip)
        if tool_roll["label"] is not None:
            try:
                tool_roll["label"].text = f"TOOL {tool_roll_name(flip)}"
            except Exception:
                pass
        if path_session is not None:
            path_session.tool_label = f"TOOL {tool_roll_name(flip)}"
        if announce:
            print(f"\033[96mtool orientation -> {tool_roll_name(flip)}\033[0m",
                  flush=True)

    def toggle_tool_roll():
        """Flip the tool 180 deg about its own axis. Free space only.

        A 180 deg roll is a real wrist move, and commanding one while the tool
        is on the part drags it across the surface at the full angular
        stiffness. Contact and a running path both refuse.
        """
        if mode["active"] == "contact":
            print("\033[93mtool flip refused: leave contact mode first (the "
                  "tool is on the part)\033[0m", flush=True)
            return
        if path_session is not None and path_session.running:
            print("\033[93mtool flip refused: a path is running ('a' aborts "
                  "it first)\033[0m", flush=True)
            return
        apply_tool_roll(not tool_roll["flip"])
        print("the arm will roll wrist 3 through 180 deg as the field target "
              "moves -- watch it, and re-pick the path if you had one",
              flush=True)

    def on_path_key(key):
        if path_session is None:
            return False
        if key in ("\r", "\n"):
            path_session.request_execute()
        elif key in ("f", "F"):
            toggle_tool_roll()
        elif key == "p":
            path_session.request_preview()
        elif key == "a":
            path_session.request_abort()
        elif key == "r":
            path_session.request("reset")
        else:
            return False
        return True

    def path_key_help() -> str:
        return ("keys: Enter=execute  p=preview  a=abort  r=reset picks  "
                "f=flip tool 180 deg")

    def start_path_keys():
        """Enter=execute, p=preview, a=abort, r=reset. Survives align finishing.

        Align gets first refusal on every key while it is running ('t' and
        Enter are its), and anything it does not claim falls through to the
        path -- otherwise previewing and clearing the picks are dead until the
        arm has aligned, which is exactly backwards.

        The loop watches ``stop_keys`` and restores the terminal on the way
        out. A daemon thread sitting in cbreak that never exits leaves the
        shell without echo after Ctrl+C, which reads as "the script crashed".
        """

        def _run():
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not stop_keys.is_set():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if not ready:
                        continue
                    key = sys.stdin.read(1)
                    if align is not None and not align.finished:
                        if on_align_key(key):
                            continue
                    on_path_key(key)
            except Exception:
                pass
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, saved)
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True, name="path-keys").start()

    def enter_direct_torque():
        """speedJ (gafropy) → ur_rtde directTorque. One-shot; stays on DTC.

        After this, ``interface`` is the DTC backend (duck-types get_state /
        get_system / get_task_space). Align has already finished under speedJ
        (or ``--no-startup-align``). Contact toggle only changes Z overlay /
        gains — no further actuator swap.
        """
        nonlocal interface, approach_controller, task_space, joint_idx
        if dtc_backend is None or interface is dtc_backend:
            return
        if handoff_busy["busy"]:
            return
        handoff_busy["busy"] = True
        try:
            # Snapshot full q while gafropy still owns the arm.
            dtc_backend.seed_from_interface(interface)
            print(
                "-> dtc handoff: settling speedJ, then directTorque for all "
                "teleop (free space + contact)...",
                flush=True,
            )
            approach_controller.set_input_active(False)
            approach_controller.hold_live_pose()
            time.sleep(0.2)
            shutdown_gafropy_for_handoff(interface, settle_s=0.35)
            dtc_backend.connect()
            if args.dtc_hold_zero:
                print_dtc_rate_verification(dtc_backend.verify_rate(duration_s=2.0))
            interface = dtc_backend
            task_space = dtc_backend.get_task_space(args.task_space)
            if task_space is None:
                raise RuntimeError(
                    f"task space '{args.task_space}' missing on DTC backend")
            joint_idx = np.array(task_space.get_joint_indices())
            # Rebuild with DTC wrench gains if any; align used the speedJ plant.
            ee = live_ee_motor()
            approach_controller = _make_approach_controller(
                dtc_backend, task_space, ee,
                free_along_normal=args.free_along_normal,
                dls_damping=args.dls_damping,
                per_joint_velocity=args.per_joint_velocity,
                xy_stiffness=args.xy_stiffness,
                ang_stiffness=args.ang_stiffness,
                **_wrench_law_kwargs(args))
            if contact_controller is not None:
                contact_controller.set_interface(dtc_backend)
                contact_controller.set_task_space(task_space)
                contact_controller.set_conditioning_fn(
                    approach_controller.conditioning)
            approach_controller.set_input_active(False)
            approach_controller.hold_live_pose()
            if speedj_trace is not None:
                if args.dtc_law == "wrench":
                    dtc_backend.set_command_trace(speedj_trace.record)
                else:
                    approach_controller.set_command_trace(speedj_trace.record)
            dtc_backend.start(approach_controller)
            print(
                "-> dtc teleop live: stick field-follows under directTorque; "
                f"button {args.mode_button} toggles contact Z overlay",
                flush=True,
            )
        except Exception as exc:
            print(f"\033[91mdtc handoff failed: {exc}\033[0m", flush=True)
            raise
        finally:
            handoff_busy["busy"] = False

    def start_control_and_device():
        nonlocal device, speedj_trace
        if dtc_push_test_mode(args):
            if speedj_trace is None and args.speedj_plot:
                from geodude_teleoperation.speedj_plot import SpeedJTrace
                speedj_trace = SpeedJTrace()
            approach_controller.set_input_active(False)
            approach_controller.hold_live_pose()
            enter_direct_torque()
            if speedj_trace is not None and dtc_backend is not None:
                dtc_backend.set_command_trace(speedj_trace.record)
                speedj_trace.start(figsize=tuple(args.plot_size))
            print(
                "dtc push test: lambda=on, joint hold=off, pose frozen at "
                "handoff, no device. Push the arm -- it should return when "
                "released. |W_ang| stays ~0 by construction: this run does not "
                "test --dtc-damping-ratio-ang. Ctrl+C to quit.",
                flush=True,
            )
            return
        # Start frozen: no speedJ until a non-idle SpaceMouse reading arrives.
        defer = defer_dtc_until_align_done(
            dtc_backend=dtc_backend, align=align, args=args)
        if args.speedj_plot and speedj_trace is None:
            from geodude_teleoperation.speedj_plot import SpeedJTrace
            speedj_trace = SpeedJTrace()
            if not defer and dtc_backend is not None and args.dtc_law == "wrench":
                dtc_backend.set_command_trace(speedj_trace.record)
            else:
                # Align (and speedj backend) use the controller trace.
                approach_controller.set_command_trace(speedj_trace.record)
            speedj_trace.start(figsize=tuple(args.plot_size))
        approach_controller.set_input_active(False)
        approach_controller.hold_live_pose()
        if dtc_backend is None:
            interface.control(approach_controller, blocking=False)
        elif defer:
            print(
                "-> speedJ for startup align; directTorque handoff after align "
                "done (then DTC for free space + contact)",
                flush=True,
            )
            interface.control(approach_controller, blocking=False)
        else:
            enter_direct_torque()
        if path_mode_on(args):
            print(
                "path mode: no input device opened. Click a start and an end "
                "point on the object in viser. " + path_key_help() + ".",
                flush=True,
            )
            return
        device = KeyboardTeleop(
            step=0.0025) if args.device == "keyboard" else SpaceMouseTeleop(
                # Slow enough that the 20 Hz field carrot stays inside the 2 cm
                # lead and speedJ can catch it (full stick ~7.5 cm/s at 50 Hz).
                linear_scale=args.stick_linear_scale,
                angular_scale=args.stick_angular_scale,
                deadzone=0.02,
                device_path=args.spacemouse_path,
                device_index=args.spacemouse_index)
        if args.device == "keyboard":
            device.start(on_twist, on_button, on_key=on_align_key)
        else:
            device.start(on_twist, on_button)
        if sim_probe["p"] is not None:
            print(
                "viser: orange ghost is the live-TCP snap preview. Press t to "
                "SpaceMouse-teleop the ghost in sim; t again resets it to live TCP. "
                "Enter aligns hardware under speedJ"
                + (", then DTC teleop" if dtc_backend is not None else "")
                + f". After align, stick linear field-follows at --follow-hz; "
                "angular ignored.",
                flush=True,
            )
        else:
            print(
                "free-space scale: SpaceMouse lin=0.0015 ang=0.05, "
                "admittance K=2000 D=20 dt=0.001 max|qdot|=0.8, twist_linear_scale=1 "
                "(no VF project in approach)",
                flush=True,
            )
        if dtc_compliance_hold_mode(args):
            print(
                "dtc compliance hold: hold_at_tick=off — quiet stick keeps the "
                "commanded pose; external pushes should return when released. "
                "SpaceMouse still drives field-follow after align.",
                flush=True,
            )

    try:
        backend_note = {
            "speedj": "speedj contact (pose spring + Z overlay)",
            "force_mode": "force_mode contact (not admittance)",
            "dtc": (
                "speedJ align → dtc teleop (free + contact via directTorque)"
                + (" [HOLD-ZERO]" if args.dtc_hold_zero else "")
                + (" [PUSH-TEST]" if dtc_push_test_mode(args)
                   else " [COMPLIANCE-HOLD]" if dtc_compliance_hold_mode(args)
                   else "")
            ),
        }[args.contact_backend]
        if args.no_viz:
            print(f"Teleoperating '{args.task_space}' on real geodude (headless) "
                  f"-- {backend_note}; button {args.mode_button} toggles. Ctrl+C to quit.")
            start_control_and_device()
            if path_mode_on(args):
                apply_tool_roll(tool_roll["flip"], announce=False)
                print(f"tool orientation: {tool_roll_name()}  (F flips it)",
                      flush=True)
                start_path_keys()
            elif align is not None and args.align_delay is None:
                print(
                    "startup align: Press Enter to slerp UR onto the field. "
                    "Press t for viser-only stick (headless has no ghost).",
                    flush=True,
                )
                start_align_keys()
            hold_log_t = {"t": float("-inf")}
            loop_dt = 1.0 / max(1.0, float(args.loop_hz))
            while True:
                if dtc_push_test_mode(args):
                    now = time.monotonic()
                    if (now - hold_log_t["t"]) >= 1.0 and dtc_backend is not None:
                        hold_log_t["t"] = now
                        print(push_test_line(approach_controller,
                                             dtc_backend.stats()), flush=True)
                    time.sleep(loop_dt)
                    continue
                q = q_with_vention()
                T_flange = chamfer_flange_from_q(q)
                T_ee = np.asarray(live_ee_motor().to_transformation_matrix())
                tick_path(T_flange)
                if align is not None and not align.finished:
                    run_startup_align(T_flange, T_ee)
                elif field_follow_in_mode(mode["active"], args.contact_backend):
                    run_field_follow(T_flange, T_ee)
                time.sleep(loop_dt)
        else:
            viz = gafropy_viz.Visualizer()
            print(f"viser running at http://localhost:{viz.port} -- "
                  "holding pose; open the page before moving the spacemouse.")

            robot_viz = viz.add_robot(kin_system,
                                      name="geodude", joint_sliders=False)
            robot_viz.update(q_with_vention())
            n_hidden = hide_gripper_visuals(robot_viz)
            update_chamfer = add_right_chamfer_visual(
                viz, kin_system, q=q_with_vention())
            update_ghost = add_ghost_chamfer_visual(viz, visible=True)
            q0 = q_with_vention()
            fk0 = kin_system.compute_forward_kinematics(q0)
            T_wrist0 = np.asarray(
                fk0.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
            T_live0 = chamfer_world_from_wrist3(T_wrist0)
            sim_probe["p"] = SimTcpProbe(live_tcp_xyz_from_flange(T_live0))
            print(
                f"ee viz: hid {n_hidden} gripper meshes; "
                "gray chamfer = held hardware; orange ghost is the snap target "
                "(field orientation at the live TCP, same pose as after pressing t) "
                "until you press Enter. "
                "Hide /right_chamfer_ee_field_ghost in the scene tree if you do not want it.",
                flush=True,
            )

            viz.scene.add_point_cloud(
                "/object", points=object_surface.vertices, colors=(200, 200, 200), point_size=0.002
            )
            viz.scene.add_point_cloud(
                "/field_source", points=vf.source_points, colors=(220, 50, 40),
                point_size=0.004,
            )
            viz.scene.add_point_cloud(
                "/field_sink", points=vf.sink_points, colors=(40, 90, 220),
                point_size=0.004,
            )
            print(
                f"field BCs: {len(vf.source_points)} source (red /field_source) and "
                f"{len(vf.sink_points)} sink (blue /field_sink) end-cap vertices. "
                "Toggle them in the scene tree.",
                flush=True,
            )
            verts = np.asarray(object_surface.vertices, dtype=float)
            bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
            z_len = float(np.clip(0.04 * bbox_diag, 0.015, 0.04))
            print("field Z arrows: sampling approach axis on the object (scene tree /field_z)...",
                  flush=True)
            field_z = vf.field_z_arrows(verts, max_count=256, length=z_len)
            viz.scene.add_arrows(
                "/field_z",
                points=field_z,
                colors=(40, 90, 220),
                shaft_radius=0.08 * z_len,
                head_radius=0.2 * z_len,
                head_length=0.35 * z_len,
                visible=True,
            )
            print(
                f"field Z: {len(field_z)} blue arrows OUT of the top (object +normal). "
                f"Orange EE +Z points into the surface (anti-parallel to arrows). "
                f"Toggle /field_z and /right_chamfer_ee_field_ghost in the viser scene tree.",
                flush=True,
            )
            ws_z = workstation_field_z_arrows(verts, length=z_len)
            viz.scene.add_arrows(
                "/field_z_workstation",
                points=ws_z,
                colors=(150, 200, 255),
                shaft_radius=0.06 * z_len,
                head_radius=0.16 * z_len,
                head_length=0.3 * z_len,
                visible=True,
            )
            print(
                f"workstation field: {len(ws_z)} light-blue +Z arrows on the table plane "
                f"(outer ~6 cm of the object overlaps this frame, full table by ~45 cm). "
                f"Toggle /field_z_workstation in the scene tree.",
                flush=True,
            )

            def update():
                try:
                    _update()
                except Exception as exc:
                    print(f"viser tick skipped: {exc}", flush=True)

            def _update():
                if dtc_push_test_mode(args):
                    if handoff_busy["busy"]:
                        return
                    q = q_with_vention()
                    robot_viz.update(q)
                    update_chamfer(q)
                    return
                if handoff_busy["busy"] or (
                    mode["active"] == "contact"
                    and args.contact_backend == "force_mode"
                ):
                    # force_mode shuts down gafropy; speedj contact still has live state.
                    return
                q = q_with_vention()
                robot_viz.update(q)
                update_chamfer(q)
                fk = kin_system.compute_forward_kinematics(q)
                T_wrist = np.asarray(
                    fk.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
                T_live_flange = chamfer_world_from_wrist3(T_wrist)
                T_ee = np.asarray(live_ee_motor().to_transformation_matrix())
                if path_session is not None:
                    # Outside the field-follow branch too, so picking and the
                    # status readout work while startup align is still running.
                    path_session.set_live_tcp(
                        live_tcp_xyz_from_flange(T_live_flange))
                    path_session.refresh_gui()
                if path_session is not None:
                    # Visuals only here: start_path_loop owns every command in
                    # path mode, so this tick must not drive anything.
                    p_preview = path_session.preview_xyz()
                    if p_preview is not None:
                        update_ghost(ghost_flange_world(
                            T_live_flange, vf.approach_basis_at(p_preview),
                            p_preview, 1.0))
                    elif follow.last_target is not None:
                        T_tcp = matrix_from_motor(follow.last_target)
                        update_ghost(ghost_flange_world(
                            T_live_flange, T_tcp[:3, :3], T_tcp[:3, 3], 1.0))
                    return
                if align is not None and not align.finished:
                    run_startup_align(T_live_flange, T_ee)
                    probe = sim_probe["p"]
                    if not sim_teleop["on"]:
                        T_frozen = align.desired_flange()
                        if T_frozen is not None:
                            if probe is not None:
                                probe.set_xyz(live_tcp_xyz_from_flange(T_live_flange))
                            update_ghost(T_frozen)
                            return
                        if probe is not None:
                            probe.set_xyz(live_tcp_xyz_from_flange(T_live_flange))
                    if probe is not None:
                        R_field = field_preview.update_approach_basis(
                            probe.xyz(), now=time.monotonic())
                        p_tcp = (
                            probe.step(R_field) if sim_teleop["on"] else probe.xyz())
                        update_ghost(ghost_flange_world(
                            T_live_flange, R_field, p_tcp, 1.0))
                    return
                if field_follow_in_mode(mode["active"], args.contact_backend):
                    # tick_path already ran above, before the align branch, so
                    # the preview and the requests are handled in one place.
                    run_field_follow(T_live_flange, T_ee)
                    if follow.last_target is not None:
                        T_tcp = matrix_from_motor(follow.last_target)
                        update_ghost(ghost_flange_world(
                            T_live_flange, T_tcp[:3, :3], T_tcp[:3, 3], 1.0))
                    return
                probe = sim_probe["p"]
                R_field = field_preview.update_approach_basis(
                    probe.xyz(), now=time.monotonic())
                p_tcp = probe.step(R_field)
                T_ghost = ghost_flange_world(T_live_flange, R_field, p_tcp, 1.0)
                update_ghost(T_ghost)

            viz.add_ticker(lambda: update())

            if path_session is not None:
                path_session.on_flip_tool = toggle_tool_roll
                path_session.attach(viz.server)
                try:
                    # A scene label, not just a GUI field: which way the tool
                    # faces is a property of the arm in the scene, and the
                    # operator is looking at the arm, not at the sidebar.
                    tool_roll["label"] = viz.scene.add_label(
                        "/tool_orientation",
                        text=f"TOOL {tool_roll_name()}",
                        position=(0.0, 0.0, 0.6))
                except Exception as exc:
                    print(f"path: tool label unavailable ({exc})", flush=True)
                apply_tool_roll(tool_roll["flip"], announce=False)
                print(f"tool orientation: {tool_roll_name()}  "
                      f"(F flips it 180 deg about tool Z, free space only)",
                      flush=True)
                print(
                    "path mode: click a START point then an END point on the "
                    "object in viser (a third click starts over). "
                    + path_key_help() + ".",
                    flush=True,
                )
            start_control_and_device()
            if path_mode_on(args):
                start_path_keys()
                start_path_loop()
            elif align is not None and args.align_delay is None:
                print(
                    "startup align: orange ghost is the live-TCP snap. Press t to "
                    "teleop it in viser; t again resets to live TCP. Enter slerps UR.",
                    flush=True,
                )
                start_align_keys()

            viz.show(message=f"startup align on '{args.task_space}': t=sim ghost, "
                     f"Enter=hardware slerp under speedJ"
                     + (", then DTC teleop" if dtc_backend is not None else "")
                     + f", then field-follow at {args.follow_hz:.0f} Hz "
                     f"-- {backend_note}. viser at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        # FIRST, before anything else: tau = 0 is float (gravity is compensated
        # on-controller), and the control loop keeps driving at full authority
        # through the whole teardown -- toward a carrot the main loop stopped
        # updating the instant it exited. Observed on hardware as the wrist
        # swinging after Ctrl+C.
        for _b in (interface, dtc_backend):
            _float = getattr(_b, "float_now", None)
            if callable(_float):
                try:
                    _float()
                except Exception:
                    pass
                break
        # Stop the key reader and put the terminal back. It runs in cbreak so a
        # single keypress arrives without Enter, and a daemon thread parked
        # there through exit leaves the shell with no echo -- which looks like
        # the script crashed, and outlives it.
        if path_mode_on(args) and path_session is not None:
            # Ctrl+C is how most runs end, and it is the run you most want to
            # look at. Synchronous here (the logs are already closing) rather
            # than the deferred thread the normal end-of-path uses.
            _render_path_plot(delay_s=0.0)
        stop_keys.set()
        if tty_saved is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN,
                                  tty_saved)
            except Exception:
                pass
        if device is not None:
            device.stop()
        if speedj_trace is not None:
            speedj_trace.stop()
        if follow is not None:
            follow.close()
        if force_session is not None and force_session.active:
            force_session.exit(settle_s=0.2)
        if contact_controller is not None:
            contact_controller.stop()
        if close_tcp_force is not None:
            close_tcp_force()
        try:
            # Under dtc `interface` *is* the backend and shutdown() is its stop():
            # zero tau, leave torque mode, stopScript, disconnect both clients.
            interface.shutdown()
        except Exception:
            pass
        if dtc_backend is not None and dtc_backend is not interface:
            # Handoff never happened (or failed): the backend may still hold RTDE.
            try:
                dtc_backend.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
