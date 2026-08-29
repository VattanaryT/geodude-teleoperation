"""On-robot DTC latch: patch stock ur_rtde script + install via setCustomScriptFile."""
from __future__ import annotations

from pathlib import Path

import pytest

from geodude_teleoperation.direct_torque import (
    DTC_LATCH_MARKER,
    install_dtc_latch_control_script,
    patch_rtde_control_script_for_dtc_latch,
)


# Minimal stock fragment with the hooks our patcher must rewrite.
_STOCK_FRAGMENT = """\
    global servo_thrd = 0
    global servoc_thrd = 0
    global speed_thrd = 0

    thread servo_thread():
        while (True):
            servoj(q, t=dt, lookahead_time=lh_time, gain=g)
        end
    end

    thread servoc_thread():
        while (True):
            servoc(servoc_target, a=1.0, v=0.1, r=0.0)
        end
    end

        elif cmd == 15:
            textmsg("speed_stop")
            deceleration_rate = read_input_float_reg(0)
            enter_critical
            is_speeding = 0
            kill speed_thrd
            speed_thrd = 0
            if speed_type == 0:
                stopl(deceleration_rate)
            else:
                stopj(deceleration_rate)
            end
            exit_critical
            textmsg("speed_stop done")
        elif cmd == 16:
            textmsg("servo_stop")
            deceleration_rate = read_input_float_reg(0)
            enter_critical
            is_servoing = 0
            kill servo_thrd
            kill servoc_thrd
            servo_thrd = 0
            servoc_thrd = 0
            exit_critical
            stopl(deceleration_rate)
            textmsg("servo_stop done")
        elif cmd == 66:
              # direct_torque
              dummy_op = 1
$5.23         torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
$5.23         torque = q_from_input_float_registers(0)
$5.23         friction_comp = True
$5.23         friction_comp_val = read_input_integer_reg(1)
$5.23         if friction_comp_val == 1:
$5.23             friction_comp = True
$5.23         elif friction_comp_val == 0:
$5.23             friction_comp = False
$5.23         end
$5.23         direct_torque(torque, friction_comp)
        elif cmd == 67:
              # get_mass_matrix
"""


def test_patch_injects_dtc_thread_and_latches_cmd66():
    out = patch_rtde_control_script_for_dtc_latch(_STOCK_FRAGMENT)
    assert DTC_LATCH_MARKER in out
    assert "thread dtc_thread():" in out
    assert "run dtc_thread()" in out
    assert "dtc_target = torque" in out
    # Stock one-shot apply must not remain as the only cmd-66 body.
    assert "direct_torque(torque, friction_comp)\n        elif cmd == 67" not in out
    assert "direct_torque(tau, fc)" in out or "direct_torque(tau, dtc_friction_comp)" in out
    assert "kill dtc_thrd" in out


def test_wrap_rtde_control_script_for_custom_upload():
    """ur_rtde setCustomScriptFile sends the file as-is (no def wrap)."""
    from geodude_teleoperation.direct_torque import wrap_rtde_control_script

    body = patch_rtde_control_script_for_dtc_latch(_STOCK_FRAGMENT)
    wrapped = wrap_rtde_control_script(body)
    assert wrapped.startswith("def rtde_control():\n")
    assert wrapped.rstrip().endswith("end")
    assert DTC_LATCH_MARKER in wrapped
    # Idempotent
    assert wrap_rtde_control_script(wrapped) == wrapped


def test_patch_is_idempotent():
    once = patch_rtde_control_script_for_dtc_latch(_STOCK_FRAGMENT)
    twice = patch_rtde_control_script_for_dtc_latch(once)
    assert once == twice


def test_install_dtc_latch_writes_and_uploads(tmp_path: Path):
    stock = tmp_path / "stock.script"
    stock.write_text(_STOCK_FRAGMENT, encoding="utf-8")
    uploaded: list[str] = []

    class FakeCtrl:
        def setCustomScriptFile(self, path: str) -> None:
            uploaded.append(path)

        def isProgramRunning(self) -> bool:
            return True

    out = install_dtc_latch_control_script(
        FakeCtrl(), stock_script=stock, out_dir=tmp_path
    )
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("def rtde_control():\n")
    assert DTC_LATCH_MARKER in text
    assert uploaded == [str(out)]


def test_install_requires_setCustomScriptFile():
    with pytest.raises(RuntimeError, match="setCustomScriptFile"):
        install_dtc_latch_control_script(object(), stock_script=_STOCK_FRAGMENT)


def test_install_fails_if_program_not_running(tmp_path: Path):
    stock = tmp_path / "stock.script"
    stock.write_text(_STOCK_FRAGMENT, encoding="utf-8")

    class FakeCtrl:
        def setCustomScriptFile(self, path: str) -> None:
            _ = path

        def isProgramRunning(self) -> bool:
            return False

    with pytest.raises(RuntimeError, match="program not running"):
        install_dtc_latch_control_script(
            FakeCtrl(),
            stock_script=stock,
            out_dir=tmp_path,
            ready_timeout_s=0.15,
        )


# --- publish-side staleness -------------------------------------------------
#
# The bound exists so a *stalled* control loop floats the arm. It is not
# allowed to chop a *live* one: measured on this rig the tick jittered to 25 ms
# against a 16 ms bound, so the publisher stepped tau to zero and back several
# times a second, and the on-robot latch applied every edge at 500 Hz. On the
# wrist (I ~= 0.05 kg.m^2) a 10 N.m pulse 22 ms wide is 4.4 rad/s of joint
# velocity nobody commanded -- the C283A2 "robot is moving when in stop mode"
# the pendant reported. Hence: decay, do not step.


def _streamer(**kw):
    from geodude_teleoperation.direct_torque import HostDirectTorqueStreamer

    kw.setdefault("stale_tau_s", 0.02)
    kw.setdefault("stale_zero_s", 0.06)
    return HostDirectTorqueStreamer(object(), **kw)


def test_fresh_tau_is_published_untouched():
    s = _streamer()
    assert s._stale_scale(0.0) == 1.0
    assert s._stale_scale(0.02) == 1.0


def test_a_late_tick_decays_tau_instead_of_stepping_it_to_zero():
    s = _streamer()
    mid = s._stale_scale(0.04)
    assert 0.0 < mid < 1.0
    assert abs(mid - 0.5) < 1e-9


def test_full_float_still_arrives_for_a_dead_loop():
    s = _streamer()
    assert s._stale_scale(0.06) == 0.0
    assert s._stale_scale(10.0) == 0.0


def test_the_zero_age_defaults_to_twice_the_bound():
    s = _streamer(stale_zero_s=None)
    assert s._zero_age() == pytest.approx(0.04)


def test_no_bound_means_the_raw_hold_the_rate_check_wants():
    s = _streamer(stale_tau_s=None, stale_zero_s=None)
    assert s._stale_scale(10.0) == 1.0


def test_a_zero_age_inside_the_bound_is_refused():
    with pytest.raises(ValueError):
        _streamer(stale_tau_s=0.05, stale_zero_s=0.05)


def test_the_publisher_reports_its_own_stale_count_separately():
    """The console line printed the backend's read-stale (0) while this counted 34."""
    s = _streamer()
    st = s.stats()
    assert st["n_stale"] == 0 and st["n_zeroed"] == 0
    assert st["stale_tau_ms"] == pytest.approx(20.0)
