import unittest

import numpy as np
import trimesh

from geodude_teleoperation.path_plan import (
    PHASE_APPROACH,
    PHASE_DONE,
    PHASE_PLUNGE,
    PHASE_RETRACT,
    PHASE_TRAVERSE,
    PathRunner,
    plan_surface_path,
    preview_positions,
    project_to_surface,
)
from geodude_teleoperation.press_latch import (
    AllOf,
    DistanceLatch,
    ForceLatch,
    LatchState,
    PhaseLatch,
)


def _plate(z=0.0, half=0.5):
    """Flat plate in the z = ``z`` plane, +Z outward."""
    verts = np.array([
        [-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.mesh = _plate()

    def test_projects_picks_onto_the_surface(self):
        p, n = project_to_surface(self.mesh, [0.1, 0.1, 0.07])
        np.testing.assert_allclose(p, [0.1, 0.1, 0.0], atol=1e-9)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-9)

    def test_phase_order_and_standoff(self):
        plan = plan_surface_path(self.mesh, [0.0, 0.0, 0.05], [0.1, 0.0, 0.05],
                                 standoff_m=0.03, ds_m=0.05)
        phases = [w.phase for w in plan.waypoints]
        self.assertEqual(phases[0], PHASE_APPROACH)
        self.assertEqual(phases[1], PHASE_PLUNGE)
        self.assertEqual(phases[-1], PHASE_RETRACT)
        self.assertTrue(all(p == PHASE_TRAVERSE for p in phases[2:-1]))
        # Approach and retract sit exactly standoff above their contact points.
        np.testing.assert_allclose(plan.waypoints[0].xyz, [0.0, 0.0, 0.03], atol=1e-9)
        np.testing.assert_allclose(plan.waypoints[-1].xyz, [0.1, 0.0, 0.03], atol=1e-9)
        # The plunge lands on the surface.
        np.testing.assert_allclose(plan.waypoints[1].xyz, [0.0, 0.0, 0.0], atol=1e-9)

    def test_traverse_is_sampled_at_ds(self):
        plan = plan_surface_path(self.mesh, [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                                 standoff_m=0.02, ds_m=0.01)
        traverse = [w for w in plan.waypoints if w.phase == PHASE_TRAVERSE]
        self.assertEqual(len(traverse), 10)
        gaps = np.linalg.norm(
            np.diff(np.array([w.xyz for w in traverse]), axis=0), axis=1)
        np.testing.assert_allclose(gaps, 0.01, atol=1e-9)

    def test_press_phases(self):
        plan = plan_surface_path(self.mesh, [0.0, 0.0, 0.0], [0.05, 0.0, 0.0],
                                 standoff_m=0.02, ds_m=0.05)
        press = {w.phase: w.press_phase for w in plan.waypoints}
        self.assertFalse(press[PHASE_APPROACH])
        self.assertTrue(press[PHASE_PLUNGE])
        self.assertTrue(press[PHASE_TRAVERSE])
        self.assertFalse(press[PHASE_RETRACT])

    def test_normal_fn_overrides_the_mesh_normal(self):
        plan = plan_surface_path(
            self.mesh, [0.0, 0.0, 0.0], [0.05, 0.0, 0.0], standoff_m=0.02,
            ds_m=0.05, normal_fn=lambda p: np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(plan.waypoints[0].xyz, [0.0, 0.0, -0.02],
                                   atol=1e-9)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.mesh = _plate()
        self.plan = plan_surface_path(
            self.mesh, [0.0, 0.0, 0.0], [0.05, 0.0, 0.0],
            standoff_m=0.02, ds_m=0.01)

    def test_idle_until_started(self):
        runner = PathRunner(self.plan)
        step = runner.step(0.02, [0.0, 0.0, 0.05])
        np.testing.assert_allclose(step.delta_world, np.zeros(3))
        self.assertFalse(step.running)

    def test_step_is_speed_bounded(self):
        runner = PathRunner(self.plan, free_speed_m_s=0.02)
        runner.start()
        step = runner.step(0.01, [0.0, 0.0, 0.5])
        self.assertAlmostEqual(float(np.linalg.norm(step.delta_world)), 0.0002,
                               places=9)

    def test_contact_speed_applies_on_press_phases(self):
        runner = PathRunner(self.plan, free_speed_m_s=0.02,
                            contact_speed_m_s=0.005)
        runner.start()
        # Sit on the approach waypoint so the runner retires it and moves on to
        # the plunge, which is a press phase.
        first = self.plan.waypoints[0].xyz
        step = runner.step(0.01, first)
        self.assertEqual(step.phase, PHASE_PLUNGE)
        self.assertAlmostEqual(float(np.linalg.norm(step.delta_world)), 0.00005,
                               places=9)

    def test_walks_to_the_end_and_stops(self):
        runner = PathRunner(self.plan, free_speed_m_s=0.05,
                            contact_speed_m_s=0.05)
        runner.start()
        p = np.array(self.plan.waypoints[0].xyz, dtype=float)
        for _ in range(20000):
            step = runner.step(0.005, p)
            if step.done:
                break
            p = p + step.delta_world
        self.assertTrue(step.done)
        self.assertEqual(runner.phase(), PHASE_DONE)
        np.testing.assert_allclose(p, self.plan.waypoints[-1].xyz, atol=3e-3)

    def test_abort_stops_commanding(self):
        runner = PathRunner(self.plan)
        runner.start()
        runner.abort()
        step = runner.step(0.02, [0.0, 0.0, 0.05])
        np.testing.assert_allclose(step.delta_world, np.zeros(3))

    def test_preview_covers_the_plan(self):
        pts = preview_positions(self.plan, free_speed_m_s=0.05,
                                contact_speed_m_s=0.05, hz=50.0)
        self.assertGreater(len(pts), 10)
        np.testing.assert_allclose(pts[0], self.plan.waypoints[0].xyz, atol=1e-9)
        np.testing.assert_allclose(pts[-1], self.plan.waypoints[-1].xyz, atol=3e-3)


class LatchTest(unittest.TestCase):
    def _state(self, height_m, phase_press=True, fs=None):
        return LatchState(tcp_xyz=[0.0, 0.0, height_m],
                          surface_xyz=[0.0, 0.0, 0.0],
                          normal=[0.0, 0.0, 1.0],
                          phase_press=phase_press, fs=fs)

    def test_distance_latch_hysteresis(self):
        latch = DistanceLatch(engage_gap_m=0.002, release_gap_m=0.005)
        self.assertFalse(latch.update(self._state(0.010)))
        self.assertFalse(latch.update(self._state(0.003)))
        self.assertTrue(latch.update(self._state(0.002)))
        # Inside the hysteresis band it stays engaged.
        self.assertTrue(latch.update(self._state(0.004)))
        self.assertFalse(latch.update(self._state(0.006)))

    def test_distance_latch_ignores_force_entirely(self):
        latch = DistanceLatch()
        state = self._state(0.0, fs=float("nan"))
        self.assertTrue(latch.update(state))

    def test_distance_latch_holds_on_missing_geometry(self):
        latch = DistanceLatch()
        latch.update(self._state(0.0))
        self.assertTrue(latch.update(LatchState()))

    def test_phase_and_distance_must_both_agree(self):
        latch = AllOf([PhaseLatch(), DistanceLatch(engage_gap_m=0.002,
                                                   release_gap_m=0.005)])
        self.assertFalse(latch.update(self._state(0.0, phase_press=False)))
        self.assertTrue(latch.update(self._state(0.0, phase_press=True)))
        self.assertFalse(latch.update(self._state(0.0, phase_press=False)))

    def test_force_latch_still_works_for_spacemouse_teleop(self):
        latch = ForceLatch(15.0, engage_frac=0.5, release_frac=0.15)
        self.assertFalse(latch.update(self._state(0.0, fs=1.0)))
        self.assertTrue(latch.update(self._state(0.0, fs=8.0)))
        self.assertTrue(latch.update(self._state(0.0, fs=3.0)))
        self.assertFalse(latch.update(self._state(0.0, fs=1.0)))


if __name__ == "__main__":
    unittest.main()


class SessionRequestTest(unittest.TestCase):
    """Requests are recorded by viser's threads and consumed in tick().

    A tick that is skipped is a button that does nothing -- which is exactly
    what happened on the first hardware run: tick() was only called from the
    field-follow branch, unreachable until startup align finished, so 'p', 'r'
    and every GUI button queued forever.
    """

    def _session(self):
        from geodude_teleoperation.path_session import (
            PathSession, STATE_PICK, STATE_PREVIEW, STATE_READY, STATE_RUN)
        self.STATE_PICK = STATE_PICK
        self.STATE_PREVIEW = STATE_PREVIEW
        self.STATE_READY = STATE_READY
        self.STATE_RUN = STATE_RUN
        s = PathSession(vf=None, mesh=_plate(), use_field_normal=False,
                        standoff_m=0.02, ds_m=0.01)
        s.pick([0.0, 0.0, 0.0])
        s.pick([0.05, 0.0, 0.0])
        return s

    def test_preview_runs_even_when_execution_is_blocked(self):
        s = self._session()
        s.request_preview()
        s.tick(0.02, [0.0, 0.0, 0.02], allow_execute=False,
               block_reason="align has not finished")
        self.assertEqual(s.state, self.STATE_PREVIEW)
        self.assertIsNotNone(s.preview_xyz())

    def test_reset_and_abort_run_even_when_execution_is_blocked(self):
        s = self._session()
        s.request_abort()
        s.tick(0.02, [0.0, 0.0, 0.02], allow_execute=False)
        self.assertEqual(s.state, self.STATE_READY)
        s.request("reset")
        s.tick(0.02, [0.0, 0.0, 0.02], allow_execute=False)
        self.assertEqual(s.state, self.STATE_PICK)
        self.assertIsNone(s.p_start)

    def test_execute_is_refused_and_dropped_while_blocked(self):
        s = self._session()
        s.request_execute()
        s.tick(0.02, [0.0, 0.0, 0.02], allow_execute=False,
               block_reason="align has not finished")
        self.assertEqual(s.state, self.STATE_READY)
        # Dropped, not queued: it must not fire the moment the gate opens.
        s.tick(0.02, [0.0, 0.0, 0.02], allow_execute=True)
        self.assertEqual(s.state, self.STATE_READY)

    def test_execute_runs_once_the_gate_opens(self):
        s = self._session()
        s.request_execute()
        s.tick(0.02, [0.0, 0.0, 0.02], allow_execute=True)
        self.assertEqual(s.state, self.STATE_RUN)

    def test_execute_calls_back_exactly_once(self):
        from geodude_teleoperation.path_session import PathSession
        calls = []
        s = PathSession(vf=None, mesh=_plate(), use_field_normal=False,
                        standoff_m=0.02, ds_m=0.01,
                        on_execute_start=lambda: calls.append("start"),
                        on_execute_end=lambda: calls.append("end"))
        s.pick([0.0, 0.0, 0.0])
        s.pick([0.02, 0.0, 0.0])
        s.request_execute()
        s.tick(0.02, [0.0, 0.0, 0.02])
        self.assertEqual(calls, ["start"])
        s.request_abort()
        s.tick(0.02, [0.0, 0.0, 0.02])
        self.assertEqual(calls, ["start", "end"])


class DepthAndTouchTest(unittest.TestCase):
    """Depth sinks the commanded points; touch confirms they reached anything.

    The two are different failures. Depth is the scan being a few mm off, which
    geometry can absorb blindly. Touch is the scan being off by MORE than the
    depth, which only a force reading can catch -- and the consequence of
    missing it is the press handing over while the tool is still in the air,
    where a pure TCP force rotates the wrist instead of translating the arm.
    """

    def setUp(self):
        self.mesh = _plate()

    def test_depth_sinks_the_press_phases_only(self):
        plan = plan_surface_path(self.mesh, [0.0, 0.0, 0.0], [0.05, 0.0, 0.0],
                                 standoff_m=0.02, ds_m=0.05, depth_m=0.003)
        by_phase = {w.phase: w for w in plan.waypoints}
        self.assertAlmostEqual(by_phase[PHASE_PLUNGE].xyz[2], -0.003)
        self.assertAlmostEqual(by_phase[PHASE_TRAVERSE].xyz[2], -0.003)
        # Approach and retract keep their standoff above the surface.
        self.assertAlmostEqual(by_phase[PHASE_APPROACH].xyz[2], 0.02)
        self.assertAlmostEqual(by_phase[PHASE_RETRACT].xyz[2], 0.02)

    def test_depth_does_not_move_the_surface_the_latch_measures(self):
        plan = plan_surface_path(self.mesh, [0.0, 0.0, 0.0], [0.05, 0.0, 0.0],
                                 standoff_m=0.02, ds_m=0.05, depth_m=0.003)
        for w in plan.waypoints:
            self.assertAlmostEqual(w.surface_xyz[2], 0.0)

    def test_depth_bias_deepens_press_goals_and_leaves_free_ones_alone(self):
        plan = plan_surface_path(self.mesh, [0.0, 0.0, 0.0], [0.05, 0.0, 0.0],
                                 standoff_m=0.02, ds_m=0.05)
        runner = PathRunner(plan)
        runner.depth_bias = 0.004
        by_phase = {w.phase: w for w in plan.waypoints}
        self.assertAlmostEqual(runner.goal_of(by_phase[PHASE_PLUNGE])[2], -0.004)
        self.assertAlmostEqual(runner.goal_of(by_phase[PHASE_APPROACH])[2], 0.02)

    def _run_session(self, **kw):
        from geodude_teleoperation.path_session import PathSession
        s = PathSession(vf=None, mesh=self.mesh, use_field_normal=False,
                        standoff_m=0.02, ds_m=0.02, contact_speed_m_s=0.01, **kw)
        s.pick([0.0, 0.0, 0.0])
        s.pick([0.04, 0.0, 0.0])
        s.request_execute()
        return s

    def test_seek_deepens_until_the_force_arrives(self):
        s = self._run_session(touch_n=3.0, max_seek_m=0.01)
        p = np.array([0.0, 0.0, 0.02])
        # No force: the plunge keeps sinking instead of moving on.
        for _ in range(50):
            step = s.tick(0.01, p, tcp_xyz=p, fs=0.2)
            p = p + step.delta_world
        self.assertGreater(s.runner.depth_bias, 0.0)
        self.assertEqual(s.phase, PHASE_PLUNGE)
        self.assertFalse(s.press_engaged)
        # Force arrives: the bias freezes and the run proceeds.
        frozen = s.runner.depth_bias
        s.tick(0.01, p, tcp_xyz=p, fs=5.0)
        s.tick(0.01, p, tcp_xyz=p, fs=5.0)
        self.assertAlmostEqual(s.runner.depth_bias, frozen)

    def test_seek_aborts_rather_than_diving_forever(self):
        s = self._run_session(touch_n=3.0, max_seek_m=0.002)
        p = np.array([0.0, 0.0, 0.02])
        for _ in range(400):
            step = s.tick(0.01, p, tcp_xyz=p, fs=0.0)
            if step is None:
                break
            p = p + step.delta_world
        self.assertLessEqual(s.runner.depth_bias, 0.002 + 1e-9)
        self.assertFalse(s.running)

    def test_touch_gate_is_part_of_the_latch(self):
        s = self._run_session(touch_n=3.0)
        # On the surface, in a press phase, but nothing is being felt.
        p = np.array([0.0, 0.0, 0.0])
        for _ in range(5):
            s.tick(0.01, p, tcp_xyz=p, fs=0.1)
        self.assertFalse(s.press_engaged)

    def test_geometry_only_latch_is_the_default(self):
        s = self._run_session()
        p = np.array([0.0, 0.0, 0.02])
        for _ in range(400):
            step = s.tick(0.01, p, tcp_xyz=p)
            if step is None or step.done:
                break
            p = p + step.delta_world
        self.assertEqual(s.runner.depth_bias, 0.0)
