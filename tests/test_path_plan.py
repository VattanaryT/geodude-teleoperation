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
