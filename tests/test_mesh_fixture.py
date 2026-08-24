import unittest

import numpy as np

from geodude_teleoperation.mesh_fixture import (
    compute_resting_orientation,
    table_placement_translation,
)


def _unit_hemisphere_points(n_theta=36, n_phi=18):
    """Unit dome standing on z=0: pole at +z, equatorial rim in the xy plane."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phis = np.linspace(0.0, 0.5 * np.pi, n_phi)
    points = [
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        for phi in phis for theta in thetas
    ]
    return np.asarray(points, dtype=np.float64)


def _place(points, table_xyz, rotation=None):
    if rotation is None:
        rotation = compute_resting_orientation(points)
    translation = table_placement_translation(points, rotation, table_xyz)
    return points @ np.asarray(rotation).T + translation


def _top_is_peak_not_rim(world):
    """True for a dome (highest point on axis); false for a bowl (highest at the rim)."""
    top = world[np.argmax(world[:, 2])]
    xy_center = world.mean(axis=0)[:2]
    top_radius = np.linalg.norm(top[:2] - xy_center)
    low = world[:, 2] < world[:, 2].min() + 0.08 * (world[:, 2].max() - world[:, 2].min())
    rim_radius = np.linalg.norm(world[low][:, :2] - xy_center, axis=1).mean()
    return top_radius < 0.5 * rim_radius


class RestingOrientationTest(unittest.TestCase):
    def test_upright_hemisphere_stays_a_dome_on_the_table(self):
        table = np.array([0.2, -0.4, 0.3])
        world = _place(_unit_hemisphere_points(), table)
        self.assertGreaterEqual(world[:, 2].min() + 1e-9, table[2])
        self.assertLess(abs(world[:, 2].min() - table[2]), 1e-9)
        self.assertTrue(_top_is_peak_not_rim(world))

    def test_inverted_hemisphere_is_flipped_to_a_dome_on_the_table(self):
        table = np.array([0.0, 0.0, 0.0])
        bowl = _unit_hemisphere_points()
        bowl[:, 2] *= -1.0
        world = _place(bowl, table)
        self.assertGreaterEqual(world[:, 2].min() + 1e-9, table[2])
        self.assertTrue(_top_is_peak_not_rim(world))

    def test_old_centered_z_formula_is_not_used(self):
        """load_object_mesh applies R then +t with no centering. Mixing a centered
        min-z with the raw centroid z hangs a flipped hemisphere below the table."""
        points = _unit_hemisphere_points()
        rotation = compute_resting_orientation(points)
        table = np.array([1.0, 2.0, 0.5])
        translation = table_placement_translation(points, rotation, table)
        world = points @ rotation.T + translation
        self.assertGreaterEqual(world[:, 2].min() + 1e-9, table[2])
        np.testing.assert_allclose(world.mean(axis=0)[:2], table[:2] + np.array([0.15, 0.0]))
