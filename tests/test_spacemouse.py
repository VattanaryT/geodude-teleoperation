import unittest
from unittest import mock

from geodude_teleoperation.spacemouse import (
    rising_button_indices,
    resolve_spacemouse_path,
)


class ResolveSpacemousePathTest(unittest.TestCase):
    @mock.patch("geodude_teleoperation.spacemouse.list_spacemouse_paths")
    def test_explicit_path(self, list_paths):
        list_paths.return_value = {"/dev/hidraw0": "SpaceMouseCompact"}
        self.assertEqual(
            resolve_spacemouse_path("/dev/hidraw0"), "/dev/hidraw0")

    @mock.patch("geodude_teleoperation.spacemouse.list_spacemouse_paths")
    def test_device_index(self, list_paths):
        list_paths.return_value = {
            "/dev/hidraw0": "SpaceMouseCompact",
            "/dev/hidraw5": "SpaceMouseCompact",
        }
        self.assertEqual(resolve_spacemouse_path(None, device_index=1), "/dev/hidraw5")


class RisingButtonIndicesTest(unittest.TestCase):
    def test_held_at_start_does_not_fire(self):
        prev, edges = rising_button_indices(None, (1, 0))
        self.assertEqual(prev, (1, 0))
        self.assertEqual(edges, ())

    def test_rising_edge(self):
        _, edges = rising_button_indices((0, 0), (1, 0))
        self.assertEqual(edges, (0,))

    def test_click_inside_drain_is_not_lost(self):
        prev, edges = rising_button_indices((0, 0), (0, 0))
        self.assertEqual(edges, ())
        prev, edges = rising_button_indices(prev, (1, 0))
        self.assertEqual(edges, (0,))
        prev, edges = rising_button_indices(prev, (0, 0))
        self.assertEqual(edges, ())

    def test_hold_does_not_repeat(self):
        prev, edges = rising_button_indices((0, 0), (1, 0))
        self.assertEqual(edges, (0,))
        _, edges = rising_button_indices(prev, (1, 0))
        self.assertEqual(edges, ())


if __name__ == "__main__":
    unittest.main()
