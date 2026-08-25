import unittest

from geodude_teleoperation.spacemouse import rising_button_indices


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
