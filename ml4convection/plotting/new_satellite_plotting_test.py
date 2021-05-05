"""Unit tests for new_satellite_plotting.py."""

import unittest
import numpy
from ml4convection.plotting import new_satellite_plotting

TOLERANCE = 1e-6
CENTER_LATITUDES_DEG_N = numpy.array([30, 35, 42, 47], dtype=float)
EDGE_LATITUDES_DEG_N = numpy.array([27.5, 32.5, 38.5, 44.5, 49.5])


class NewSatellitePlottingTests(unittest.TestCase):
    """Each method is a unit test for new_satellite_plotting.py."""

    def test_grid_points_to_edges(self):
        """Ensures correct output from _grid_points_to_edges."""

        these_latitudes_deg_n = new_satellite_plotting._grid_points_to_edges(
            CENTER_LATITUDES_DEG_N
        )
        self.assertTrue(numpy.allclose(
            these_latitudes_deg_n, EDGE_LATITUDES_DEG_N, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
