"""Unit tests for radar_utils.py."""

import unittest
import numpy
from ml4convection.utils import radar_utils

FIRST_GRID_LATITUDES_DEG_N = numpy.linspace(20, 30, num=21, dtype=float)
FIRST_GRID_LONGITUDES_DEG_E = numpy.linspace(115, 125, num=11, dtype=float)
FIRST_ROW_INDICES = numpy.array([10, 8, 6, 4], dtype=int)
FIRST_COLUMN_INDICES = numpy.array([7, 7, 5, 6], dtype=int)

SECOND_GRID_LATITUDES_DEG_N = numpy.linspace(20, 30, num=41, dtype=float)
SECOND_GRID_LONGITUDES_DEG_E = numpy.linspace(115, 125, num=41, dtype=float)
SECOND_ROW_INDICES = numpy.array([20, 16, 13, 8], dtype=int)
SECOND_COLUMN_INDICES = numpy.array([27, 26, 20, 23], dtype=int)


class RadarUtilsTests(unittest.TestCase):
    """Each method is a unit test for radar_utils.py."""

    def test_radar_sites_to_grid_points_first(self):
        """Ensures correct output from radar_sites_to_grid_points.

        In this case, using first grid.
        """

        these_row_indices, these_column_indices = (
            radar_utils.radar_sites_to_grid_points(
                grid_latitudes_deg_n=FIRST_GRID_LATITUDES_DEG_N,
                grid_longitudes_deg_e=FIRST_GRID_LONGITUDES_DEG_E
            )
        )

        self.assertTrue(numpy.array_equal(
            these_row_indices, FIRST_ROW_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_column_indices, FIRST_COLUMN_INDICES
        ))

    def test_radar_sites_to_grid_points_second(self):
        """Ensures correct output from radar_sites_to_grid_points.

        In this case, using second grid.
        """

        these_row_indices, these_column_indices = (
            radar_utils.radar_sites_to_grid_points(
                grid_latitudes_deg_n=SECOND_GRID_LATITUDES_DEG_N,
                grid_longitudes_deg_e=SECOND_GRID_LONGITUDES_DEG_E
            )
        )

        self.assertTrue(numpy.array_equal(
            these_row_indices, SECOND_ROW_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_column_indices, SECOND_COLUMN_INDICES
        ))


if __name__ == '__main__':
    unittest.main()
