"""Unit tests for evaluation_plotting.py."""

import unittest
import numpy
from ml4convection.plotting import evaluation_plotting as eval_plotting

TOLERANCE = 1e-6

# The following constants are used to test _get_positive_skill_area and
# _get_zero_skill_line.
MEAN_VALUE_IN_TRAINING = 0.2

X_COORDS_NO_SKILL = numpy.array([0, 1], dtype=float)
Y_COORDS_NO_SKILL = numpy.array([0.1, 0.6])

X_COORDS_LEFT = numpy.array([0, 0.2, 0.2, 0, 0])
Y_COORDS_LEFT = numpy.array([0, 0, 0.2, 0.1, 0])
X_COORDS_RIGHT = numpy.array([0.2, 1, 1, 0.2, 0.2])
Y_COORDS_RIGHT = numpy.array([0.2, 0.6, 1, 1, 0.2])

# The following constants are used to test _get_pofd_pod_grid, _get_sr_pod_grid,
# _csi_from_sr_and_pod, and _bias_from_sr_and_pod.
POD_GRID_SPACING = 0.5
POFD_GRID_SPACING = 0.25
SUCCESS_RATIO_GRID_SPACING = 0.25

POD_MATRIX = numpy.array([
    [0.75, 0.75, 0.75, 0.75],
    [0.25, 0.25, 0.25, 0.25]
])

SUCCESS_RATIO_MATRIX = numpy.array([
    [0.125, 0.375, 0.625, 0.875],
    [0.125, 0.375, 0.625, 0.875]
])

POFD_MATRIX = SUCCESS_RATIO_MATRIX + 0.

FREQUENCY_BIAS_MATRIX = numpy.array([
    [6, 2, 1.2, 6. / 7],
    [2, 2. / 3, 0.4, 2. / 7]
])

CSI_MATRIX = numpy.array([
    [25. / 3, 3, 0.6 + 4. / 3, 1. / 7 + 4. / 3],
    [11, 17. / 3, 4.6, 29. / 7]
]) ** -1


class EvaluationPlottingTests(unittest.TestCase):
    """Each method is a unit test for evaluation_plotting.py."""

    def test_get_positive_skill_area(self):
        """Ensures correct output from _get_positive_skill_area."""

        (
            these_x_coords_left, these_y_coords_left,
            these_x_coords_right, these_y_coords_right
        ) = eval_plotting._get_positive_skill_area(
            mean_value_in_training=MEAN_VALUE_IN_TRAINING,
            min_value_in_plot=0., max_value_in_plot=1.
        )

        self.assertTrue(numpy.allclose(
            these_x_coords_left, X_COORDS_LEFT, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_coords_left, Y_COORDS_LEFT, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_x_coords_right, X_COORDS_RIGHT, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_coords_right, Y_COORDS_RIGHT, atol=TOLERANCE
        ))

    def test_get_zero_skill_line(self):
        """Ensures correct output from _get_zero_skill_line."""

        these_x_coords, these_y_coords = eval_plotting._get_zero_skill_line(
            mean_value_in_training=MEAN_VALUE_IN_TRAINING,
            min_value_in_plot=0., max_value_in_plot=1.
        )

        self.assertTrue(numpy.allclose(
            these_x_coords, X_COORDS_NO_SKILL, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_coords, Y_COORDS_NO_SKILL, atol=TOLERANCE
        ))

    def test_get_pofd_pod_grid(self):
        """Ensures correct output from _get_pofd_pod_grid."""

        this_pofd_matrix, this_pod_matrix = eval_plotting._get_pofd_pod_grid(
            pofd_spacing=POFD_GRID_SPACING, pod_spacing=POD_GRID_SPACING
        )

        self.assertTrue(numpy.allclose(
            this_pofd_matrix, POFD_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_pod_matrix, POD_MATRIX, atol=TOLERANCE
        ))

    def test_get_sr_pod_grid(self):
        """Ensures correct output from _get_sr_pod_grid."""

        this_success_ratio_matrix, this_pod_matrix = (
            eval_plotting._get_sr_pod_grid(
                success_ratio_spacing=SUCCESS_RATIO_GRID_SPACING,
                pod_spacing=POD_GRID_SPACING
            )
        )

        self.assertTrue(numpy.allclose(
            this_success_ratio_matrix, SUCCESS_RATIO_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_pod_matrix, POD_MATRIX, atol=TOLERANCE
        ))

    def test_csi_from_sr_and_pod(self):
        """Ensures correct output from _csi_from_sr_and_pod."""

        this_csi_matrix = eval_plotting._csi_from_sr_and_pod(
            success_ratio_array=SUCCESS_RATIO_MATRIX, pod_array=POD_MATRIX
        )

        self.assertTrue(numpy.allclose(
            this_csi_matrix, CSI_MATRIX, atol=TOLERANCE
        ))

    def test_bias_from_sr_and_pod(self):
        """Ensures correct output from _bias_from_sr_and_pod."""

        this_bias_matrix = eval_plotting._bias_from_sr_and_pod(
            success_ratio_array=SUCCESS_RATIO_MATRIX, pod_array=POD_MATRIX
        )

        self.assertTrue(numpy.allclose(
            this_bias_matrix, FREQUENCY_BIAS_MATRIX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
