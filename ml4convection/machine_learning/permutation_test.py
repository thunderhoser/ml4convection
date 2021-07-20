"""Unit tests for permutation.py."""

import unittest
import numpy
from ml4convection.machine_learning import neural_net
from ml4convection.machine_learning import permutation

TOLERANCE = 1e-6

# The following constants are used to _permute_values and _depermute_values.
PREDICTOR_MATRIX = numpy.random.uniform(low=0., high=1., size=(10, 205, 205, 7))
CHANNEL_INDEX_TO_PERMUTE = 5
DUMMY_DATA_OPTION_DICT = {
    neural_net.INCLUDE_TIME_DIM_KEY: True
}


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_permute_values(self):
        """Ensures correct output from _permute_values."""

        new_predictor_matrix, permuted_example_indices = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX + 0.,
                channel_index=CHANNEL_INDEX_TO_PERMUTE,
                data_option_dict=DUMMY_DATA_OPTION_DICT,
                permuted_example_indices=None
            )
        )

        num_channels = new_predictor_matrix.shape[-1]
        indices_to_compare = (
            numpy.arange(num_channels) != CHANNEL_INDEX_TO_PERMUTE
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix[..., indices_to_compare],
            PREDICTOR_MATRIX[..., indices_to_compare], atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX + 0.,
            channel_index=CHANNEL_INDEX_TO_PERMUTE,
            data_option_dict=DUMMY_DATA_OPTION_DICT,
            permuted_example_indices=permuted_example_indices
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_depermute_values(self):
        """Ensures correct output from _depermute_values."""

        new_predictor_matrix, permuted_example_indices = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX + 0.,
                channel_index=CHANNEL_INDEX_TO_PERMUTE,
                data_option_dict=DUMMY_DATA_OPTION_DICT,
                permuted_example_indices=None
            )
        )

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX, atol=TOLERANCE
        ))

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            channel_index=CHANNEL_INDEX_TO_PERMUTE,
            data_option_dict=DUMMY_DATA_OPTION_DICT,
            permuted_example_indices=permuted_example_indices
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
