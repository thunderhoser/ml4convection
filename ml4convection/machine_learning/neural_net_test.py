"""Unit tests for neural_net.py."""

import unittest
from ml4convection.io import example_io
from ml4convection.machine_learning import neural_net

# The following constants are used to test _find_days_with_preprocessed_inputs.
TOP_PREDICTOR_DIR_NAME = 'foo'
TOP_TARGET_DIR_NAME = 'bar'

PREDICTOR_DATE_STRINGS = [
    '20200105', '20200103', '20200110', '20200115', '20200109', '20200104',
    '20200112', '20200113', '20200102', '20200106'
]
TARGET_DATE_STRINGS = [
    '20200109', '20200107', '20200111', '20200108', '20200113', '20200103',
    '20200112', '20200101', '20200110', '20200115'
]

PREDICTOR_FILE_NAMES = [
    example_io.find_predictor_file(
        top_directory_name=TOP_PREDICTOR_DIR_NAME, date_string=d,
        raise_error_if_missing=False
    ) for d in PREDICTOR_DATE_STRINGS
]

TARGET_FILE_NAMES = [
    example_io.find_target_file(
        top_directory_name=TOP_TARGET_DIR_NAME, date_string=d,
        raise_error_if_missing=False
    ) for d in TARGET_DATE_STRINGS
]

VALID_DATE_STRINGS_ZERO_LEAD = [
    '20200109', '20200113', '20200103', '20200112', '20200110', '20200115'
]
VALID_DATE_STRINGS_NONZERO_LEAD = ['20200113', '20200103', '20200110']


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_find_days_zero_lead(self):
        """Ensures correct output from _find_days_with_preprocessed_inputs.

        In this case, lead time is zero.
        """

        these_date_strings = neural_net._find_days_with_preprocessed_inputs(
            predictor_file_names=PREDICTOR_FILE_NAMES,
            target_file_names=TARGET_FILE_NAMES,
            lead_time_seconds=0
        )

        self.assertTrue(these_date_strings == VALID_DATE_STRINGS_ZERO_LEAD)

    def test_find_days_nonzero_lead(self):
        """Ensures correct output from _find_days_with_preprocessed_inputs.

        In this case, lead time is non-zero.
        """

        these_date_strings = neural_net._find_days_with_preprocessed_inputs(
            predictor_file_names=PREDICTOR_FILE_NAMES,
            target_file_names=TARGET_FILE_NAMES,
            lead_time_seconds=600
        )

        self.assertTrue(these_date_strings == VALID_DATE_STRINGS_NONZERO_LEAD)


if __name__ == '__main__':
    unittest.main()
