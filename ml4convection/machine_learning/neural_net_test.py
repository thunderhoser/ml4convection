"""Unit tests for neural_net.py."""

import unittest
from ml4convection.io import radar_io
from ml4convection.io import satellite_io
from ml4convection.machine_learning import neural_net

# The following constants are used to test _find_days_with_raw_inputs.
TOP_SATELLITE_DIR_NAME = 'foo'
TOP_ECHO_CLASSIFN_DIR_NAME = 'bar'

SATELLITE_DATE_STRINGS = [
    '20200105', '20200103', '20200110', '20200115', '20200109', '20200104',
    '20200112', '20200113', '20200102', '20200106'
]
ECHO_CLASSIFN_DATE_STRINGS = [
    '20200109', '20200107', '20200111', '20200108', '20200113', '20200103',
    '20200112', '20200101', '20200110', '20200115'
]

SATELLITE_FILE_NAMES = [
    satellite_io.find_file(
        top_directory_name=TOP_SATELLITE_DIR_NAME, valid_date_string=d,
        raise_error_if_missing=False
    ) for d in SATELLITE_DATE_STRINGS
]

ECHO_CLASSIFN_FILE_NAMES = [
    radar_io.find_file(
        top_directory_name=TOP_ECHO_CLASSIFN_DIR_NAME, valid_date_string=d,
        file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
        raise_error_if_missing=False
    ) for d in ECHO_CLASSIFN_DATE_STRINGS
]

VALID_DATE_STRINGS_ZERO_LEAD = [
    '20200109', '20200113', '20200103', '20200112', '20200110', '20200115'
]
VALID_DATE_STRINGS_NONZERO_LEAD = ['20200113', '20200103', '20200110']


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_find_days_zero_lead(self):
        """Ensures correct output from _find_days_with_raw_inputs.

        In this case, lead time is zero.
        """

        these_date_strings = neural_net._find_days_with_raw_inputs(
            satellite_file_names=SATELLITE_FILE_NAMES,
            echo_classifn_file_names=ECHO_CLASSIFN_FILE_NAMES,
            lead_time_seconds=0
        )

        self.assertTrue(these_date_strings == VALID_DATE_STRINGS_ZERO_LEAD)

    def test_find_days_nonzero_lead(self):
        """Ensures correct output from _find_days_with_raw_inputs.

        In this case, lead time is non-zero.
        """

        these_date_strings = neural_net._find_days_with_raw_inputs(
            satellite_file_names=SATELLITE_FILE_NAMES,
            echo_classifn_file_names=ECHO_CLASSIFN_FILE_NAMES,
            lead_time_seconds=600
        )

        self.assertTrue(these_date_strings == VALID_DATE_STRINGS_NONZERO_LEAD)


if __name__ == '__main__':
    unittest.main()
