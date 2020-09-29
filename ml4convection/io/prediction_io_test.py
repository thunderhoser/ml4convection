"""Unit tests for prediction_io.py."""

import unittest
from ml4convection.io import prediction_io

TOP_DIRECTORY_NAME = 'foo'
VALID_DATE_STRING = '19670502'
PREDICTION_FILE_NAME = 'foo/1967/predictions_19670502.nc'


class PredictionIoTests(unittest.TestCase):
    """Each method is a unit test for prediction_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME)

    def test_file_name_to_date(self):
        """Ensures correct output from file_name_to_date."""

        self.assertTrue(
            prediction_io.file_name_to_date(PREDICTION_FILE_NAME) ==
            VALID_DATE_STRING
        )


if __name__ == '__main__':
    unittest.main()
