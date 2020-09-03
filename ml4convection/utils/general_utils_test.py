"""Unit tests for general_utils.py."""

import unittest
from ml4convection.utils import general_utils

# The following constants are used to test get_previous_date and get_next_date.
CURRENT_DATE_STRINGS = [
    '20191231', '20200101', '20200228', '20200229', '20200301'
]
PREVIOUS_DATE_STRINGS = [
    '20191230', '20191231', '20200227', '20200228', '20200229'
]
NEXT_DATE_STRINGS = [
    '20200101', '20200102', '20200229', '20200301', '20200302'
]


class GeneralUtilsTests(unittest.TestCase):
    """Each method is a unit test for general_utils.py."""

    def test_get_previous_date(self):
        """Ensures correct output from get_previous_date."""

        these_previous_date_strings = [
            general_utils.get_previous_date(d) for d in CURRENT_DATE_STRINGS
        ]
        self.assertTrue(these_previous_date_strings == PREVIOUS_DATE_STRINGS)

    def test_get_next_date(self):
        """Ensures correct output from get_next_date."""

        these_next_date_strings = [
            general_utils.get_next_date(d) for d in CURRENT_DATE_STRINGS
        ]
        self.assertTrue(these_next_date_strings == NEXT_DATE_STRINGS)


if __name__ == '__main__':
    unittest.main()
