"""Averages full-grid predictions over time at each grid cell."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io

DUMMY_TIME_UNIX_SEC = 0
DATE_FORMAT = prediction_io.DATE_FORMAT

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with input predictions (one file per day).  '
    'Files therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will average predictions over all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Averaged predictions will be written '
    'here by `prediction_io.write_file`, to an exact location determined by '
    '`prediction_io.find_file` with a dummy date.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_input_dir_name, first_date_string, last_date_string,
         top_output_dir_name):
    """Averages full-grid predictions over time at each grid cell.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param top_output_dir_name: Same.
    """

    input_file_names = prediction_io.find_many_files(
        top_directory_name=top_input_dir_name,
        first_date_string=first_date_string, last_date_string=last_date_string,
        prefer_zipped=False, allow_other_format=True, radar_number=None,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True
    )

    prediction_dict = None
    num_times = 0

    for this_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = prediction_io.read_file(this_file_name)

        # this_prediction_dict[prediction_io.TARGET_MATRIX_KEY] = numpy.sum(
        #     this_prediction_dict[prediction_io.TARGET_MATRIX_KEY],
        #     axis=0, keepdims=True
        # )
        this_prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.sum(
            this_prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
            axis=0, keepdims=True
        )
        num_times += len(this_prediction_dict[prediction_io.VALID_TIMES_KEY])

        if prediction_dict is None:
            prediction_dict = copy.deepcopy(this_prediction_dict)
            continue

        # prediction_dict[prediction_io.TARGET_MATRIX_KEY] += (
        #     this_prediction_dict[prediction_io.TARGET_MATRIX_KEY]
        # )
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] += (
            this_prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
        )

    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][[0], ...]
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = numpy.minimum(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY], 0
    )

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] / num_times
    )
    prediction_dict[prediction_io.VALID_TIMES_KEY] = numpy.array(
        [DUMMY_TIME_UNIX_SEC], dtype=int
    )

    dummy_date_string = time_conversion.unix_sec_to_string(
        DUMMY_TIME_UNIX_SEC, DATE_FORMAT
    )
    output_file_name = prediction_io.find_file(
        top_directory_name=top_output_dir_name,
        valid_date_string=dummy_date_string, radar_number=None,
        prefer_zipped=False, allow_other_format=False,
        raise_error_if_missing=False
    )

    print('Writing average predictions to: "{0:s}"...'.format(output_file_name))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=prediction_dict[prediction_io.TARGET_MATRIX_KEY],
        forecast_probability_matrix=
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
        valid_times_unix_sec=prediction_dict[prediction_io.VALID_TIMES_KEY],
        latitudes_deg_n=prediction_dict[prediction_io.LATITUDES_KEY],
        longitudes_deg_e=prediction_dict[prediction_io.LONGITUDES_KEY],
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
