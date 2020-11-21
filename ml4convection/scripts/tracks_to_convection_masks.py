"""Converts storm-tracking data to convection masks (one per time step)."""

import os
import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_classification as echo_classifn
from ml4convection.io import radar_io
from ml4convection.io import twb_radar_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATE_FORMAT = radar_io.DATE_FORMAT
DUMMY_OPTION_DICT = echo_classifn.DEFAULT_OPTION_DICT
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

DAYS_TO_SECONDS = 86400
TIME_INTERVAL_SEC = twb_radar_io.TIME_INTERVAL_SEC
GRID_LATITUDES_DEG_N = twb_radar_io.GRID_LATITUDES_DEG_N
GRID_LONGITUDES_DEG_E = twb_radar_io.GRID_LONGITUDES_DEG_E

INPUT_DIR_ARG_NAME = 'input_tracking_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_DIR_ARG_NAME = 'output_mask_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking data.  Files therein will '
    'be found by `storm_tracking_io.find_file` and read by '
    '`storm_tracking_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  This script will create one mask for each time '
    'step in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files will be written here by '
    '`radar_io.write_echo_classifn_file`, to exact locations determined by '
    '`radar_io.find_file`.'
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


def _tracks_to_masks_one_day(
        top_tracking_dir_name, date_string, top_echo_classifn_dir_name):
    """Converts tracking data to masks for one day.

    :param top_tracking_dir_name: See documentation at top of file.
    :param date_string: Same.
    :param top_echo_classifn_dir_name: Same.
    :raises: ValueError: if no tracking files are found.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        date_string, DATE_FORMAT
    )
    last_time_unix_sec = (
        first_time_unix_sec + DAYS_TO_SECONDS - TIME_INTERVAL_SEC
    )
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    tracking_file_names = [
        tracking_io.find_file(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            valid_time_unix_sec=t,
            spc_date_string=time_conversion.time_to_spc_date_string(t),
            raise_error_if_missing=False
        ) for t in valid_times_unix_sec
    ]
    tracking_file_names = [
        f for f in tracking_file_names if os.path.isfile(f)
    ]

    if len(tracking_file_names) == 0:
        raise ValueError(
            'Could not find any tracking files for date {0:s}.'.format(
                date_string
            )
        )

    valid_times_unix_sec = numpy.array([
        tracking_io.file_name_to_time(f) for f in tracking_file_names
    ], dtype=int)

    num_times = len(valid_times_unix_sec)
    num_grid_rows = len(GRID_LATITUDES_DEG_N)
    num_grid_columns = len(GRID_LONGITUDES_DEG_E)

    convective_flag_matrix = numpy.full(
        (num_times, num_grid_rows, num_grid_columns), False, dtype=bool
    )

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(
            tracking_file_names[i]
        ))
        storm_object_table = tracking_io.read_file(tracking_file_names[i])
        num_storm_objects = len(storm_object_table.index)

        for j in range(num_storm_objects):
            these_rows = storm_object_table[
                tracking_utils.ROWS_IN_STORM_COLUMN
            ].values[j]

            these_columns = storm_object_table[
                tracking_utils.COLUMNS_IN_STORM_COLUMN
            ].values[j]

            convective_flag_matrix[i, ...][these_rows, these_columns] = True

    convective_flag_matrix = numpy.flip(convective_flag_matrix, axis=1)

    output_file_name = radar_io.find_file(
        top_directory_name=top_echo_classifn_dir_name,
        valid_date_string=date_string,
        file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
        prefer_zipped=False, allow_other_format=False,
        raise_error_if_missing=False
    )

    print('\nWriting results to: "{0:s}"...'.format(output_file_name))
    radar_io.write_echo_classifn_file(
        netcdf_file_name=output_file_name,
        convective_flag_matrix=convective_flag_matrix,
        latitudes_deg_n=GRID_LATITUDES_DEG_N,
        longitudes_deg_e=GRID_LONGITUDES_DEG_E,
        valid_times_unix_sec=valid_times_unix_sec,
        option_dict=DUMMY_OPTION_DICT
    )

    radar_io.compress_file(output_file_name)
    os.remove(output_file_name)


def _run(top_tracking_dir_name, first_date_string, last_date_string,
         top_echo_classifn_dir_name):
    """Converts storm-tracking data to convection masks (one per time step).

    This is effectively the main method.

    :param top_tracking_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param top_echo_classifn_dir_name: Same.
    :raises: ValueError: if no tracking files are found.
    """

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    for this_date_string in date_strings:
        _tracks_to_masks_one_day(
            top_tracking_dir_name=top_tracking_dir_name,
            date_string=this_date_string,
            top_echo_classifn_dir_name=top_echo_classifn_dir_name
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
