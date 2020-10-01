"""Finds event (convection) frequency for given dilation distance."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import radar_io
from ml4convection.io import example_io
from ml4convection.io import twb_satellite_io
from ml4convection.utils import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

NUM_HOURS_PER_DAY = 24

TARGET_DIR_ARG_NAME = 'input_target_dir_name'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_px'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
MONTH_ARG_NAME = 'desired_month'
SPLIT_BY_HOUR_ARG_NAME = 'split_by_hour'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with target values.  Files therein will be '
    'found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
DILATION_DISTANCE_HELP_STRING = 'Dilation distance for convective pixels.'
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will include grids for all days in the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

MONTH_HELP_STRING = (
    'Will find event frequency only for this month (integer in 1...12).  To use'
    ' all months, leave this alone.'
)
SPLIT_BY_HOUR_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Boolean flag.  If 1, will find event '
    'frequency for each hour.  If 0, will use all hours.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `evaluation.write_climo_to_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=True,
    help=DILATION_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MONTH_ARG_NAME, type=int, required=False, default=-1,
    help=MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPLIT_BY_HOUR_ARG_NAME, type=int, required=False, default=0,
    help=SPLIT_BY_HOUR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_target_dir_name, first_date_string, last_date_string,
         dilation_distance_px, desired_month, split_by_hour, output_file_name):
    """Finds event (convection) frequency for given dilation distance.

    This is effectively the main method.

    :param top_target_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param dilation_distance_px: Same.
    :param desired_month: Same.
    :param split_by_hour: Same.
    :param output_file_name: Same.
    """

    if desired_month <= 0:
        desired_month = None

    if desired_month is not None:
        split_by_hour = False
        error_checking.assert_is_leq(desired_month, 12)

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string, raise_error_if_any_missing=False
    )
    date_strings = [
        example_io.file_name_to_date(f) for f in target_file_names
    ]

    if desired_month is not None:
        dates_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, evaluation.DATE_FORMAT)
            for t in date_strings
        ], dtype=int)

        months = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%m'))
            for t in dates_unix_sec
        ], dtype=int)

        good_indices = numpy.where(months == desired_month)[0]
        target_file_names = [target_file_names[k] for k in good_indices]

        del date_strings, dates_unix_sec, months

    first_target_dict = example_io.read_target_file(target_file_names[0])
    mask_file_name = first_target_dict[example_io.MASK_FILE_KEY]

    print('Reading mask from: "{0:s}"...'.format(mask_file_name))
    mask_dict = radar_io.read_mask_file(mask_file_name)
    mask_dict = radar_io.expand_to_satellite_grid(any_radar_dict=mask_dict)

    num_target_latitudes = len(first_target_dict[example_io.LATITUDES_KEY])
    num_full_latitudes = len(twb_satellite_io.GRID_LATITUDES_DEG_N)
    downsampling_factor = int(numpy.floor(
        float(num_full_latitudes) / num_target_latitudes
    ))

    if downsampling_factor > 1:
        print('Downsampling mask to {0:d}x spatial resolution...'.format(
            downsampling_factor
        ))
        mask_dict = radar_io.downsample_in_space(
            any_radar_dict=mask_dict, downsampling_factor=4
        )

    print('Eroding mask with erosion distance = {0:f} pixels...'.format(
        dilation_distance_px
    ))
    mask_matrix = evaluation.erode_binary_matrix(
        binary_matrix=mask_dict[radar_io.MASK_MATRIX_KEY],
        buffer_distance_px=dilation_distance_px
    )
    mask_matrix = mask_matrix.astype(bool)

    print(SEPARATOR_STRING)

    num_splits = NUM_HOURS_PER_DAY if split_by_hour else 1
    num_pixels_by_split = numpy.full(num_splits, 0, dtype=int)
    num_convective_pixels_by_split = numpy.full(num_splits, 0, dtype=int)

    for this_file_name in target_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        target_dict = example_io.read_target_file(this_file_name)
        valid_times_unix_sec = target_dict[example_io.VALID_TIMES_KEY]

        for i in range(len(valid_times_unix_sec)):
            target_matrix = evaluation.dilate_binary_matrix(
                binary_matrix=
                target_dict[example_io.TARGET_MATRIX_KEY][i, ...],
                buffer_distance_px=dilation_distance_px
            )
            target_matrix[mask_matrix == False] = -1

            if split_by_hour:
                j = int(
                    time_conversion.unix_sec_to_string(
                        valid_times_unix_sec[i], '%H'
                    )
                )
            else:
                j = 0

            num_pixels_by_split[j] += numpy.sum(target_matrix >= 0)
            num_convective_pixels_by_split[j] += numpy.sum(target_matrix == 1)
            this_frequency = (
                float(num_convective_pixels_by_split[j]) /
                num_pixels_by_split[j]
            )

            print((
                'Number of convective pixels{0:s} = {1:d} of {2:d} = {3:10f}'
            ).format(
                ' for hour {0:d}'.format(j) if split_by_hour else '',
                num_convective_pixels_by_split[j],
                num_pixels_by_split[j],
                this_frequency
            ))

        if this_file_name == target_file_names[-1]:
            print(SEPARATOR_STRING)
        else:
            print(MINOR_SEPARATOR_STRING)

    event_frequency_by_split = numpy.array([
        float(ncp) / np
        for ncp, np in zip(num_convective_pixels_by_split, num_pixels_by_split)
    ])

    for j in range(num_splits):
        print((
            'Number of convective pixels{0:s} = {1:d} of {2:d} = {3:10f}'
        ).format(
            ' for hour {0:d}'.format(j) if split_by_hour else '',
            num_convective_pixels_by_split[j],
            num_pixels_by_split[j],
            event_frequency_by_split[j]
        ))

    print(SEPARATOR_STRING)

    if os.path.isfile(output_file_name):
        print('Reading previous event frequencies from: "{0:s}"...'.format(
            output_file_name
        ))

        (
            event_frequency_overall,
            event_frequency_by_hour,
            event_frequency_by_month
        ) = evaluation.read_climo_from_file(output_file_name)
    else:
        event_frequency_overall = numpy.nan
        event_frequency_by_hour = numpy.full(24, numpy.nan)
        event_frequency_by_month = numpy.full(12, numpy.nan)

    if desired_month is not None:
        event_frequency_by_month[desired_month - 1] = (
            event_frequency_by_split[0]
        )
    elif split_by_hour:
        event_frequency_by_hour = event_frequency_by_split
    else:
        event_frequency_overall = event_frequency_by_split[0]

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    evaluation.write_climo_to_file(
        event_frequency_overall=event_frequency_overall,
        event_frequency_by_hour=event_frequency_by_hour,
        event_frequency_by_month=event_frequency_by_month,
        pickle_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        dilation_distance_px=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME
        ),
        desired_month=getattr(INPUT_ARG_OBJECT, MONTH_ARG_NAME),
        split_by_hour=bool(getattr(INPUT_ARG_OBJECT, SPLIT_BY_HOUR_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
