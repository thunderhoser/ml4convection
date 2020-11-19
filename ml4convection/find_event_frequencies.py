"""Finds event (convection) frequencies for given dilation distance."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import example_io
import climatology_io
import radar_utils
import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

NUM_HOURS_PER_DAY = 24
NUM_MONTHS_PER_YEAR = 12
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

TARGET_DIR_ARG_NAME = 'input_target_dir_name'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_px'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with target values.  Files therein will be '
    'found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will find event frequencies on partial '
    'radar-centered (full) grids.'
)
DILATION_DISTANCE_HELP_STRING = 'Dilation distance for convective pixels.'
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will include grids for all days in the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `climatology_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=True,
    help=USE_PARTIAL_GRIDS_HELP_STRING
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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_target_dir_name, use_partial_grids, first_date_string,
         last_date_string, dilation_distance_px, output_file_name):
    """Finds event (convection) frequencies for given dilation distance.

    This is effectively the main method.

    :param top_target_dir_name: See documentation at top of file.
    :param use_partial_grids: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param dilation_distance_px: Same.
    :param output_file_name: Same.
    """

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        radar_number=0 if use_partial_grids else None,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    date_strings = [example_io.file_name_to_date(f) for f in target_file_names]

    if use_partial_grids:
        for k in range(1, NUM_RADARS):
            target_file_names += [
                example_io.find_target_file(
                    top_directory_name=top_target_dir_name, date_string=d,
                    radar_number=k, prefer_zipped=False,
                    allow_other_format=True, raise_error_if_missing=True
                ) for d in date_strings
            ]

    first_target_dict = example_io.read_target_file(target_file_names[0])
    mask_matrix = first_target_dict[example_io.MASK_MATRIX_KEY]

    print('Eroding mask with erosion distance = {0:f} pixels...'.format(
        dilation_distance_px
    ))
    mask_matrix = general_utils.erode_binary_matrix(
        binary_matrix=mask_matrix, buffer_distance_px=dilation_distance_px
    )
    mask_matrix = mask_matrix.astype(bool)

    print(SEPARATOR_STRING)

    num_examples_overall = 0
    num_pos_examples_overall = 0
    num_examples_by_hour = numpy.full(NUM_HOURS_PER_DAY, 0, dtype=int)
    num_pos_examples_by_hour = numpy.full(NUM_HOURS_PER_DAY, 0, dtype=int)
    num_examples_by_month = numpy.full(NUM_MONTHS_PER_YEAR, 0, dtype=int)
    num_pos_examples_by_month = numpy.full(NUM_MONTHS_PER_YEAR, 0, dtype=int)
    num_examples_by_pixel = numpy.full(mask_matrix.shape, 0, dtype=int)
    num_pos_examples_by_pixel = numpy.full(mask_matrix.shape, 0, dtype=int)

    for this_file_name in target_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        target_dict = example_io.read_target_file(this_file_name)
        valid_times_unix_sec = target_dict[example_io.VALID_TIMES_KEY]

        for i in range(len(valid_times_unix_sec)):
            target_matrix = general_utils.dilate_binary_matrix(
                binary_matrix=
                target_dict[example_io.TARGET_MATRIX_KEY][i, ...],
                buffer_distance_px=dilation_distance_px
            )
            target_matrix[mask_matrix == False] = -1

            num_examples_by_pixel += (target_matrix >= 0).astype(int)
            num_pos_examples_by_pixel += (target_matrix == 1).astype(int)

            num_examples_overall += numpy.sum(target_matrix >= 0)
            num_pos_examples_overall += numpy.sum(target_matrix == 1)

            hour_index = int(time_conversion.unix_sec_to_string(
                valid_times_unix_sec[i], '%H'
            ))
            num_examples_by_hour[hour_index] += numpy.sum(target_matrix >= 0)
            num_pos_examples_by_hour[hour_index] += numpy.sum(
                target_matrix == 1
            )

            month_index = -1 + int(time_conversion.unix_sec_to_string(
                valid_times_unix_sec[i], '%m'
            ))
            num_examples_by_month[month_index] += numpy.sum(target_matrix >= 0)
            num_pos_examples_by_month[month_index] += numpy.sum(
                target_matrix == 1
            )

            print((
                'Number of convective examples overall = {0:d} of {1:d} = '
                '{2:10f}'
            ).format(
                num_pos_examples_overall, num_examples_overall,
                float(num_pos_examples_overall) / num_examples_overall
            ))

            this_freq = (
                float(num_pos_examples_by_hour[hour_index]) /
                num_examples_by_hour[hour_index]
            )

            print((
                'Number of convective examples for hour {0:d} = {1:d} of {2:d}'
                ' = {3:10f}'
            ).format(
                hour_index, num_pos_examples_by_hour[hour_index],
                num_examples_by_hour[hour_index], this_freq
            ))

            this_freq = (
                float(num_pos_examples_by_month[month_index]) /
                num_examples_by_month[month_index]
            )

            print((
                'Number of convective examples for month {0:d} = {1:d} of {2:d}'
                ' = {3:10f}'
            ).format(
                month_index + 1, num_pos_examples_by_month[month_index],
                num_examples_by_month[month_index], this_freq
            ))

        if this_file_name == target_file_names[-1]:
            print(SEPARATOR_STRING)
        else:
            print(MINOR_SEPARATOR_STRING)

    num_examples_by_hour = num_examples_by_hour.astype(float)
    num_examples_by_month = num_examples_by_month.astype(float)
    num_examples_by_pixel = num_examples_by_pixel.astype(float)

    num_examples_by_hour[num_examples_by_hour == 0] = numpy.nan
    num_examples_by_month[num_examples_by_month == 0] = numpy.nan
    num_examples_by_pixel[num_examples_by_pixel == 0] = numpy.nan

    event_frequency_overall = (
        float(num_pos_examples_overall) / num_examples_overall
    )
    event_frequency_by_hour = (
        num_pos_examples_by_hour.astype(float) / num_examples_by_hour
    )
    event_frequency_by_month = (
        num_pos_examples_by_month.astype(float) / num_examples_by_month
    )
    event_frequency_by_pixel = (
        num_pos_examples_by_pixel.astype(float) / num_examples_by_pixel
    )

    print((
        'Number of convective examples overall = {0:d} of {1:d} = {2:10f}\n'
    ).format(
        num_pos_examples_overall, num_examples_overall,
        event_frequency_overall
    ))

    for j in range(NUM_HOURS_PER_DAY):
        this_num_examples = (
            0 if numpy.isnan(num_examples_by_hour[j])
            else int(numpy.round(num_examples_by_hour[j]))
        )

        print((
            'Number of convective examples for hour {0:d} = {1:d} of {2:d} = '
            '{3:10f}'
        ).format(
            j, num_pos_examples_by_hour[j], this_num_examples,
            event_frequency_by_hour[j]
        ))

    print('\n')

    for j in range(NUM_MONTHS_PER_YEAR):
        this_num_examples = (
            0 if numpy.isnan(num_examples_by_month[j])
            else int(numpy.round(num_examples_by_month[j]))
        )

        print((
            'Number of convective examples for month {0:d} = {1:d} of {2:d} = '
            '{3:10f}'
        ).format(
            j + 1, num_pos_examples_by_month[j], this_num_examples,
            event_frequency_by_month[j]
        ))

    print((
        'Number of pixels with event frequency of NaN = {0:d} of {1:d}'
    ).format(
        numpy.sum(numpy.isnan(event_frequency_by_pixel)),
        event_frequency_by_pixel.size
    ))

    print((
        'Min/mean/max event frequency over all pixels = '
        '{0:10f}, {1:10f}, {2:10f}'
    ).format(
        numpy.nanmin(event_frequency_by_pixel),
        numpy.nanmean(event_frequency_by_pixel),
        numpy.nanmax(event_frequency_by_pixel)
    ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    climatology_io.write_file(
        event_frequency_overall=event_frequency_overall,
        event_frequency_by_hour=event_frequency_by_hour,
        event_frequency_by_month=event_frequency_by_month,
        event_frequency_by_pixel=event_frequency_by_pixel,
        latitudes_deg_n=first_target_dict[example_io.LATITUDES_KEY],
        longitudes_deg_e=first_target_dict[example_io.LONGITUDES_KEY],
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        use_partial_grids=bool(getattr(
            INPUT_ARG_OBJECT, USE_PARTIAL_GRIDS_ARG_NAME
        )),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        dilation_distance_px=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
