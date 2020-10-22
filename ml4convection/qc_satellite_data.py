"""Quality-controls satellite data (gets rid of streaks in band 8)."""

import os
import sys
import argparse
import numpy
from scipy.ndimage import label as label_image

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import satellite_io
import general_utils
import standalone_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_FOR_MESSAGES = '%Y-%m-%d-%H%M'

BAND_NUMBER = 8

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
HALF_WINDOW_SIZE_ARG_NAME = 'half_window_size_px'
MIN_DIFFERENCE_ARG_NAME = 'min_temperature_diff_kelvins'
MIN_REGION_SIZE_ARG_NAME = 'min_region_size_px'
OUTPUT_DIR_ARG_NAME = 'output_satellite_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, with non-quality-controlled data.  Files therein '
    'will be found by `satellite_io.find_file` and read by '
    '`satellite_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will QC satellite data for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

HALF_WINDOW_SIZE_HELP_STRING = (
    'Size of half-window (pixels) for filters, both the smoothing filter and '
    'max-filter.'
)
MIN_DIFFERENCE_HELP_STRING = (
    'Minimum brightness-temperature difference.  Pixels with '
    '|T - T_smooth| >= `{0:s}`, where T is the original temperature and '
    'T_smooth is the smoothed temperature, will be considered bad data.'
).format(MIN_DIFFERENCE_ARG_NAME)

MIN_REGION_SIZE_HELP_STRING = (
    'Minimum region size.  Connected regions of >= `{0:s}` bad pixels will be '
    'interpolated out.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Quality-controlled data will be written here by'
    ' `satellite_io.write_file`, to exact locations determined by '
    '`satellite_io.find_file`.'
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
    '--' + HALF_WINDOW_SIZE_ARG_NAME, type=int, required=False, default=2,
    help=HALF_WINDOW_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_DIFFERENCE_ARG_NAME, type=float, required=False, default=1.,
    help=MIN_DIFFERENCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_REGION_SIZE_ARG_NAME, type=int, required=False, default=1000,
    help=MIN_REGION_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _qc_data_one_time(
        brightness_temp_matrix_kelvins, half_window_size_px,
        min_temperature_diff_kelvins, min_region_size_px):
    """Quality-controls satellite data for one time.

    M = number of rows in grid
    N = number of columns in grid

    :param brightness_temp_matrix_kelvins: M-by-N numpy array of brightness
        temperatures.
    :param half_window_size_px: See documentation at top of file.
    :param min_temperature_diff_kelvins: Same.
    :param min_region_size_px: Same.
    :return: brightness_temp_matrix_kelvins: Quality-controlled version of input
        (same shape).
    """

    window_size_px = 2 * half_window_size_px + 1

    orig_temp_matrix_kelvins = brightness_temp_matrix_kelvins + 0.
    brightness_temp_matrix_kelvins = numpy.expand_dims(
        brightness_temp_matrix_kelvins, axis=(0, -1)
    )

    mean_kernel_matrix = numpy.full((window_size_px, window_size_px), 1.)
    mean_kernel_matrix = mean_kernel_matrix / mean_kernel_matrix.size
    mean_kernel_matrix = numpy.expand_dims(mean_kernel_matrix, axis=-1)
    mean_kernel_matrix = numpy.expand_dims(mean_kernel_matrix, axis=-1)

    mean_temp_matrix_kelvins = standalone_utils.do_2d_convolution(
        feature_matrix=brightness_temp_matrix_kelvins,
        kernel_matrix=mean_kernel_matrix,
        pad_edges=True, stride_length_px=1
    )

    j = half_window_size_px
    mean_temp_matrix_kelvins[:, j:-j, j:-j, :] = (
        brightness_temp_matrix_kelvins[:, j:-j, j:-j, :]
    )

    absolute_diff_matrix_kelvins = numpy.absolute(
        mean_temp_matrix_kelvins - brightness_temp_matrix_kelvins
    )
    absolute_diff_matrix_kelvins = standalone_utils.do_2d_pooling(
        feature_matrix=absolute_diff_matrix_kelvins, do_max_pooling=True,
        window_size_px=window_size_px, stride_length_px=1, pad_edges=True
    )[0, ..., 0]

    flag_matrix = absolute_diff_matrix_kelvins >= min_temperature_diff_kelvins
    region_id_matrix = label_image(
        flag_matrix.astype(int), structure=numpy.full((3, 3), 1.)
    )[0]
    num_regions = numpy.max(region_id_matrix)

    for i in range(num_regions):
        these_indices = numpy.where(region_id_matrix == i + 1)
        if len(these_indices[0]) >= min_region_size_px:
            continue

        region_id_matrix[these_indices] = 0

    flag_matrix = region_id_matrix > 0
    print('{0:d} of {1:d} pixels have bad data.'.format(
        numpy.sum(flag_matrix), flag_matrix.size
    ))

    if not numpy.any(flag_matrix):
        return orig_temp_matrix_kelvins

    brightness_temp_matrix_kelvins = orig_temp_matrix_kelvins + 0.
    brightness_temp_matrix_kelvins[flag_matrix == True] = numpy.nan
    print(numpy.sum(numpy.isnan(brightness_temp_matrix_kelvins)))

    brightness_temp_matrix_kelvins = general_utils.fill_nans_by_interp(
        brightness_temp_matrix_kelvins
    )
    print(numpy.sum(numpy.isnan(brightness_temp_matrix_kelvins)))

    if not numpy.any(numpy.isnan(brightness_temp_matrix_kelvins)):
        return brightness_temp_matrix_kelvins

    return general_utils.fill_nans(brightness_temp_matrix_kelvins)


def _qc_data_one_day(
        input_file_name, half_window_size_px, min_temperature_diff_kelvins,
        min_region_size_px, output_file_name):
    """Quality-controls satellite data for one day.

    :param input_file_name: Path to input file (will be read by
        `satellite_io.read_file`).
    :param half_window_size_px: See documentation at top of file.
    :param min_temperature_diff_kelvins: Same.
    :param min_region_size_px: Same.
    :param output_file_name: Path to output file (will be written by
        `satellite_io.write_file`).
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    satellite_dict = satellite_io.read_file(
        netcdf_file_name=input_file_name, fill_nans=False
    )

    band_index = numpy.where(
        satellite_dict[satellite_io.BAND_NUMBERS_KEY] == BAND_NUMBER
    )[0][0]

    valid_times_unix_sec = satellite_dict[satellite_io.VALID_TIMES_KEY]
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_MESSAGES)
        for t in valid_times_unix_sec
    ]
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        print('Quality-controlling data at {0:s}...'.format(
            valid_time_strings[i]
        ))

        this_matrix = (
            satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY][i, ..., band_index]
        )
        this_matrix = _qc_data_one_time(
            brightness_temp_matrix_kelvins=this_matrix,
            half_window_size_px=half_window_size_px,
            min_temperature_diff_kelvins=min_temperature_diff_kelvins,
            min_region_size_px=min_region_size_px
        )
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY][i, ..., band_index] = (
            this_matrix
        )

        print('Writing quality-controlled data to: "{0:s}"...'.format(
            output_file_name
        ))
        satellite_io.write_file(
            netcdf_file_name=output_file_name,
            brightness_temp_matrix_kelvins=
            satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY][i, ...],
            latitudes_deg_n=satellite_dict[satellite_io.LATITUDES_KEY],
            longitudes_deg_e=satellite_dict[satellite_io.LONGITUDES_KEY],
            band_numbers=satellite_dict[satellite_io.BAND_NUMBERS_KEY],
            valid_time_unix_sec=valid_times_unix_sec[i], append=i > 0
        )

        if i != num_times - 1:
            print('\n')


def _run(top_input_dir_name, first_date_string, last_date_string,
         half_window_size_px, min_temperature_diff_kelvins, min_region_size_px,
         top_output_dir_name):
    """Quality-controls satellite data (gets rid of streaks in band 8).

    :param top_input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param half_window_size_px: Same.
    :param min_temperature_diff_kelvins: Same.
    :param min_region_size_px: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_geq(half_window_size_px, 1)
    error_checking.assert_is_greater(min_temperature_diff_kelvins, 0.)
    error_checking.assert_is_geq(min_region_size_px, 2)

    input_file_names = satellite_io.find_many_files(
        top_directory_name=top_input_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    date_strings = [satellite_io.file_name_to_date(f) for f in input_file_names]
    num_days = len(date_strings)

    output_file_names = [
        satellite_io.find_file(
            top_directory_name=top_output_dir_name, valid_date_string=d,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        ) for d in date_strings
    ]

    for i in range(num_days):
        _qc_data_one_day(
            input_file_name=input_file_names[i],
            half_window_size_px=half_window_size_px,
            min_temperature_diff_kelvins=min_temperature_diff_kelvins,
            min_region_size_px=min_region_size_px,
            output_file_name=output_file_names[i]
        )

        satellite_io.compress_file(output_file_names[i])
        os.remove(output_file_names[i])

        if i != num_days - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        half_window_size_px=getattr(
            INPUT_ARG_OBJECT, HALF_WINDOW_SIZE_ARG_NAME
        ),
        min_temperature_diff_kelvins=getattr(
            INPUT_ARG_OBJECT, MIN_DIFFERENCE_ARG_NAME
        ),
        min_region_size_px=getattr(INPUT_ARG_OBJECT, MIN_REGION_SIZE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
