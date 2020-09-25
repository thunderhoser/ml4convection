"""For each time step, concatenates figures with the following data:

- Predictions (forecast convection probabilities) and targets
- Radar data
- Satellite data
"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import imagemagick_utils

DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

CONCAT_FIGURE_SIZE_PX = int(1e7)

PREDICTION_FIGURE_DIR_ARG_NAME = 'prediction_figure_dir_name'
REFL_FIGURE_DIR_ARG_NAME = 'refl_figure_dir_name'
SATELLITE_FIGURE_DIR_ARG_NAME = 'satellite_figure_dir_name'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
DAILY_TIMES_ARG_NAME = 'daily_times_seconds'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FIGURE_DIR_HELP_STRING = (
    'Name of directory with figures showing predictions and targets, created by'
    ' plot_predictions.py.'
)
REFL_FIGURE_DIR_HELP_STRING = (
    'Name of directory with figures showing composite reflectivity, created by '
    'plot_radar.py.'
)
SATELLITE_FIGURE_DIR_HELP_STRING = (
    'Name of top-level directory (one subdirectory per spectral band) showing '
    'brightness temperatures, created by plot_satellite.py.'
)
BAND_NUMBERS_HELP_STRING = (
    'List of band numbers for satellite figures.  Will use figures only for '
    'these spectral bands.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will concatenate figures for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

DAILY_TIMES_HELP_STRING = (
    'List of times to use for each day.  All values should be in the range '
    '0...86399.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Concatenated (paneled) figures will be saved '
    'here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FIGURE_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_FIGURE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REFL_FIGURE_DIR_ARG_NAME, type=str, required=True,
    help=REFL_FIGURE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_FIGURE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_FIGURE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BAND_NUMBERS_ARG_NAME, type=int, nargs='+', required=True,
    help=BAND_NUMBERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DAILY_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=DAILY_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _find_input_files(
        prediction_figure_dir_name, refl_figure_dir_name,
        satellite_figure_dir_name, band_numbers, first_date_string,
        last_date_string, daily_times_seconds):
    """Finds input files (figures to concatenate).

    T = number of time steps
    B = number of satellite bands

    :param prediction_figure_dir_name: See documentation at top of file.
    :param refl_figure_dir_name: Same.
    :param satellite_figure_dir_name: Same.
    :param band_numbers: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_times_seconds: Same.
    :return: prediction_figure_file_names: length-T list of paths to prediction
        figures.
    :return: refl_figure_file_names: length-T list of paths to reflectivity
        figures.
    :return: satellite_fig_file_name_matrix: T-by-B numpy array of paths to
        satellite figures.
    :return: valid_time_strings: length-T numpy array of valid times (format
        "yyyy-mm-dd-HHMM").
    :raises: ValueError: if any expected reflectivity or satellite figure is not
        found.
    """

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )
    dates_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(d, time_conversion.SPC_DATE_FORMAT)
        for d in date_strings
    ], dtype=int)

    valid_times_unix_sec = numpy.concatenate([
        d + daily_times_seconds for d in dates_unix_sec
    ])

    possible_valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]
    possible_pred_fig_file_names = [
        '{0:s}/predictions_{1:s}.jpg'.format(prediction_figure_dir_name, t)
        for t in possible_valid_time_strings
    ]

    valid_time_strings = []
    prediction_figure_file_names = []

    for i in range(len(possible_valid_time_strings)):
        if not os.path.isfile(possible_pred_fig_file_names[i]):
            continue

        valid_time_strings.append(possible_valid_time_strings[i])
        prediction_figure_file_names.append(possible_pred_fig_file_names[i])

    refl_figure_file_names = [
        '{0:s}/composite_reflectivity_{1:s}.jpg'.format(refl_figure_dir_name, t)
        for t in valid_time_strings
    ]
    bad_file_names = [
        f for f in refl_figure_file_names if not os.path.isfile(f)
    ]

    if len(bad_file_names) > 0:
        error_string = (
            '\nFiles were expected, but not found, at the following locations:'
            '\n{0:s}'
        ).format(
            str(bad_file_names)
        )

        raise ValueError(error_string)

    num_times = len(valid_time_strings)
    num_bands = len(band_numbers)
    satellite_fig_file_name_matrix = numpy.full(
        (num_times, num_bands), '', dtype=object
    )

    for i in range(num_times):
        for j in range(num_bands):
            satellite_fig_file_name_matrix[i, j] = (
                '{0:s}/band{1:02d}/brightness-temperature_band{1:02d}_{2:s}.jpg'
            ).format(
                satellite_figure_dir_name,
                band_numbers[j], valid_time_strings[i]
            )

    bad_file_names = [
        f for f in numpy.ravel(satellite_fig_file_name_matrix)
        if not os.path.isfile(f)
    ]

    if len(bad_file_names) > 0:
        error_string = (
            '\nFiles were expected, but not found, at the following locations:'
            '\n{0:s}'
        ).format(
            str(bad_file_names)
        )

        raise ValueError(error_string)

    return (
        prediction_figure_file_names,
        refl_figure_file_names,
        satellite_fig_file_name_matrix,
        valid_time_strings
    )


def _run(prediction_figure_dir_name, refl_figure_dir_name,
         satellite_figure_dir_name, band_numbers, first_date_string,
         last_date_string, daily_times_seconds, output_dir_name):
    """For each time step, concatenates figures with the following data:

    - Predictions (forecast convection probabilities) and targets
    - Radar data
    - Satellite data

    This is effectively the main method.

    :param prediction_figure_dir_name: See documentation at top of file.
    :param refl_figure_dir_name: Same.
    :param satellite_figure_dir_name: Same.
    :param band_numbers: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_times_seconds: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq_numpy_array(daily_times_seconds, 0)
    error_checking.assert_is_less_than_numpy_array(
        daily_times_seconds, DAYS_TO_SECONDS
    )
    error_checking.assert_is_greater_numpy_array(band_numbers, 0)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    (
        prediction_figure_file_names,
        refl_figure_file_names,
        satellite_fig_file_name_matrix,
        valid_time_strings
    ) = _find_input_files(
        prediction_figure_dir_name=prediction_figure_dir_name,
        refl_figure_dir_name=refl_figure_dir_name,
        satellite_figure_dir_name=satellite_figure_dir_name,
        band_numbers=band_numbers,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        daily_times_seconds=daily_times_seconds
    )

    num_times = satellite_fig_file_name_matrix.shape[0]
    num_bands = satellite_fig_file_name_matrix.shape[1]

    num_panels = num_bands + 2
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    for i in range(num_times):
        these_input_file_names = (
            satellite_fig_file_name_matrix[i, :].tolist() +
            [refl_figure_file_names[i], prediction_figure_file_names[i]]
        )
        this_output_file_name = '{0:s}/all_data_{1:s}.jpg'.format(
            output_dir_name, valid_time_strings[i]
        )

        print('Concatenating figures to: "{0:s}"...'.format(
            this_output_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=these_input_file_names,
            output_file_name=this_output_file_name,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
        )

        imagemagick_utils.resize_image(
            input_file_name=this_output_file_name,
            output_file_name=this_output_file_name,
            output_size_pixels=CONCAT_FIGURE_SIZE_PX
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_figure_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FIGURE_DIR_ARG_NAME
        ),
        refl_figure_dir_name=getattr(
            INPUT_ARG_OBJECT, REFL_FIGURE_DIR_ARG_NAME
        ),
        satellite_figure_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_FIGURE_DIR_ARG_NAME
        ),
        band_numbers=numpy.array(
            getattr(INPUT_ARG_OBJECT, BAND_NUMBERS_ARG_NAME), dtype=int
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        daily_times_seconds=numpy.array(
            getattr(INPUT_ARG_OBJECT, DAILY_TIMES_ARG_NAME), dtype=int
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
