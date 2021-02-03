"""For each time step, concatenates figures with the following data:

- Predictions (forecast convection probabilities) and targets
- Radar data
- Satellite data
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.machine_learning import neural_net

DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

CONCAT_FIGURE_SIZE_PX = int(1e7)

PREDICTION_FIGURE_DIR_ARG_NAME = 'prediction_figure_dir_name'
REFL_FIGURE_DIR_ARG_NAME = 'refl_figure_dir_name'
SATELLITE_FIGURE_DIR_ARG_NAME = 'satellite_figure_dir_name'
MODEL_METAFILE_ARG_NAME = 'model_metafile_name'
PERSISTENCE_ARG_NAME = 'is_persistence_model'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
DAILY_VALID_TIMES_ARG_NAME = 'daily_valid_times_seconds'
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
MODEL_METAFILE_HELP_STRING = (
    'Path to file with metadata for model that generated predictions.  Will be '
    'read by `neural_net.read_metafile`.'
)
PERSISTENCE_HELP_STRING = (
    'Boolean flag.  If 1 (0), is persistence (actual) model.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will concatenate figures for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

DAILY_VALID_TIMES_HELP_STRING = (
    'List of valid times (target times) to use for each day.  All values should'
    ' be in the range 0...86399.'
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
    '--' + MODEL_METAFILE_ARG_NAME, type=str, required=True,
    help=MODEL_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PERSISTENCE_ARG_NAME, type=int, required=False, default=0,
    help=PERSISTENCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DAILY_VALID_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=DAILY_VALID_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _find_input_files(
        prediction_figure_dir_name, refl_figure_dir_name,
        satellite_figure_dir_name, model_metafile_name, is_persistence_model,
        first_date_string, last_date_string, daily_valid_times_seconds):
    """Finds input files (figures to concatenate).

    V = number of valid times
    B = number of satellite bands
    L = number of lag times

    :param prediction_figure_dir_name: See documentation at top of file.
    :param refl_figure_dir_name: Same.
    :param satellite_figure_dir_name: Same.
    :param model_metafile_name: Same.
    :param is_persistence_model: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_valid_times_seconds: Same.
    :return: prediction_figure_file_names: length-V list of paths to prediction
        figures.
    :return: refl_figure_file_names: length-V list of paths to reflectivity
        figures.
    :return: satellite_fig_file_name_matrix: V-by-B-by-L numpy array of paths to
        satellite figures.
    :return: valid_time_strings: length-V numpy array of valid times (format
        "yyyy-mm-dd-HHMM").
    :raises: ValueError: if any expected reflectivity or satellite figure is not
        found.
    """

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )
    dates_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(d, time_conversion.SPC_DATE_FORMAT)
        for d in date_strings
    ], dtype=int)

    possible_valid_times_unix_sec = numpy.concatenate([
        d + daily_valid_times_seconds for d in dates_unix_sec
    ])

    possible_valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in possible_valid_times_unix_sec
    ]
    possible_pred_fig_file_names = [
        '{0:s}/predictions_{1:s}.jpg'.format(prediction_figure_dir_name, t)
        for t in possible_valid_time_strings
    ]

    valid_times_unix_sec = []
    prediction_figure_file_names = []

    for i in range(len(possible_valid_time_strings)):
        if not os.path.isfile(possible_pred_fig_file_names[i]):
            continue

        valid_times_unix_sec.append(possible_valid_times_unix_sec[i])
        prediction_figure_file_names.append(possible_pred_fig_file_names[i])

    valid_times_unix_sec = numpy.array(valid_times_unix_sec, dtype=int)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]

    refl_figure_file_names = [
        '{0:s}/{1:s}/{2:s}/reflectivity_{3:s}_composite.jpg'.format(
            refl_figure_dir_name, t[:4], t[:10].replace('-', ''), t
        )
        for t in valid_time_strings
    ]

    if not is_persistence_model:
        bad_file_names = [
            f for f in refl_figure_file_names if not os.path.isfile(f)
        ]

        if len(bad_file_names) > 0:
            error_string = (
                '\nFiles were expected, but not found, at the following '
                'locations:\n{0:s}'
            ).format(
                str(bad_file_names)
            )

            raise ValueError(error_string)

    good_flags = numpy.array([
        os.path.isfile(f) for f in refl_figure_file_names
    ], dtype=bool)

    good_indices = numpy.where(good_flags)[0]

    valid_times_unix_sec = valid_times_unix_sec[good_indices]
    valid_time_strings = [valid_time_strings[k] for k in good_indices]
    prediction_figure_file_names = [
        prediction_figure_file_names[k] for k in good_indices
    ]
    refl_figure_file_names = [refl_figure_file_names[k] for k in good_indices]

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    band_numbers = training_option_dict[neural_net.BAND_NUMBERS_KEY]
    lead_time_seconds = training_option_dict[neural_net.LEAD_TIME_KEY]
    lag_times_seconds = training_option_dict[neural_net.LAG_TIMES_KEY]

    num_valid_times = len(valid_time_strings)
    num_bands = len(band_numbers)
    num_lag_times = len(lag_times_seconds)

    satellite_fig_file_name_matrix = numpy.full(
        (num_valid_times, num_bands, num_lag_times), '', dtype=object
    )

    for i in range(num_valid_times):
        for j in range(num_bands):
            for k in range(num_lag_times):
                this_time_string = time_conversion.unix_sec_to_string(
                    valid_times_unix_sec[i] - lag_times_seconds[k] -
                    lead_time_seconds,
                    TIME_FORMAT
                )

                satellite_fig_file_name_matrix[i, j, k] = (
                    '{0:s}/{1:s}/brightness-temperature_{2:s}_band{3:02d}.jpg'
                ).format(
                    satellite_figure_dir_name,
                    this_time_string[:10].replace('-', ''),
                    this_time_string, band_numbers[j]
                )

    if not is_persistence_model:
        bad_file_names = [
            f for f in numpy.ravel(satellite_fig_file_name_matrix)
            if not os.path.isfile(f)
        ]

        if len(bad_file_names) > 0:
            error_string = (
                '\nFiles were expected, but not found, at the following '
                'locations:\n{0:s}'
            ).format(
                str(bad_file_names)
            )

            raise ValueError(error_string)

    good_flags = numpy.array([
        os.path.isfile(f) for f in numpy.ravel(satellite_fig_file_name_matrix)
    ], dtype=bool)

    good_flag_matrix = numpy.reshape(
        good_flags, satellite_fig_file_name_matrix.shape
    )
    good_flags = numpy.all(good_flag_matrix, axis=(1, 2))
    good_indices = numpy.where(good_flags)[0]

    # valid_times_unix_sec = valid_times_unix_sec[good_indices]
    valid_time_strings = [valid_time_strings[k] for k in good_indices]
    prediction_figure_file_names = [
        prediction_figure_file_names[k] for k in good_indices
    ]
    refl_figure_file_names = [refl_figure_file_names[k] for k in good_indices]
    satellite_fig_file_name_matrix = (
        satellite_fig_file_name_matrix[good_indices, ...]
    )

    return (
        prediction_figure_file_names,
        refl_figure_file_names,
        satellite_fig_file_name_matrix,
        valid_time_strings
    )


def _run(prediction_figure_dir_name, refl_figure_dir_name,
         satellite_figure_dir_name, model_metafile_name, is_persistence_model,
         first_date_string, last_date_string, daily_valid_times_seconds,
         output_dir_name):
    """For each time step, concatenates figures with the following data:

    - Predictions (forecast convection probabilities) and targets
    - Radar data
    - Satellite data

    This is effectively the main method.

    :param prediction_figure_dir_name: See documentation at top of file.
    :param refl_figure_dir_name: Same.
    :param satellite_figure_dir_name: Same.
    :param model_metafile_name: Same.
    :param is_persistence_model: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_valid_times_seconds: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq_numpy_array(daily_valid_times_seconds, 0)
    error_checking.assert_is_less_than_numpy_array(
        daily_valid_times_seconds, DAYS_TO_SECONDS
    )

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
        model_metafile_name=model_metafile_name,
        is_persistence_model=is_persistence_model,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        daily_valid_times_seconds=daily_valid_times_seconds
    )

    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    lag_times_seconds = training_option_dict[neural_net.LAG_TIMES_KEY]

    num_valid_times = satellite_fig_file_name_matrix.shape[0]
    num_bands = satellite_fig_file_name_matrix.shape[1]
    num_lag_times = len(lag_times_seconds)

    num_panels = num_bands + 2
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    for i in range(num_valid_times):
        for k in range(num_lag_times):
            these_input_file_names = (
                satellite_fig_file_name_matrix[i, :, k].tolist() +
                [refl_figure_file_names[i], prediction_figure_file_names[i]]
            )
            this_output_file_name = (
                '{0:s}/all_data_valid-{1:s}_lag-time-seconds={2:05d}.jpg'
            ).format(
                output_dir_name, valid_time_strings[i], lag_times_seconds[k]
            )

            print('Concatenating figures to: "{0:s}"...'.format(
                this_output_file_name
            ))
            imagemagick_utils.concatenate_images(
                input_file_names=these_input_file_names,
                output_file_name=this_output_file_name,
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns
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
        model_metafile_name=getattr(INPUT_ARG_OBJECT, MODEL_METAFILE_ARG_NAME),
        is_persistence_model=bool(getattr(
            INPUT_ARG_OBJECT, PERSISTENCE_ARG_NAME
        )),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        daily_valid_times_seconds=numpy.array(
            getattr(INPUT_ARG_OBJECT, DAILY_VALID_TIMES_ARG_NAME), dtype=int
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
