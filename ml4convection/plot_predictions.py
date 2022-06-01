"""Plot predictions (and targets) for the given days."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import time_conversion
import file_system_utils
import error_checking
import gg_plotting_utils
import border_io
import prediction_io
import radar_utils
import neural_net
import plotting_utils
import prediction_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)
DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

MASK_OUTLINE_COLOUR = numpy.full(3, 152. / 255)
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
DAILY_TIMES_ARG_NAME = 'daily_times_seconds'
PLOT_DETERMINISTIC_ARG_NAME = 'plot_deterministic'
PLOT_UNCERTAINTY_ARG_NAME = 'plot_uncertainty'
PROB_THRESHOLD_ARG_NAME = 'probability_threshold'
MAX_PROB_ARG_NAME = 'max_prob_in_colour_bar'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will plot predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot predictions on partial (full) grids.'
)
SMOOTHING_RADIUS_HELP_STRING = (
    '[used only if {0:s} == 0] Radius for Gaussian smoother.  If you do not '
    'want to smooth predictions, leave this alone.'
).format(USE_PARTIAL_GRIDS_ARG_NAME)

DAILY_TIMES_HELP_STRING = (
    'List of times to plot for each day.  All values should be in the range '
    '0...86399.'
)
PLOT_DETERMINISTIC_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot deterministic (probabilistic) '
    'predictions.'
)
PLOT_UNCERTAINTY_HELP_STRING = (
    'Boolean flag.  If 1, will plot uncertainty estimates.  For each time '
    'step, central predictions will be one plot and uncertainty will be '
    'another plot.'
)
PROB_THRESHOLD_HELP_STRING = (
    '[used only if `{0:s} == 1`] Threshold used to convert probabilistic '
    'forecasts to deterministic.  All probabilities >= `{1:s}` will be '
    'considered "yes" forecasts, and all probabilities < `{1:s}` will be '
    'considered "no" forecasts.'
).format(PLOT_DETERMINISTIC_ARG_NAME, PROB_THRESHOLD_ARG_NAME)

MAX_PROB_HELP_STRING = (
    '[used only if `{0:s} == 0`] Max probability in colour bar.'
).format(PLOT_DETERMINISTIC_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
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
    '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=False, default=0,
    help=USE_PARTIAL_GRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False, default=-1,
    help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DAILY_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=DAILY_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_DETERMINISTIC_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_DETERMINISTIC_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_UNCERTAINTY_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_UNCERTAINTY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=PROB_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PROB_ARG_NAME, type=float, required=False, default=1.,
    help=MAX_PROB_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_predictions_one_example(
        prediction_dict, example_index, border_latitudes_deg_n,
        border_longitudes_deg_e, mask_matrix, plot_deterministic,
        probability_threshold, colour_map_object, colour_norm_object,
        output_dir_name, plot_uncertainty=False, plot_colour_bar=True,
        title_string=None, font_size=30, cbar_extend_min=True,
        latlng_visible=True):
    """Plots predictions (and targets) for one example (time step).

    M = number of rows in grid
    N = number of columns in grid

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param example_index: Will plot [i]th example, where i = `example_index`.
    :param border_latitudes_deg_n: See doc for `_plot_predictions_one_example`.
    :param border_longitudes_deg_e: Same.
    :param mask_matrix: M-by-N numpy array of integers (0 or 1), where 1 means
        the grid point is unmasked.
    :param plot_deterministic: See documentation at top of file.
    :param probability_threshold: Same.
    :param colour_map_object: See doc for
        `prediction_plotting.plot_probabilistic`.
    :param colour_norm_object: Same.
    :param output_dir_name: See documentation at top of file.
    :param plot_uncertainty: Same.
    :param plot_colour_bar: Boolean flag.
    :param title_string: Figure title.
    :param font_size: Font size.
    :param cbar_extend_min: Boolean flag.
    :param latlng_visible: Boolean flag.
    :return: output_file_name: Path to output file.
    :return: figure_object: Figure handle.
    :return: axes_object: Axes handle.
    """

    latitudes_deg_n = prediction_dict[prediction_io.LATITUDES_KEY]
    longitudes_deg_e = prediction_dict[prediction_io.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_width=4
    )

    pyplot.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=4, linestyles='solid',
        axes=axes_object
    )

    i = example_index
    valid_time_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY][i]
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )

    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...]

    if plot_uncertainty:
        matrix_to_plot = prediction_io.get_predictive_stdevs(prediction_dict)[
            i, ...
        ]

        # TODO(thunderhoser): HACK
        # TODO(thunderhoser): Also, this script should have an option of whether
        # or not to use the fancy quantile-based method to compute stdev.
        colour_map_object, colour_norm_object = (
            prediction_plotting.get_prob_colour_scheme(
                max_probability=0.2, make_lowest_prob_grey=False
            )
        )
    else:
        matrix_to_plot = prediction_io.get_mean_predictions(prediction_dict)[
            i, ...
        ]

    if plot_deterministic:
        prediction_matrix = (
            (matrix_to_plot >= probability_threshold).astype(int)
        )

        prediction_plotting.plot_deterministic(
            target_matrix=target_matrix, prediction_matrix=prediction_matrix,
            axes_object=axes_object,
            min_latitude_deg_n=latitudes_deg_n[0],
            min_longitude_deg_e=longitudes_deg_e[0],
            latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
            longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0]
        )

        if title_string is None:
            title_string = (
                'Actual (pink) and predicted (grey) convection at {0:s}'
            ).format(valid_time_string)
    else:
        prediction_plotting.plot_probabilistic(
            target_matrix=target_matrix, probability_matrix=matrix_to_plot,
            figure_object=figure_object, axes_object=axes_object,
            min_latitude_deg_n=latitudes_deg_n[0],
            min_longitude_deg_e=longitudes_deg_e[0],
            latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
            longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object
        )

        if title_string is None:
            if plot_uncertainty:
                title_string = (
                    'Uncertainty in convection probabilities at {0:s}'
                ).format(valid_time_string)
            else:
                title_string = (
                    'Forecast convection probabilities at {0:s}'
                ).format(valid_time_string)

        if plot_colour_bar:
            gg_plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical', extend_min=cbar_extend_min,
                extend_max=False, font_size=font_size
            )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
        meridian_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
        font_size=font_size
    )

    axes_object.set_title(title_string, fontsize=font_size)

    if not latlng_visible:
        axes_object.set_xticklabels(
            [''] * len(axes_object.get_xticks())
        )
        axes_object.set_yticklabels(
            [''] * len(axes_object.get_xticks())
        )

        for tick_object in axes_object.get_xticklines():
            tick_object.set_color(
                numpy.full(3, 1.)
            )

        for tick_object in axes_object.get_yticklines():
            tick_object.set_color(
                numpy.full(3, 1.)
            )

    if output_dir_name is None:
        return None, figure_object, axes_object

    output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
        output_dir_name,
        'uncertainty' if plot_uncertainty else 'predictions',
        valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name, None, None


def _plot_predictions_one_day(
        prediction_file_name, border_latitudes_deg_n, border_longitudes_deg_e,
        use_partial_grids, daily_times_seconds, plot_deterministic,
        plot_uncertainty, probability_threshold, colour_map_object,
        colour_norm_object, output_dir_name, smoothing_radius_px=None):
    """Plots predictions (and targets) for one day.

    P = number of points in border set

    :param prediction_file_name: Path to prediction file.  Will be read by
        `prediction_io.read_file`.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param use_partial_grids: See documentation at top of file.
    :param daily_times_seconds: Same.
    :param plot_deterministic: Same.
    :param plot_uncertainty: Same.
    :param probability_threshold: Same.
    :param colour_map_object: See doc for
        `prediction_plotting.plot_probabilistic`.
    :param colour_norm_object: Same.
    :param output_dir_name: See documentation at top of file.
    :param smoothing_radius_px: Same.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    if smoothing_radius_px is not None:
        prediction_dict = prediction_io.smooth_probabilities(
            prediction_dict=prediction_dict,
            smoothing_radius_px=smoothing_radius_px
        )

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    if use_partial_grids:
        mask_matrix = model_metadata_dict[neural_net.MASK_MATRIX_KEY]
    else:
        mask_matrix = model_metadata_dict[neural_net.FULL_MASK_MATRIX_KEY]

    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    num_times = target_matrix.shape[0]

    for i in range(num_times):
        target_matrix[i, ...][mask_matrix == False] = 0

    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = target_matrix

    # TODO(thunderhoser): Put this code somewhere reusable.
    valid_times_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY]
    base_time_unix_sec = number_rounding.floor_to_nearest(
        valid_times_unix_sec[0], DAYS_TO_SECONDS
    )
    desired_times_unix_sec = numpy.round(
        base_time_unix_sec + daily_times_seconds
    ).astype(int)

    good_flags = numpy.array([
        t in valid_times_unix_sec for t in desired_times_unix_sec
    ], dtype=bool)

    if not numpy.any(good_flags):
        return

    desired_times_unix_sec = desired_times_unix_sec[good_flags]
    prediction_dict = prediction_io.subset_by_time(
        prediction_dict=prediction_dict,
        desired_times_unix_sec=desired_times_unix_sec
    )[0]

    num_examples = len(prediction_dict[prediction_io.VALID_TIMES_KEY])

    for i in range(num_examples):
        _plot_predictions_one_example(
            prediction_dict=prediction_dict, example_index=i,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix.astype(int),
            plot_deterministic=plot_deterministic, plot_uncertainty=False,
            probability_threshold=probability_threshold,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            output_dir_name=output_dir_name
        )

        if not plot_uncertainty:
            continue

        _plot_predictions_one_example(
            prediction_dict=prediction_dict, example_index=i,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix.astype(int),
            plot_deterministic=plot_deterministic, plot_uncertainty=True,
            probability_threshold=probability_threshold,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            output_dir_name=output_dir_name
        )


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         use_partial_grids, smoothing_radius_px, daily_times_seconds,
         plot_deterministic, plot_uncertainty, probability_threshold,
         max_prob_in_colour_bar, output_dir_name):
    """Plot predictions (and targets) for the given days.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param use_partial_grids: Same.
    :param smoothing_radius_px: Same.
    :param daily_times_seconds: Same.
    :param plot_deterministic: Same.
    :param plot_uncertainty: Same.
    :param probability_threshold: Same.
    :param max_prob_in_colour_bar: Same.
    :param output_dir_name: Same.
    """

    if smoothing_radius_px <= 0:
        smoothing_radius_px = None

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_geq_numpy_array(daily_times_seconds, 0)
    error_checking.assert_is_less_than_numpy_array(
        daily_times_seconds, DAYS_TO_SECONDS
    )

    if plot_deterministic:
        error_checking.assert_is_greater(probability_threshold, 0.)
        error_checking.assert_is_less_than(probability_threshold, 1.)
    else:
        probability_threshold = None

    colour_map_object, colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=max_prob_in_colour_bar, make_lowest_prob_grey=False
        )
    )

    if not use_partial_grids:
        prediction_file_names = prediction_io.find_many_files(
            top_directory_name=top_prediction_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            radar_number=None, prefer_zipped=True, allow_other_format=True,
            raise_error_if_any_missing=False
        )

        for i in range(len(prediction_file_names)):
            _plot_predictions_one_day(
                prediction_file_name=prediction_file_names[i],
                daily_times_seconds=daily_times_seconds,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                use_partial_grids=use_partial_grids,
                smoothing_radius_px=smoothing_radius_px,
                plot_deterministic=plot_deterministic,
                plot_uncertainty=plot_uncertainty,
                probability_threshold=probability_threshold,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                output_dir_name=output_dir_name
            )

            if i != len(prediction_file_names) - 1:
                print(SEPARATOR_STRING)

        return

    date_strings = []

    for k in range(NUM_RADARS):
        if len(date_strings) == 0:
            prediction_file_names = prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                radar_number=k, prefer_zipped=True, allow_other_format=True,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=k > 0
            )

            date_strings = [
                prediction_io.file_name_to_date(f)
                for f in prediction_file_names
            ]
        else:
            prediction_file_names = [
                prediction_io.find_file(
                    top_directory_name=top_prediction_dir_name,
                    valid_date_string=d, radar_number=k,
                    prefer_zipped=True, allow_other_format=True,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

        this_output_dir_name = '{0:s}/radar{1:d}'.format(output_dir_name, k)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name
        )

        num_days = len(date_strings)

        for i in range(num_days):
            _plot_predictions_one_day(
                prediction_file_name=prediction_file_names[i],
                daily_times_seconds=daily_times_seconds,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                use_partial_grids=use_partial_grids,
                smoothing_radius_px=smoothing_radius_px,
                plot_deterministic=plot_deterministic,
                plot_uncertainty=plot_uncertainty,
                probability_threshold=probability_threshold,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                output_dir_name=this_output_dir_name
            )

            if not (i == num_days - 1 and k == NUM_RADARS - 1):
                print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        use_partial_grids=bool(
            getattr(INPUT_ARG_OBJECT, USE_PARTIAL_GRIDS_ARG_NAME)
        ),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        daily_times_seconds=numpy.array(
            getattr(INPUT_ARG_OBJECT, DAILY_TIMES_ARG_NAME), dtype=int
        ),
        plot_deterministic=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_DETERMINISTIC_ARG_NAME
        )),
        plot_uncertainty=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_UNCERTAINTY_ARG_NAME
        )),
        probability_threshold=getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME
        ),
        max_prob_in_colour_bar=getattr(INPUT_ARG_OBJECT, MAX_PROB_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
