"""Plot predictions (and targets) for the given days."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.plotting import prediction_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

LATLNG_FONT_SIZE = 16
LATLNG_COLOUR = numpy.full(3, 152. / 255)
BORDER_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
DAILY_TIMES_ARG_NAME = 'daily_times_seconds'
PLOT_DETERMINISTIC_ARG_NAME = 'plot_deterministic'
PROB_THRESHOLD_ARG_NAME = 'probability_threshold'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will plot predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

DAILY_TIMES_HELP_STRING = (
    'List of times to plot for each day.  All values should be in the range '
    '0...86399.'
)
PLOT_DETERMINISTIC_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot deterministic (probabilistic) '
    'predictions.'
)
PROB_THRESHOLD_HELP_STRING = (
    '[used only if `{0:s} == 1`] Threshold used to convert probabilistic '
    'forecasts to deterministic.  All probabilities >= `{1:s}` will be '
    'considered "yes" forecasts, and all probabilities < `{1:s}` will be '
    'considered "no" forecasts.'
).format(PLOT_DETERMINISTIC_ARG_NAME, PROB_THRESHOLD_ARG_NAME)

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
    '--' + DAILY_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=DAILY_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_DETERMINISTIC_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_DETERMINISTIC_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=PROB_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def add_latlng_ticks(latitudes_deg_n, longitudes_deg_e, axes_object,
                     font_size):
    """Adds tick marks for latitude and longitude to existing plot.

    :param latitudes_deg_n: 1-D numpy array of latitudes in plot (deg N).
    :param longitudes_deg_e: 1-D numpy array of longitudes in plot (deg E).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param font_size: Font size.
    """

    # TODO(thunderhoser): Put this method elsewhere.

    tick_latitudes_deg_n = numpy.unique(
        number_rounding.round_to_nearest(latitudes_deg_n, 2.)
    )
    tick_latitudes_deg_n = tick_latitudes_deg_n[
        tick_latitudes_deg_n >= numpy.min(latitudes_deg_n)
    ]
    tick_latitudes_deg_n = tick_latitudes_deg_n[
        tick_latitudes_deg_n <= numpy.max(latitudes_deg_n)
    ]

    tick_longitudes_deg_e = numpy.unique(
        number_rounding.round_to_nearest(longitudes_deg_e, 2.)
    )
    tick_longitudes_deg_e = tick_longitudes_deg_e[
        tick_longitudes_deg_e >= numpy.min(longitudes_deg_e)
    ]
    tick_longitudes_deg_e = tick_longitudes_deg_e[
        tick_longitudes_deg_e <= numpy.max(longitudes_deg_e)
    ]

    axes_object.set_xticks(tick_longitudes_deg_e)
    axes_object.set_yticks(tick_latitudes_deg_n)
    axes_object.grid(
        b=True, which='major', axis='both', linestyle='--', linewidth=2,
        color=LATLNG_COLOUR
    )

    axes_object.set_xlabel(r'Longitude ($^{\circ}$E)', fontsize=font_size)
    axes_object.set_ylabel(r'Latitude ($^{\circ}$N)', fontsize=font_size)


def _plot_predictions_one_example(
        prediction_dict, example_index, border_latitudes_deg_n,
        border_longitudes_deg_e, plot_deterministic, probability_threshold,
        output_dir_name):
    """Plots predictions (and targets) for one example (time step).

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param example_index: Will plot [i]th example, where i = `example_index`.
    :param border_latitudes_deg_n: See doc for `_plot_predictions_one_example`.
    :param border_longitudes_deg_e: Same.
    :param plot_deterministic: See documentation at top of file.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

    latitudes_deg_n = prediction_dict[prediction_io.LATITUDES_KEY]
    longitudes_deg_e = prediction_dict[prediction_io.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.plot(
        border_longitudes_deg_e, border_latitudes_deg_n, color=BORDER_COLOUR,
        linestyle='solid', linewidth=2, zorder=-1e8
    )

    i = example_index
    valid_time_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY][i]
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )

    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...]
    probability_matrix = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...]
    )

    if plot_deterministic:
        prediction_matrix = (
            (probability_matrix >= probability_threshold).astype(int)
        )

        prediction_plotting.plot_deterministic(
            target_matrix=target_matrix, prediction_matrix=prediction_matrix,
            axes_object=axes_object,
            min_latitude_deg_n=latitudes_deg_n[0],
            min_longitude_deg_e=longitudes_deg_e[0],
            latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
            longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0]
        )

        title_string = (
            'Actual (pink) and predicted (grey) convection at {0:s}'
        ).format(valid_time_string)
    else:
        prediction_plotting.plot_probabilistic(
            target_matrix=target_matrix, probability_matrix=probability_matrix,
            axes_object=axes_object,
            min_latitude_deg_n=latitudes_deg_n[0],
            min_longitude_deg_e=longitudes_deg_e[0],
            latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
            longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0]
        )

        title_string = 'Forecast convection probabilities at {0:s}'.format(
            valid_time_string
        )

        colour_map_object, colour_norm_object = (
            prediction_plotting.get_prob_colour_scheme()
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=probability_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=False
        )

    add_latlng_ticks(
        latitudes_deg_n=latitudes_deg_n, longitudes_deg_e=longitudes_deg_e,
        axes_object=axes_object, font_size=LATLNG_FONT_SIZE
    )

    axes_object.set_title(title_string)

    output_file_name = '{0:s}/predictions_{1:s}.jpg'.format(
        output_dir_name, valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_predictions_one_day(
        prediction_file_name, border_latitudes_deg_n, border_longitudes_deg_e,
        daily_times_seconds, plot_deterministic, probability_threshold,
        output_dir_name):
    """Plots predictions (and targets) for one day.

    P = number of points in borders

    :param prediction_file_name: Path to prediction file.  Will be read by
        `prediction_io.read_file`.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param daily_times_seconds: See documentation at top of file.
    :param plot_deterministic: Same.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

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
            plot_deterministic=plot_deterministic,
            probability_threshold=probability_threshold,
            output_dir_name=output_dir_name
        )


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         daily_times_seconds, plot_deterministic, probability_threshold,
         output_dir_name):
    """Plot predictions (and targets) for the given days.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_times_seconds: Same.
    :param plot_deterministic: Same.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

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

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        raise_error_if_any_missing=False
    )

    for i in range(len(prediction_file_names)):
        _plot_predictions_one_day(
            prediction_file_name=prediction_file_names[i],
            daily_times_seconds=daily_times_seconds,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            plot_deterministic=plot_deterministic,
            probability_threshold=probability_threshold,
            output_dir_name=output_dir_name
        )

        if i != len(prediction_file_names) - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        daily_times_seconds=numpy.array(
            getattr(INPUT_ARG_OBJECT, DAILY_TIMES_ARG_NAME), dtype=int
        ),
        plot_deterministic=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_DETERMINISTIC_ARG_NAME
        )),
        probability_threshold=getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
