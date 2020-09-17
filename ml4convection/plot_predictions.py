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
import plotting_utils
import prediction_io
import prediction_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
DAILY_TIMES_ARG_NAME = 'daily_times_seconds'
PLOT_RANDOM_ARG_NAME = 'plot_random_examples'
NUM_EXAMPLES_PER_DAY_ARG_NAME = 'num_examples_per_day'
PLOT_BASEMAP_ARG_NAME = 'plot_basemap'
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
PLOT_RANDOM_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Boolean flag.  If 1, will randomly '
    'draw `{1:s}` examples from each day.  If 0, will draw the first `{1:s}` '
    'examples from each day.'
).format(DAILY_TIMES_ARG_NAME, NUM_EXAMPLES_PER_DAY_ARG_NAME)

NUM_EXAMPLES_PER_DAY_HELP_STRING = (
    'See documentation for `{0:s}`.'.format(PLOT_RANDOM_ARG_NAME)
)
PLOT_BASEMAP_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot radar images with (without) basemap.'
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
    '--' + DAILY_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=DAILY_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_RANDOM_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_RANDOM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_DAY_ARG_NAME, type=int, required=False, default=5,
    help=NUM_EXAMPLES_PER_DAY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BASEMAP_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_BASEMAP_HELP_STRING
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


def _plot_predictions_one_example(
        prediction_dict, example_index, plot_basemap, plot_deterministic,
        probability_threshold, output_dir_name):
    """Plots predictions (and targets) for one example (time step).

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param example_index: Will plot [i]th example, where i = `example_index`.
    :param plot_basemap: See documentation at top of file.
    :param plot_deterministic: Same.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

    latitudes_deg_n = prediction_dict[prediction_io.LATITUDES_KEY]
    longitudes_deg_e = prediction_dict[prediction_io.LONGITUDES_KEY]

    if plot_basemap:
        figure_object, axes_object, basemap_object = (
            plotting_utils.create_equidist_cylindrical_map(
                min_latitude_deg=numpy.min(latitudes_deg_n),
                max_latitude_deg=numpy.max(latitudes_deg_n),
                min_longitude_deg=numpy.min(longitudes_deg_e),
                max_longitude_deg=numpy.max(longitudes_deg_e),
                resolution_string='i'
            )
        )

        plotting_utils.plot_coastlines(
            basemap_object=basemap_object, axes_object=axes_object,
            line_colour=plotting_utils.DEFAULT_COUNTRY_COLOUR
        )
        plotting_utils.plot_countries(
            basemap_object=basemap_object, axes_object=axes_object
        )
        plotting_utils.plot_states_and_provinces(
            basemap_object=basemap_object, axes_object=axes_object
        )
        plotting_utils.plot_parallels(
            basemap_object=basemap_object, axes_object=axes_object,
            num_parallels=NUM_PARALLELS
        )
        plotting_utils.plot_meridians(
            basemap_object=basemap_object, axes_object=axes_object,
            num_meridians=NUM_MERIDIANS
        )
    else:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
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

    if not plot_basemap:
        tick_latitudes_deg_n = numpy.unique(numpy.round(latitudes_deg_n))
        tick_latitudes_deg_n = tick_latitudes_deg_n[
            tick_latitudes_deg_n >= numpy.min(latitudes_deg_n)
        ]
        tick_latitudes_deg_n = tick_latitudes_deg_n[
            tick_latitudes_deg_n <= numpy.max(latitudes_deg_n)
        ]

        tick_longitudes_deg_e = numpy.unique(numpy.round(longitudes_deg_e))
        tick_longitudes_deg_e = tick_longitudes_deg_e[
            tick_longitudes_deg_e >= numpy.min(longitudes_deg_e)
        ]
        tick_longitudes_deg_e = tick_longitudes_deg_e[
            tick_longitudes_deg_e <= numpy.max(longitudes_deg_e)
        ]

        axes_object.set_xticks(tick_longitudes_deg_e)
        axes_object.set_yticks(tick_latitudes_deg_n)
        axes_object.grid(
            b=True, which='major', axis='both', linestyle='--', linewidth=2
        )

        axes_object.set_xlabel(r'Longitude ($^{\circ}$E)')
        axes_object.set_ylabel(r'Latitude ($^{\circ}$N)')

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
        prediction_file_name, daily_times_seconds, plot_random_examples,
        num_examples, plot_basemap, plot_deterministic, probability_threshold,
        output_dir_name):
    """Plots predictions (and targets) for one day.

    :param prediction_file_name: Path to prediction file.  Will be read by
        `prediction_io.read_file`.
    :param daily_times_seconds: See documentation at top of file.
    :param plot_random_examples: Same.
    :param num_examples: Same.
    :param plot_basemap: Same.
    :param plot_deterministic: Same.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

    # TODO(thunderhoser): Put this code somewhere reusable.

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    if daily_times_seconds is not None:
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
    else:
        num_examples_total = len(prediction_dict[prediction_io.VALID_TIMES_KEY])
        desired_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int
        )

        if num_examples < num_examples_total:
            if plot_random_examples:
                desired_indices = numpy.random.choice(
                    desired_indices, size=num_examples, replace=False
                )
            else:
                desired_indices = desired_indices[:num_examples]

        prediction_dict = prediction_io.subset_by_index(
            prediction_dict=prediction_dict, desired_indices=desired_indices
        )

    num_examples = len(prediction_dict[prediction_io.VALID_TIMES_KEY])

    for i in range(num_examples):
        _plot_predictions_one_example(
            prediction_dict=prediction_dict, example_index=i,
            plot_basemap=plot_basemap,
            plot_deterministic=plot_deterministic,
            probability_threshold=probability_threshold,
            output_dir_name=output_dir_name
        )


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         daily_times_seconds, plot_random_examples, num_examples_per_day,
         plot_basemap, plot_deterministic, probability_threshold,
         output_dir_name):
    """Plot predictions (and targets) for the given days.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_times_seconds: Same.
    :param plot_random_examples: Same.
    :param num_examples_per_day: Same.
    :param plot_basemap: Same.
    :param plot_deterministic: Same.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if len(daily_times_seconds) == 1 and daily_times_seconds[0] < 0:
        daily_times_seconds = None

    if daily_times_seconds is not None:
        error_checking.assert_is_geq_numpy_array(daily_times_seconds, 0)
        error_checking.assert_is_less_than_numpy_array(
            daily_times_seconds, DAYS_TO_SECONDS
        )
        plot_random_examples = False

    if plot_deterministic:
        probability_threshold = None
    else:
        error_checking.assert_is_greater(probability_threshold, 0.)
        error_checking.assert_is_less_than(probability_threshold, 1.)

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
            plot_random_examples=plot_random_examples,
            num_examples=num_examples_per_day,
            plot_basemap=plot_basemap,
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
        plot_random_examples=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_RANDOM_ARG_NAME
        )),
        num_examples_per_day=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_DAY_ARG_NAME
        ),
        plot_basemap=bool(getattr(INPUT_ARG_OBJECT, PLOT_BASEMAP_ARG_NAME)),
        plot_deterministic=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_DETERMINISTIC_ARG_NAME
        )),
        probability_threshold=getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
