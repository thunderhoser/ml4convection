"""Plots results of model evaluation by hour and month."""

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

import time_conversion
import file_system_utils
import evaluation
import evaluation_plotting as eval_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_HOURS_PER_DAY = 24

LABEL_FONT_SIZE = 16
LABEL_BOUNDING_BOX_DICT = {
    'alpha': 0.5, 'edgecolor': 'k', 'linewidth': 1
}
TEMPORAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('twilight')

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_DIR_ARG_NAME = 'input_dir_name'
PROB_THRESHOLD_ARG_NAME = 'probability_threshold'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`evaluation.find_advanced_score_file` and read by `evaluation.read_file`.'
)
PROB_THRESHOLD_HELP_STRING = (
    'Probability threshold used to compute CSI, POD, and FAR.  If you do not '
    'want to plot the aforelisted scores, leave this argument alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=PROB_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_performance_diagrams(score_tables_xarray, output_file_name):
    """Plots performance diagrams.

    :param score_tables_xarray: 1-D list of tables in format returned by
        `evaluation.read_file`, where each table corresponds to either one month
        or one hour.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    num_tables = len(score_tables_xarray)
    time_indices = numpy.linspace(
        0, num_tables - 1, num=num_tables, dtype=float
    )

    colour_norm_object = pyplot.Normalize(
        numpy.min(time_indices), numpy.max(time_indices)
    )
    colour_matrix = TEMPORAL_COLOUR_MAP_OBJECT(colour_norm_object(time_indices))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    for i in range(num_tables):
        these_pod = score_tables_xarray[i][evaluation.POD_KEY].values
        these_success_ratios = (
            score_tables_xarray[i][evaluation.SUCCESS_RATIO_KEY].values
        )
        eval_plotting.plot_performance_diagram(
            axes_object=axes_object, pod_by_threshold=these_pod,
            success_ratio_by_threshold=these_success_ratios,
            line_colour=colour_matrix[i, ...],
            plot_background=i == 0, plot_csi_in_green=True
        )

        normalized_index = float(i) / (num_tables - 1)
        label_index = int(numpy.round(
            normalized_index * (len(these_pod) - 1)
        ))

        if num_tables == NUM_HOURS_PER_DAY:
            label_string = '{0:02d}'.format(i)
        else:
            valid_time_string = '2000-{0:02d}'.format(i + 1)
            valid_time_unix_sec = time_conversion.string_to_unix_sec(
                valid_time_string, '%Y-%m'
            )
            label_string = time_conversion.unix_sec_to_string(
                valid_time_unix_sec, '%b'
            )
            label_string = '{0:s}{1:s}'.format(
                label_string[0].upper(), label_string[1:]
            )

        axes_object.text(
            these_success_ratios[label_index], these_pod[label_index],
            label_string, fontsize=LABEL_FONT_SIZE, color=colour_matrix[i, ...],
            bbox=LABEL_BOUNDING_BOX_DICT, horizontalalignment='center',
            verticalalignment='center', zorder=1e10
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_reliability_curves(score_tables_xarray, output_file_name):
    """Plots reliability curves.

    :param score_tables_xarray: See doc for `_plot_performance_diagrams`.
    :param output_file_name: Same.
    """

    num_tables = len(score_tables_xarray)
    time_indices = numpy.linspace(
        0, num_tables - 1, num=num_tables, dtype=float
    )

    colour_norm_object = pyplot.Normalize(
        numpy.min(time_indices), numpy.max(time_indices)
    )
    colour_matrix = TEMPORAL_COLOUR_MAP_OBJECT(colour_norm_object(time_indices))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    for i in range(num_tables):
        these_mean_probs = (
            score_tables_xarray[i][evaluation.MEAN_FORECAST_PROB_KEY].values
        )
        these_event_freqs = (
            score_tables_xarray[i][evaluation.EVENT_FREQUENCY_KEY].values
        )
        eval_plotting.plot_reliability_curve(
            axes_object=axes_object,
            mean_predictions=these_mean_probs,
            mean_observations=these_event_freqs,
            min_value_to_plot=0., max_value_to_plot=1.,
            line_colour=colour_matrix[i, ...], plot_background=i == 0
        )

        normalized_index = float(i) / (num_tables - 1)
        label_index = int(numpy.round(
            normalized_index * (len(these_mean_probs) - 1)
        ))

        if num_tables == NUM_HOURS_PER_DAY:
            label_string = '{0:02d}'.format(i)
        else:
            valid_time_string = '2000-{0:02d}'.format(i + 1)
            valid_time_unix_sec = time_conversion.string_to_unix_sec(
                valid_time_string, '%Y-%m'
            )
            label_string = time_conversion.unix_sec_to_string(
                valid_time_unix_sec, '%b'
            )
            label_string = '{0:s}{1:s}'.format(
                label_string[0].upper(), label_string[1:]
            )

        axes_object.text(
            these_mean_probs[label_index], these_event_freqs[label_index],
            label_string, fontsize=LABEL_FONT_SIZE, color=colour_matrix[i, ...],
            bbox=LABEL_BOUNDING_BOX_DICT, horizontalalignment='center',
            verticalalignment='center', zorder=1e10
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(input_dir_name, probability_threshold, output_dir_name):
    """Plots results of model evaluation by hour and month.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    hours = numpy.linspace(0, 23, num=24, dtype=int)
    hourly_input_file_names = [
        evaluation.find_advanced_score_file(
            directory_name=input_dir_name, hour=h
        )
        for h in hours
    ]

    num_hours = len(hours)
    hourly_score_tables_xarray = [None] * num_hours

    for i in range(num_hours):
        print('Reading data from: "{0:s}"...'.format(
            hourly_input_file_names[i]
        ))
        hourly_score_tables_xarray[i] = evaluation.read_file(
            hourly_input_file_names[i]
        )

    months = numpy.linspace(1, 12, num=12, dtype=int)
    monthly_input_file_names = [
        evaluation.find_advanced_score_file(
            directory_name=input_dir_name, month=m
        )
        for m in months
    ]

    num_months = len(months)
    monthly_score_tables_xarray = [None] * num_months

    for i in range(num_months):
        print('Reading data from: "{0:s}"...'.format(
            monthly_input_file_names[i]
        ))
        monthly_score_tables_xarray[i] = evaluation.read_file(
            monthly_input_file_names[i]
        )

    print(SEPARATOR_STRING)

    _plot_performance_diagrams(
        score_tables_xarray=hourly_score_tables_xarray,
        output_file_name=
        '{0:s}/hourly_performance_diagrams.jpg'.format(output_dir_name)
    )

    _plot_performance_diagrams(
        score_tables_xarray=monthly_score_tables_xarray,
        output_file_name=
        '{0:s}/monthly_performance_diagrams.jpg'.format(output_dir_name)
    )

    _plot_reliability_curves(
        score_tables_xarray=hourly_score_tables_xarray,
        output_file_name=
        '{0:s}/hourly_reliability_curves.jpg'.format(output_dir_name)
    )

    _plot_reliability_curves(
        score_tables_xarray=monthly_score_tables_xarray,
        output_file_name=
        '{0:s}/monthly_reliability_curves.jpg'.format(output_dir_name)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        probability_threshold=getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
