"""Plots results of model evaluation by hour and month."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import evaluation
import evaluation_plotting as eval_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TOLERANCE = 1e-6

NUM_HOURS_PER_DAY = 24

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4

FIRST_SCORE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SECOND_SCORE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
THIRD_SCORE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
FOURTH_SCORE_COLOUR = numpy.full(3, 0.)

HISTOGRAM_EDGE_WIDTH = 1.5
HISTOGRAM_FACE_COLOUR = numpy.full(3, 152. / 255)
HISTOGRAM_FACE_COLOUR = matplotlib.colors.to_rgba(HISTOGRAM_FACE_COLOUR, 0.5)
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 152. / 255)

LABEL_FONT_SIZE = 40
LABEL_BOUNDING_BOX_DICT = {
    'alpha': 0.5, 'edgecolor': 'k', 'linewidth': 1
}
TEMPORAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('hsv')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_dir_name'
PROB_THRESHOLD_ARG_NAME = 'probability_threshold'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`evaluation.find_advanced_score_file` and read by '
    '`evaluation.read_advanced_score_file`.'
)
PROB_THRESHOLD_HELP_STRING = (
    'Probability threshold used to compute POD, success ratio, and CSI.  If you'
    ' do not want to plot the aforelisted scores, leave this argument alone.'
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


def _plot_performance_diagrams(score_tables_xarray):
    """Plots performance diagrams.

    :param score_tables_xarray: 1-D list of tables in format returned by
        `evaluation.read_advanced_score_file`, where each table corresponds to
        either one month or one hour.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
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
            plot_background=i == 0, plot_csi_in_grey=True
        )

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

        real_indices = numpy.where(numpy.invert(numpy.logical_or(
            numpy.isnan(these_success_ratios), numpy.isnan(these_pod)
        )))[0]

        fill_values = (
            these_success_ratios[real_indices][0],
            these_success_ratios[real_indices][-1]
        )

        interp_object = interp1d(
            x=these_pod[real_indices], y=these_success_ratios[real_indices],
            kind='linear', assume_sorted=False, bounds_error=False,
            fill_value=fill_values
        )

        label_y_coord = 1. - float(i) / (num_tables - 1)
        label_x_coord = interp_object(label_y_coord)

        axes_object.text(
            label_x_coord, label_y_coord, label_string,
            fontsize=LABEL_FONT_SIZE, color=colour_matrix[i, ...],
            bbox=LABEL_BOUNDING_BOX_DICT, horizontalalignment='center',
            verticalalignment='center', zorder=1e10
        )

    return figure_object, axes_object


def _plot_reliability_curves(score_tables_xarray):
    """Plots reliability curves.

    :param score_tables_xarray: See doc for `_plot_performance_diagrams`.
    :return: figure_object: Same.
    :return: axes_object: Same.
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
            score_tables_xarray[i][evaluation.BINNED_MEAN_PROBS_KEY].values
        )
        these_event_freqs = (
            score_tables_xarray[i][evaluation.BINNED_EVENT_FREQS_KEY].values
        )
        eval_plotting.plot_reliability_curve(
            axes_object=axes_object,
            mean_predictions=these_mean_probs,
            mean_observations=these_event_freqs,
            min_value_to_plot=0., max_value_to_plot=1.,
            line_colour=colour_matrix[i, ...], plot_background=i == 0
        )

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

        real_indices = numpy.where(numpy.invert(numpy.logical_or(
            numpy.isnan(these_mean_probs), numpy.isnan(these_event_freqs)
        )))[0]

        fill_values = (
            these_event_freqs[real_indices][0],
            these_event_freqs[real_indices][-1]
        )

        interp_object = interp1d(
            x=these_mean_probs[real_indices], y=these_event_freqs[real_indices],
            kind='linear', assume_sorted=True, bounds_error=False,
            fill_value=fill_values
        )

        label_x_coord = float(i) / (num_tables - 1)
        label_y_coord = interp_object(label_x_coord)

        axes_object.text(
            label_x_coord, label_y_coord, label_string,
            fontsize=LABEL_FONT_SIZE, color=colour_matrix[i, ...],
            bbox=LABEL_BOUNDING_BOX_DICT, horizontalalignment='center',
            verticalalignment='center', zorder=1e10
        )

    return figure_object, axes_object


def _plot_scores_as_graph(score_tables_xarray, probability_threshold):
    """Plots scores vs. time as graph.

    :param score_tables_xarray: See doc for `_plot_performance_diagrams`.
    :param probability_threshold: Probability threshold at which to compute CSI
        and frequency bias.
    :return: figure_object: See doc for `_plot_performance_diagrams`.
    :return: axes_object: Same.
    :raises: ValueError: if desired probability threshold cannot be found.
    """

    # Housekeeping.
    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    histogram_axes_object = main_axes_object.twinx()
    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    main_axes_object.patch.set_visible(False)

    num_tables = len(score_tables_xarray)
    x_values = numpy.linspace(0, num_tables - 1, num=num_tables, dtype=float)

    # Plot FSS.
    fss_values = numpy.array([
        t[evaluation.FSS_KEY][0] for t in score_tables_xarray
    ])

    this_handle = main_axes_object.plot(
        x_values, fss_values, color=FIRST_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=FIRST_SCORE_COLOUR,
        markeredgecolor=FIRST_SCORE_COLOUR
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['FSS']

    # Plot BSS.
    bss_values = numpy.array([
        t[evaluation.BRIER_SKILL_SCORE_KEY][0] for t in score_tables_xarray
    ])

    this_handle = main_axes_object.plot(
        x_values, bss_values, color=SECOND_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=SECOND_SCORE_COLOUR,
        markeredgecolor=SECOND_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('BSS')

    # Plot CSI.
    all_prob_thresholds = score_tables_xarray[0].coords[
        evaluation.PROBABILITY_THRESHOLD_DIM
    ].values

    prob_threshold_index = numpy.argmin(numpy.absolute(
        all_prob_thresholds - probability_threshold
    ))
    min_difference = (
        all_prob_thresholds[prob_threshold_index] - probability_threshold
    )

    if min_difference > TOLERANCE:
        error_string = (
            'Cannot find desired probability threshold ({0:.6f}).  Nearest '
            'is {1:.6f}.'
        ).format(
            probability_threshold, all_prob_thresholds[prob_threshold_index]
        )

        raise ValueError(error_string)

    csi_values = numpy.array([
        t[evaluation.CSI_KEY][prob_threshold_index]
        for t in score_tables_xarray
    ])

    this_handle = main_axes_object.plot(
        x_values, csi_values, color=THIRD_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=THIRD_SCORE_COLOUR,
        markeredgecolor=THIRD_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('CSI')

    # Plot frequency bias.
    bias_values = numpy.array([
        t[evaluation.FREQUENCY_BIAS_KEY][prob_threshold_index]
        for t in score_tables_xarray
    ])

    this_handle = main_axes_object.plot(
        x_values, bias_values, color=FOURTH_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=FOURTH_SCORE_COLOUR,
        markeredgecolor=FOURTH_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Bias')

    y_min, y_max = main_axes_object.get_ylim()
    y_min = max([y_min, -1.])
    y_max = min([y_max, 2.])
    main_axes_object.set_ylim(y_min, y_max)

    # Plot event frequencies.
    example_counts = numpy.array([
        numpy.sum(t[evaluation.BINNED_NUM_EXAMPLES_KEY].values)
        for t in score_tables_xarray
    ], dtype=float)

    positive_example_counts = numpy.array([
        numpy.nansum(
            t[evaluation.BINNED_NUM_EXAMPLES_KEY].values *
            t[evaluation.BINNED_EVENT_FREQS_KEY].values
        )
        for t in score_tables_xarray
    ])

    example_counts[example_counts == 0] = numpy.nan
    event_frequencies = positive_example_counts / example_counts
    event_frequencies[numpy.isnan(event_frequencies)] = 0.

    histogram_axes_object.bar(
        x=x_values, height=event_frequencies, width=1.,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )

    histogram_axes_object.set_ylabel('Event frequency')
    print('Event frequency by split: {0:s}'.format(str(event_frequencies)))

    main_axes_object.legend(
        legend_handles, legend_strings, loc='lower center',
        bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
        ncol=2
    )

    return figure_object, main_axes_object


def _run(input_dir_name, probability_threshold, output_dir_name):
    """Plots results of model evaluation by hour and month.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    """

    if probability_threshold <= 0:
        probability_threshold = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    hours = numpy.linspace(0, 23, num=24, dtype=int)
    hourly_input_file_names = [
        evaluation.find_advanced_score_file(
            directory_name=input_dir_name, gridded=False, hour=h
        )
        for h in hours
    ]

    num_hours = len(hours)
    hourly_score_tables_xarray = [None] * num_hours

    for i in range(num_hours):
        print('Reading data from: "{0:s}"...'.format(
            hourly_input_file_names[i]
        ))
        hourly_score_tables_xarray[i] = evaluation.read_advanced_score_file(
            hourly_input_file_names[i]
        )

    months = numpy.linspace(1, 12, num=12, dtype=int)
    monthly_input_file_names = [
        evaluation.find_advanced_score_file(
            directory_name=input_dir_name, gridded=False, month=m
        )
        for m in months
    ]

    num_months = len(months)
    monthly_score_tables_xarray = [None] * num_months

    for i in range(num_months):
        print('Reading data from: "{0:s}"...'.format(
            monthly_input_file_names[i]
        ))
        monthly_score_tables_xarray[i] = evaluation.read_advanced_score_file(
            monthly_input_file_names[i]
        )

    print(SEPARATOR_STRING)

    # Plot hourly performance diagrams.
    figure_object, axes_object = _plot_performance_diagrams(
        score_tables_xarray=hourly_score_tables_xarray
    )
    axes_object.set_title('Performance diagram by UTC hour')

    output_file_name = '{0:s}/hourly_performance_diagrams.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot monthly performance diagrams.
    figure_object, axes_object = _plot_performance_diagrams(
        score_tables_xarray=monthly_score_tables_xarray
    )
    axes_object.set_title('Performance diagram by month')

    output_file_name = '{0:s}/monthly_performance_diagrams.jpg'.format(
        output_dir_name
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot hourly reliability curves.
    figure_object, axes_object = _plot_reliability_curves(
        score_tables_xarray=hourly_score_tables_xarray
    )
    axes_object.set_title('Reliability curve by UTC hour')

    output_file_name = '{0:s}/hourly_reliability_curves.jpg'.format(
        output_dir_name
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot monthly reliability curves.
    figure_object, axes_object = _plot_reliability_curves(
        score_tables_xarray=monthly_score_tables_xarray
    )
    axes_object.set_title('Reliability curve by month')

    output_file_name = '{0:s}/monthly_reliability_curves.jpg'.format(
        output_dir_name
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    if probability_threshold is None:
        return

    hour_strings = ['{0:02d}'.format(i) for i in range(24)]
    month_strings = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]

    # Plot other scores by hour.
    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=hourly_score_tables_xarray,
        probability_threshold=probability_threshold
    )

    axes_object.set_title('Other scores by UTC hour', y=1.2)
    axes_object.set_xticks(hours)
    axes_object.set_xticklabels(hour_strings, rotation=90.)
    axes_object.set_xlabel('UTC hour')

    output_file_name = '{0:s}/hourly_scores.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot other scores by month.
    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=monthly_score_tables_xarray,
        probability_threshold=probability_threshold
    )

    axes_object.set_title('Other scores by month', y=1.2)
    axes_object.set_xticks(months - 1)
    axes_object.set_xticklabels(month_strings, rotation=90.)

    output_file_name = '{0:s}/monthly_scores.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        probability_threshold=getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
