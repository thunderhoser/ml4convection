"""Plots results of model evaluation by hour and month."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.utils import evaluation
from ml4convection.plotting import evaluation_plotting as eval_plotting

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

LABEL_FONT_SIZE = 16
LABEL_BOUNDING_BOX_DICT = {
    'alpha': 0.5, 'edgecolor': 'k', 'linewidth': 1
}
TEMPORAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('twilight')

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


def _plot_performance_diagrams(score_tables_xarray, output_file_name):
    """Plots performance diagrams.

    :param score_tables_xarray: 1-D list of tables in format returned by
        `evaluation.read_advanced_score_file`, where each table corresponds to
        either one month or one hour.
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
            normalized_index * (len(these_pod) - 4)
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


def _plot_scores_as_graph(
        score_tables_xarray, plot_total_example_counts, plot_legend,
        probability_threshold=None):
    """Plots scores vs. time as graph.

    :param score_tables_xarray: See doc for `_plot_performance_diagrams`.
    :param plot_total_example_counts: Boolean flag.  If True (False), will plot
        histogram with number of total (positive) examples for each time split.
    :param plot_legend: Boolean flag.
    :param probability_threshold: Probability threshold at which to compute
        scores.  If specified, this method will plot POD, success ratio, CSI,
        and frequency bias.  If None, this method will plot BSS, FSS, and AUPD.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
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

    # Plot first score.
    prob_threshold_index = -1

    if probability_threshold is None:
        y_values = numpy.array([
            t[evaluation.FSS_KEY][0] for t in score_tables_xarray
        ])
    else:
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

        y_values = numpy.array([
            t[evaluation.CSI_KEY][prob_threshold_index]
            for t in score_tables_xarray
        ])

    this_handle = main_axes_object.plot(
        x_values, y_values, color=FIRST_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=FIRST_SCORE_COLOUR,
        markeredgecolor=FIRST_SCORE_COLOUR
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['FSS' if probability_threshold is None else 'CSI']

    # Plot second score.
    if probability_threshold is None:
        y_values = numpy.array([
            t[evaluation.BRIER_SKILL_SCORE_KEY][0]
            for t in score_tables_xarray
        ])
    else:
        y_values = numpy.array([
            t[evaluation.FREQUENCY_BIAS_KEY][prob_threshold_index]
            for t in score_tables_xarray
        ])

    this_handle = main_axes_object.plot(
        x_values, y_values, color=SECOND_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=SECOND_SCORE_COLOUR,
        markeredgecolor=SECOND_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(
        'BSS' if probability_threshold is None else 'Freq bias'
    )

    # Plot third score.
    if probability_threshold is None:
        y_values = numpy.array([
            t[evaluation.BRIER_SCORE_KEY][0]
            for t in score_tables_xarray
        ])
    else:
        y_values = numpy.array([
            t[evaluation.POD_KEY][prob_threshold_index]
            for t in score_tables_xarray
        ])

    this_handle = main_axes_object.plot(
        x_values, y_values, color=THIRD_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=THIRD_SCORE_COLOUR,
        markeredgecolor=THIRD_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(
        'BS' if probability_threshold is None else 'POD'
    )

    # Plot fourth score.
    if probability_threshold is None:
        y_values = numpy.array([
            gg_model_eval.get_area_under_perf_diagram(
                pod_by_threshold=t[evaluation.POD_KEY].values,
                success_ratio_by_threshold=
                t[evaluation.SUCCESS_RATIO_KEY].values
            )
            for t in score_tables_xarray
        ])
    else:
        y_values = numpy.array([
            t[evaluation.SUCCESS_RATIO_KEY][prob_threshold_index]
            for t in score_tables_xarray
        ])

    this_handle = main_axes_object.plot(
        x_values, y_values, color=FOURTH_SCORE_COLOUR, linewidth=LINE_WIDTH,
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=FOURTH_SCORE_COLOUR,
        markeredgecolor=FOURTH_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(
        'AUPD' if probability_threshold is None else 'Success ratio'
    )

    # Plot histogram.
    if plot_total_example_counts:
        y_values = numpy.array([
            numpy.sum(t[evaluation.EXAMPLE_COUNT_KEY].values)
            for t in score_tables_xarray
        ], dtype=int)

        histogram_name = r'Num examples'
    else:
        y_values = numpy.array([
            numpy.nansum(
                t[evaluation.EXAMPLE_COUNT_KEY].values *
                t[evaluation.EVENT_FREQUENCY_KEY].values
            )
            for t in score_tables_xarray
        ])

        y_values = numpy.round(y_values).astype(int)
        histogram_name = r'Num positive examples'

    # y_values = numpy.maximum(numpy.log10(y_values), 0.)

    this_handle = histogram_axes_object.bar(
        x=x_values, height=y_values, width=1., color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR, linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(histogram_name)
    histogram_axes_object.set_ylabel(histogram_name)

    # tick_values = histogram_axes_object.get_yticks()
    # tick_strings = [
    #     '{0:d}'.format(int(numpy.round(10 ** v))) for v in tick_values
    # ]
    # histogram_axes_object.set_yticklabels(tick_strings)

    # print('{0:s} by split: {1:s}'.format(
    #     histogram_name, str(10 ** y_values)
    # ))
    print('{0:s} by split: {1:s}'.format(
        histogram_name, str(y_values)
    ))

    if plot_legend:
        main_axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
            ncol=len(legend_handles)
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

    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=hourly_score_tables_xarray,
        plot_total_example_counts=True, plot_legend=True,
        probability_threshold=None
    )

    hour_strings = ['{0:02d}'.format(i) for i in range(24)]
    axes_object.set_xticks(hours)
    axes_object.set_xticklabels(hour_strings, rotation=90.)
    axes_object.set_xlabel('Hour (UTC)')

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(a)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    hourly_with_total_file_name = (
        '{0:s}/hourly_scores_with_total_counts.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(hourly_with_total_file_name))
    figure_object.savefig(
        hourly_with_total_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=hourly_score_tables_xarray,
        plot_total_example_counts=False, plot_legend=True,
        probability_threshold=probability_threshold
    )

    axes_object.set_xticks(hours)
    axes_object.set_xticklabels(hour_strings, rotation=90.)
    axes_object.set_xlabel('Hour (UTC)')

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(b)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    hourly_with_positive_file_name = (
        '{0:s}/hourly_scores_with_positive_counts.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(hourly_with_positive_file_name))
    figure_object.savefig(
        hourly_with_positive_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=monthly_score_tables_xarray,
        plot_total_example_counts=True, plot_legend=False,
        probability_threshold=None
    )

    month_strings = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    axes_object.set_xticks(months - 1)
    axes_object.set_xticklabels(month_strings, rotation=90.)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(c)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    monthly_with_total_file_name = (
        '{0:s}/monthly_scores_with_total_counts.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(monthly_with_total_file_name))
    figure_object.savefig(
        monthly_with_total_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=monthly_score_tables_xarray,
        plot_total_example_counts=False, plot_legend=False,
        probability_threshold=probability_threshold
    )

    axes_object.set_xticks(months - 1)
    axes_object.set_xticklabels(month_strings, rotation=90.)

    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(d)',
        x_coord_normalized=-0.075, y_coord_normalized=1.02
    )

    monthly_with_positive_file_name = (
        '{0:s}/monthly_scores_with_positive_counts.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(
        monthly_with_positive_file_name
    ))
    figure_object.savefig(
        monthly_with_positive_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}/scores_by_time.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    panel_file_names = [
        hourly_with_total_file_name, hourly_with_positive_file_name,
        monthly_with_total_file_name, monthly_with_positive_file_name
    ]
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
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
