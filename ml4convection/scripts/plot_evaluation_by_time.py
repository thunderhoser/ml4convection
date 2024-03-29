"""Plots results of model evaluation by hour and month."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
import matplotlib.patches
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4convection.utils import evaluation
from ml4convection.plotting import evaluation_plotting as eval_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TOLERANCE = 1e-6

NUM_HOURS_PER_DAY = 24
UTC_OFFSET_HOURS = 8

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4

GRID_LINE_WIDTH = 1.
GRID_LINE_COLOUR = numpy.full(3, 0.)

FIRST_SCORE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SECOND_SCORE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
THIRD_SCORE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
FOURTH_SCORE_COLOUR = numpy.full(3, 0.)
POLYGON_OPACITY = 0.5

HISTOGRAM_EDGE_WIDTH = 1.5
HISTOGRAM_FACE_COLOUR = numpy.full(3, 152. / 255)
HISTOGRAM_FACE_COLOUR = matplotlib.colors.to_rgba(HISTOGRAM_FACE_COLOUR, 0.5)
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 152. / 255)

HOURLY_CBAR_FONT_SIZE = 35
MONTHLY_CBAR_FONT_SIZE = 40
TEMPORAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('hsv')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_dir_name'
PROB_THRESHOLD_ARG_NAME = 'probability_threshold'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
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
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence intervals (if number of bootstrap replicates > 1) will be '
    'plotted at this level.'
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
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_performance_diagrams(score_tables_xarray, confidence_level):
    """Plots performance diagrams.

    :param score_tables_xarray: 1-D list of tables in format returned by
        `evaluation.read_advanced_score_file`, where each table corresponds to
        either one month or one hour.
    :param confidence_level: See documentation at top of file.
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
    tick_strings = [''] * num_tables

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    for i in range(num_tables):
        these_pod = numpy.nanmean(
            score_tables_xarray[i][evaluation.POD_KEY].values, axis=0
        )
        these_success_ratios = numpy.nanmean(
            score_tables_xarray[i][evaluation.SUCCESS_RATIO_KEY].values, axis=0
        )

        if confidence_level is None:
            eval_plotting.plot_performance_diagram(
                axes_object=axes_object,
                pod_matrix=numpy.expand_dims(these_pod, axis=0),
                success_ratio_matrix=
                numpy.expand_dims(these_success_ratios, axis=0),
                confidence_level=confidence_level,
                line_colour=colour_matrix[i, ...],
                plot_background=i == 0, plot_csi_in_grey=True
            )
        else:
            eval_plotting.plot_performance_diagram(
                axes_object=axes_object,
                pod_matrix=score_tables_xarray[i][evaluation.POD_KEY].values,
                success_ratio_matrix=
                score_tables_xarray[i][evaluation.SUCCESS_RATIO_KEY].values,
                confidence_level=confidence_level,
                line_colour=colour_matrix[i, ...],
                plot_background=i == 0, plot_csi_in_grey=True
            )

        if num_tables == NUM_HOURS_PER_DAY:
            tick_strings[i] = '{0:02d}'.format(i)
        else:
            valid_time_string = '2000-{0:02d}'.format(i + 1)
            valid_time_unix_sec = time_conversion.string_to_unix_sec(
                valid_time_string, '%Y-%m'
            )
            tick_strings[i] = time_conversion.unix_sec_to_string(
                valid_time_unix_sec, '%b'
            )
            tick_strings[i] = '{0:s}{1:s}'.format(
                tick_strings[i][0].upper(), tick_strings[i][1:]
            )

    if num_tables == NUM_HOURS_PER_DAY:
        this_font_size = HOURLY_CBAR_FONT_SIZE
    else:
        this_font_size = MONTHLY_CBAR_FONT_SIZE

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=time_indices,
        colour_map_object=TEMPORAL_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', padding=0.11,
        extend_min=False, extend_max=False, font_size=this_font_size
    )

    colour_bar_object.set_ticks(time_indices)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _plot_reliability_curves(score_tables_xarray, confidence_level):
    """Plots reliability curves.

    :param score_tables_xarray: See doc for `_plot_performance_diagrams`.
    :param confidence_level: See documentation at top of file.
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
    tick_strings = [''] * num_tables

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    for i in range(num_tables):
        these_mean_probs = numpy.nanmean(
            score_tables_xarray[i][evaluation.BINNED_MEAN_PROBS_KEY].values,
            axis=0
        )
        these_event_freqs = numpy.nanmean(
            score_tables_xarray[i][evaluation.BINNED_EVENT_FREQS_KEY].values,
            axis=0
        )

        if confidence_level is None:
            eval_plotting.plot_reliability_curve(
                axes_object=axes_object,
                mean_prediction_matrix=
                numpy.expand_dims(these_mean_probs, axis=0),
                mean_observation_matrix=
                numpy.expand_dims(these_event_freqs, axis=0),
                confidence_level=confidence_level,
                min_value_to_plot=0., max_value_to_plot=1.,
                line_colour=colour_matrix[i, ...], plot_background=i == 0,
                plot_consistency_bars=False
            )
        else:
            eval_plotting.plot_reliability_curve(
                axes_object=axes_object,
                mean_prediction_matrix=
                score_tables_xarray[i][evaluation.BINNED_MEAN_PROBS_KEY].values,
                mean_observation_matrix=score_tables_xarray[i][
                    evaluation.BINNED_EVENT_FREQS_KEY
                ].values,
                confidence_level=confidence_level,
                min_value_to_plot=0., max_value_to_plot=1.,
                line_colour=colour_matrix[i, ...], plot_background=i == 0,
                plot_consistency_bars=False
            )

        if num_tables == NUM_HOURS_PER_DAY:
            tick_strings[i] = '{0:02d}'.format(i)
        else:
            valid_time_string = '2000-{0:02d}'.format(i + 1)
            valid_time_unix_sec = time_conversion.string_to_unix_sec(
                valid_time_string, '%Y-%m'
            )
            tick_strings[i] = time_conversion.unix_sec_to_string(
                valid_time_unix_sec, '%b'
            )
            tick_strings[i] = '{0:s}{1:s}'.format(
                tick_strings[i][0].upper(), tick_strings[i][1:]
            )

    if num_tables == NUM_HOURS_PER_DAY:
        this_font_size = HOURLY_CBAR_FONT_SIZE
    else:
        this_font_size = MONTHLY_CBAR_FONT_SIZE

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=time_indices,
        colour_map_object=TEMPORAL_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', padding=0.11,
        extend_min=False, extend_max=False, font_size=this_font_size
    )

    colour_bar_object.set_ticks(time_indices)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _plot_scores_as_graph(score_tables_xarray, probability_threshold,
                          confidence_level):
    """Plots scores vs. time as graph.

    :param score_tables_xarray: See doc for `_plot_performance_diagrams`.
    :param probability_threshold: Probability threshold at which to compute CSI
        and frequency bias.
    :param confidence_level: See documentation at top of file.
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
    fss_matrix = numpy.concatenate(
        [t[evaluation.FSS_KEY] for t in score_tables_xarray], axis=1
    )
    num_bootstrap_reps = fss_matrix.shape[0]

    this_handle = main_axes_object.plot(
        x_values, numpy.mean(fss_matrix, axis=0), color=FIRST_SCORE_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markeredgewidth=0, markerfacecolor=FIRST_SCORE_COLOUR,
        markeredgecolor=FIRST_SCORE_COLOUR
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['FSS']

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(x_values, axis=0)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=0, repeats=num_bootstrap_reps
        )

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_value_matrix, y_value_matrix=fss_matrix,
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(
            FIRST_SCORE_COLOUR, POLYGON_OPACITY
        )
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    # # Plot BSS.
    # bss_values = numpy.array([
    #     t[evaluation.BRIER_SKILL_SCORE_KEY][0] for t in score_tables_xarray
    # ])
    #
    # this_handle = main_axes_object.plot(
    #     x_values, bss_values, color=SECOND_SCORE_COLOUR, linewidth=LINE_WIDTH,
    #     marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
    #     markerfacecolor=SECOND_SCORE_COLOUR,
    #     markeredgecolor=SECOND_SCORE_COLOUR
    # )[0]
    #
    # legend_handles.append(this_handle)
    # legend_strings.append('BSS')

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

    csi_matrix = numpy.concatenate([
        t[evaluation.CSI_KEY][:, [prob_threshold_index]]
        for t in score_tables_xarray
    ], axis=1)

    this_handle = main_axes_object.plot(
        x_values, numpy.mean(csi_matrix, axis=0), color=SECOND_SCORE_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markeredgewidth=0, markerfacecolor=SECOND_SCORE_COLOUR,
        markeredgecolor=SECOND_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('CSI')

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(x_values, axis=0)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=0, repeats=num_bootstrap_reps
        )

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_value_matrix, y_value_matrix=csi_matrix,
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(
            SECOND_SCORE_COLOUR, POLYGON_OPACITY
        )
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    # Plot frequency bias.
    bias_matrix = numpy.concatenate([
        t[evaluation.FREQUENCY_BIAS_KEY][:, [prob_threshold_index]]
        for t in score_tables_xarray
    ], axis=1)

    this_handle = main_axes_object.plot(
        x_values, numpy.mean(bias_matrix, axis=0), color=THIRD_SCORE_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markeredgewidth=0, markerfacecolor=THIRD_SCORE_COLOUR,
        markeredgecolor=THIRD_SCORE_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Frequency bias')

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(x_values, axis=0)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=0, repeats=num_bootstrap_reps
        )

        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=x_value_matrix, y_value_matrix=bias_matrix,
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(
            THIRD_SCORE_COLOUR, POLYGON_OPACITY
        )
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        main_axes_object.add_patch(patch_object)

    # y_min, y_max = main_axes_object.get_ylim()
    # y_min = max([y_min, -1.])
    # y_max = min([y_max, 2.])
    # main_axes_object.set_ylim(y_min, y_max)

    main_axes_object.set_ylim(0, 2)

    main_axes_object.grid(
        b=True, which='major', axis='y', linestyle='--',
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    # Plot event frequencies.
    example_counts = numpy.array([
        numpy.sum(t[evaluation.BINNED_NUM_EXAMPLES_KEY].values)
        for t in score_tables_xarray
    ], dtype=float)

    positive_example_counts = numpy.array([
        numpy.nansum(
            t[evaluation.BINNED_NUM_EXAMPLES_KEY].values *
            numpy.nanmean(t[evaluation.BINNED_EVENT_FREQS_KEY].values, axis=0)
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


def _run(input_dir_name, probability_threshold, confidence_level,
         output_dir_name):
    """Plots results of model evaluation by hour and month.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param probability_threshold: Same.
    :param confidence_level: Same.
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

    hourly_input_file_names = (
        hourly_input_file_names[-UTC_OFFSET_HOURS:] +
        hourly_input_file_names[:-UTC_OFFSET_HOURS]
    )

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

    all_prob_thresholds = monthly_score_tables_xarray[0].coords[
        evaluation.PROBABILITY_THRESHOLD_DIM
    ].values

    num_prob_thresholds = len(all_prob_thresholds)

    if num_prob_thresholds >= 5:

        # Plot hourly performance diagrams.
        figure_object, axes_object = _plot_performance_diagrams(
            score_tables_xarray=hourly_score_tables_xarray,
            confidence_level=None
        )
        axes_object.set_title('Performance diagram by hour')

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
            score_tables_xarray=monthly_score_tables_xarray,
            confidence_level=None
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
            score_tables_xarray=hourly_score_tables_xarray,
            confidence_level=None
        )
        axes_object.set_title('Reliability curve by hour')

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
            score_tables_xarray=monthly_score_tables_xarray,
            confidence_level=None
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

    # Plot scalar scores by hour.
    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=hourly_score_tables_xarray,
        probability_threshold=probability_threshold,
        confidence_level=confidence_level
    )

    axes_object.set_title('Scalar scores by hour', y=1.2)
    axes_object.set_xticks(hours)
    axes_object.set_xticklabels(hour_strings, rotation=90.)
    axes_object.set_xlabel('Hour (Taipei Standard Time)')

    output_file_name = '{0:s}/hourly_scores.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot scalar scores by month.
    figure_object, axes_object = _plot_scores_as_graph(
        score_tables_xarray=monthly_score_tables_xarray,
        probability_threshold=probability_threshold,
        confidence_level=confidence_level
    )

    axes_object.set_title('Scalar scores by month', y=1.2)
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
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
