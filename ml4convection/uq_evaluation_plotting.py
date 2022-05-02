"""Plotting methods for evaluation of uncertainty quantification (UQ)."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import uq_evaluation

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

DEFAULT_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

HISTOGRAM_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 2.

MEAN_PREDICTION_LINE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
MEAN_PREDICTION_COLOUR_STRING = 'purple'
MEAN_TARGET_LINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
MEAN_TARGET_COLOUR_STRING = 'green'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 40
INSET_FONT_SIZE = 20

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _plot_means_as_inset(
        figure_object, bin_centers, bin_mean_predictions,
        bin_mean_target_values):
    """Plots means (mean prediction and target by bin) as inset in another fig.

    B = number of bins

    :param figure_object: Will plot as inset in this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_mean_predictions: length-B numpy array with mean prediction in
        each bin.  These values will be plotted on the y-axis.
    :param bin_mean_target_values: length-B numpy array with mean target value
        (event frequency) in each bin.  These values will be plotted on the
        y-axis.
    :return: inset_axes_object: Axes handle for histogram (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # inset_axes_object = figure_object.add_axes([0.575, 0.2, 0.3, 0.3])
    inset_axes_object = figure_object.add_axes([0.625, 0.175, 0.25, 0.25])

    inset_axes_object.plot(
        bin_centers, bin_mean_target_values, color=MEAN_TARGET_LINE_COLOUR,
        linestyle='solid', linewidth=2,
        marker='o', markersize=8, markeredgewidth=0,
        markerfacecolor=MEAN_TARGET_LINE_COLOUR,
        markeredgecolor=MEAN_TARGET_LINE_COLOUR
    )
    inset_axes_object.plot(
        bin_centers, bin_mean_predictions, color=MEAN_PREDICTION_LINE_COLOUR,
        linestyle='dashed', linewidth=2,
        marker='o', markersize=8, markeredgewidth=0,
        markerfacecolor=MEAN_PREDICTION_LINE_COLOUR,
        markeredgecolor=MEAN_PREDICTION_LINE_COLOUR
    )

    y_max = max([
        numpy.nanmax(bin_mean_predictions),
        numpy.nanmax(bin_mean_target_values)
    ])
    inset_axes_object.set_ylim(0, y_max)
    inset_axes_object.set_xlim(left=0.)

    for this_tick_object in inset_axes_object.xaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)

    return inset_axes_object


def _plot_histogram(axes_object, bin_centers, bin_frequencies):
    """Plots histogram on existing axes.

    B = number of bins

    :param axes_object: Will plot histogram on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_frequencies: length-B numpy array with fraction of examples in
        each bin. These values will be plotted on the y-axis.
    :return: histogram_axes_object: Axes handle for histogram only (also
        instance of `matplotlib.axes._subplots.AxesSubplot`).
    """

    histogram_axes_object = axes_object.twinx()
    axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    axes_object.patch.set_visible(False)

    bin_diffs = numpy.diff(bin_centers)
    bin_edges = bin_centers[:-1] + bin_diffs / 2
    bottom_bin_edge = bin_centers[0] - bin_diffs[0] / 2
    top_bin_edge = bin_centers[-1] + bin_diffs[-1] / 2

    bin_edges = numpy.concatenate((
        numpy.array([bottom_bin_edge]), bin_edges, numpy.array([top_bin_edge])
    ))

    histogram_axes_object.bar(
        x=bin_edges[:-1], height=bin_frequencies, width=numpy.diff(bin_edges),
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH, align='edge'
    )

    return histogram_axes_object


def _plot_inset_histogram(
        figure_object, bin_centers, bin_frequencies,
        bar_colour=HISTOGRAM_FACE_COLOUR):
    """Plots histogram as inset in another figure.

    B = number of bins

    :param figure_object: Will plot as inset in this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_frequencies: length-B numpy array with fraction of examples in
        each bin. These values will be plotted on the y-axis.
    :param bar_colour: Bar colour (in any format accepted by matplotlib).
    :return: inset_axes_object: Axes handle for histogram (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    inset_axes_object = figure_object.add_axes([0.1875, 0.6, 0.25, 0.25])

    num_bins = len(bin_centers)
    fake_bin_centers = (
        0.5 + numpy.linspace(0, num_bins - 1, num=num_bins, dtype=float)
    )

    real_indices = numpy.where(numpy.invert(numpy.isnan(bin_centers)))[0]

    inset_axes_object.bar(
        fake_bin_centers[real_indices], bin_frequencies[real_indices], 1.,
        color=bar_colour, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )
    inset_axes_object.set_ylim(bottom=0.)

    tick_indices = []

    for i in real_indices:
        if numpy.mod(i, 2) == 0:
            tick_indices.append(i)
            continue

        if i - 1 in real_indices or i + 1 in real_indices:
            continue

        tick_indices.append(i)

    x_tick_values = fake_bin_centers[tick_indices]
    x_tick_labels = ['{0:.2g}'.format(b) for b in bin_centers[tick_indices]]
    inset_axes_object.set_xticks(x_tick_values)
    inset_axes_object.set_xticklabels(x_tick_labels)

    for this_tick_object in inset_axes_object.xaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)

    return inset_axes_object


def plot_spread_vs_skill(
        result_dict, line_colour=DEFAULT_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH):
    """Creates spread-skill plot, as in Delle Monache et al. (2013).

    Delle Monache et al. (2013): https://doi.org/10.1175/MWR-D-12-00281.1

    :param result_dict: Dictionary in format returned by
        `uq_evaluation.get_spread_vs_skill`.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Check input args.
    mean_prediction_stdevs = (
        result_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY]
    )
    rmse_values = result_dict[uq_evaluation.RMSE_VALUES_KEY]

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_prediction_stdevs),
        numpy.isnan(rmse_values)
    )
    assert not numpy.all(nan_flags)

    # Do actual stuff.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    max_value_to_plot = max([
        numpy.nanmax(mean_prediction_stdevs),
        numpy.nanmax(rmse_values)
    ])
    perfect_x_coords = numpy.array([0, max_value_to_plot])
    perfect_y_coords = numpy.array([0, max_value_to_plot])
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    real_indices = numpy.where(numpy.invert(nan_flags))[0]
    axes_object.plot(
        mean_prediction_stdevs[real_indices],
        rmse_values[real_indices],
        color=line_colour, linestyle=line_style, linewidth=line_width
    )

    axes_object.set_xlabel('Spread (stdev of predictive distribution)')
    axes_object.set_ylabel('Skill (RMSE of central prediction)')
    axes_object.set_xlim(0, max_value_to_plot)
    axes_object.set_ylim(0, max_value_to_plot)

    bin_frequencies = (
        result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY].astype(float) /
        numpy.sum(result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY])
    )

    histogram_axes_object = _plot_histogram(
        axes_object=axes_object,
        bin_centers=mean_prediction_stdevs,
        bin_frequencies=bin_frequencies
    )
    histogram_axes_object.set_ylabel('% examples in each bin')

    # inset_axes_object = _plot_inset_histogram(
    #     figure_object=figure_object,
    #     bin_centers=mean_prediction_stdevs,
    #     bin_frequencies=bin_frequencies
    # )
    # inset_axes_object.set_title(
    #     '% examples in each bin', fontsize=INSET_FONT_SIZE
    # )
    # inset_axes_object.set_xlabel('Spread', fontsize=INSET_FONT_SIZE)

    inset_axes_object = _plot_means_as_inset(
        figure_object=figure_object, bin_centers=mean_prediction_stdevs,
        bin_mean_predictions=
        result_dict[uq_evaluation.MEAN_CENTRAL_PREDICTIONS_KEY],
        bin_mean_target_values=result_dict[uq_evaluation.MEAN_TARGET_VALUES_KEY]
    )

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())

    title_string = (
        'Mean target ({0:s}) & pred\n({1:s}) in each bin'
    ).format(MEAN_TARGET_COLOUR_STRING, MEAN_PREDICTION_COLOUR_STRING)

    inset_axes_object.set_title(title_string, fontsize=INSET_FONT_SIZE)
    inset_axes_object.set_xlabel('Spread', fontsize=INSET_FONT_SIZE)

    return figure_object, axes_object


def plot_discard_test(
        result_dict, line_colour=DEFAULT_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH):
    """Plots results of discard test.

    The "discard test" (I couldn't think of a better name for it) is the one
    presented in Tables 4-6 of Ortiz et al. (2022):
    https://doi.org/10.1109/TGRS.2022.3152516

    :param result_dict: Dictionary in format returned by
        `uq_evaluation.run_discard_test`.
    :param line_colour: See doc for `plot_spread_vs_skill`.
    :param line_style: Same.
    :param line_width: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    """

    discard_fractions = result_dict[uq_evaluation.DISCARD_FRACTIONS_KEY]
    error_values = result_dict[uq_evaluation.ERROR_VALUES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        discard_fractions, error_values,
        color=line_colour, linestyle=line_style, linewidth=line_width
    )

    axes_object.set_xlabel('Discard fraction')
    axes_object.set_ylabel('Performance measure')
    axes_object.set_xlim(left=0.)

    histogram_axes_object = _plot_histogram(
        axes_object=axes_object,
        bin_centers=discard_fractions,
        bin_frequencies=result_dict[uq_evaluation.EXAMPLE_FRACTIONS_KEY]
    )
    histogram_axes_object.set_ylabel('% examples left after discard')

    # inset_axes_object = _plot_inset_histogram(
    #     figure_object=figure_object,
    #     bin_centers=discard_fractions,
    #     bin_frequencies=result_dict[uq_evaluation.EXAMPLE_FRACTIONS_KEY]
    # )
    # inset_axes_object.set_title(
    #     '% examples left after discard', fontsize=INSET_FONT_SIZE
    # )
    # inset_axes_object.set_xlabel('Discard fraction', fontsize=INSET_FONT_SIZE)

    inset_axes_object = _plot_means_as_inset(
        figure_object=figure_object, bin_centers=discard_fractions,
        bin_mean_predictions=
        result_dict[uq_evaluation.MEAN_CENTRAL_PREDICTIONS_KEY],
        bin_mean_target_values=result_dict[uq_evaluation.MEAN_TARGET_VALUES_KEY]
    )

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())

    title_string = (
        'Mean target ({0:s}) & pred\n({1:s}) after discard'
    ).format(MEAN_TARGET_COLOUR_STRING, MEAN_PREDICTION_COLOUR_STRING)

    inset_axes_object.set_title(title_string, fontsize=INSET_FONT_SIZE)
    inset_axes_object.set_xlabel('Discard fraction', fontsize=INSET_FONT_SIZE)

    return figure_object, axes_object
