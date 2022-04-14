"""Plotting methods for evaluation of uncertainty quantification (UQ)."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import error_checking

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

DEFAULT_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 40
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def plot_spread_vs_skill(
        mean_prediction_stdevs, prediction_standard_errors,
        line_colour=DEFAULT_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH):
    """Creates spread-skill plot, as in Delle Monache et al. (2013).

    Delle Monache et al. (2013): https://doi.org/10.1175/MWR-D-12-00281.1

    B = number of bins

    :param mean_prediction_stdevs: length-B numpy array, where the [i]th
        entry is the mean standard deviation of predictive distributions in the
        [i]th bin.
    :param prediction_standard_errors: length-B numpy array, where the [i]th
        entry is the standard deviation of mean absolute errors in the [i]th
        bin.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axex handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(
        mean_prediction_stdevs, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(
        mean_prediction_stdevs, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        mean_prediction_stdevs, 1., allow_nan=True
    )

    num_bins = len(mean_prediction_stdevs)
    assert num_bins >= 2

    error_checking.assert_is_numpy_array(
        prediction_standard_errors,
        exact_dimensions=numpy.array([num_bins], dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        prediction_standard_errors, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        prediction_standard_errors, 1., allow_nan=True
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_prediction_stdevs),
        numpy.isnan(prediction_standard_errors)
    )
    assert not numpy.all(nan_flags)

    # Do actual stuff.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    max_value_to_plot = max([
        numpy.nanmax(mean_prediction_stdevs),
        numpy.nanmax(prediction_standard_errors)
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
        prediction_standard_errors[real_indices],
        color=line_colour, linestyle=line_style, linewidth=line_width
    )

    axes_object.set_xlabel('Spread (stdev of predictive distribution)')
    axes_object.set_ylabel('Skill (RMSE of central prediction)')
    axes_object.set_xlim(0, max_value_to_plot)
    axes_object.set_ylim(0, max_value_to_plot)

    return figure_object, axes_object
