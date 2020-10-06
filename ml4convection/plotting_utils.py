"""Helper methods for plotting (mostly 2-D georeferenced maps)."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import longitude_conversion as lng_conversion
import error_checking

GRID_LINE_WIDTH = 1.
GRID_LINE_COLOUR = numpy.full(3, 0.)
DEFAULT_PARALLEL_SPACING_DEG = 2.
DEFAULT_MERIDIAN_SPACING_DEG = 2.

DEFAULT_BORDER_WIDTH = 2.
DEFAULT_BORDER_Z_ORDER = -1e8
DEFAULT_BORDER_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def plot_grid_lines(
        plot_latitudes_deg_n, plot_longitudes_deg_e, axes_object,
        parallel_spacing_deg=DEFAULT_PARALLEL_SPACING_DEG,
        meridian_spacing_deg=DEFAULT_MERIDIAN_SPACING_DEG,
        font_size=DEFAULT_FONT_SIZE):
    """Adds grid lines (parallels and meridians) to plot.

    :param plot_latitudes_deg_n: 1-D numpy array of latitudes in plot (deg N).
    :param plot_longitudes_deg_e: 1-D numpy array of longitudes in plot (deg E).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param parallel_spacing_deg: Spacing between adjacent parallels.
    :param meridian_spacing_deg: Spacing between adjacent meridians.
    :param font_size: Font size.
    """

    error_checking.assert_is_numpy_array(plot_latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(plot_latitudes_deg_n)
    error_checking.assert_is_numpy_array(
        plot_longitudes_deg_e, num_dimensions=1
    )
    plot_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        plot_longitudes_deg_e, allow_nan=False
    )

    error_checking.assert_is_greater(parallel_spacing_deg, 0.)
    error_checking.assert_is_greater(meridian_spacing_deg, 0.)
    error_checking.assert_is_greater(font_size, 0.)

    parallels_deg_n = numpy.unique(number_rounding.round_to_nearest(
        plot_latitudes_deg_n, parallel_spacing_deg
    ))
    parallels_deg_n = parallels_deg_n[
        parallels_deg_n >= numpy.min(plot_latitudes_deg_n)
    ]
    parallels_deg_n = parallels_deg_n[
        parallels_deg_n <= numpy.max(plot_latitudes_deg_n)
    ]
    parallel_label_strings = [
        '{0:.1f}'.format(p) if parallel_spacing_deg < 1.
        else '{0:d}'.format(int(numpy.round(p)))
        for p in parallels_deg_n
    ]
    parallel_label_strings = [
        s + r' $^{\circ}$N' for s in parallel_label_strings
    ]

    meridians_deg_e = numpy.unique(
        number_rounding.round_to_nearest(plot_longitudes_deg_e, 2.)
    )
    meridians_deg_e = meridians_deg_e[
        meridians_deg_e >= numpy.min(plot_longitudes_deg_e)
    ]
    meridians_deg_e = meridians_deg_e[
        meridians_deg_e <= numpy.max(plot_longitudes_deg_e)
    ]
    meridian_label_strings = [
        '{0:.1f}'.format(m) if meridian_spacing_deg < 1.
        else '{0:d}'.format(int(numpy.round(m)))
        for m in meridians_deg_e
    ]
    meridian_label_strings = [
        s + r' $^{\circ}$N' for s in meridian_label_strings
    ]

    axes_object.set_yticks(parallels_deg_n)
    axes_object.set_yticklabels(
        parallel_label_strings, fontdict={'fontsize': font_size}
    )

    axes_object.set_xticks(meridians_deg_e)
    axes_object.set_xticklabels(
        meridian_label_strings, fontdict={'fontsize': font_size}
    )

    axes_object.grid(
        b=True, which='major', axis='both', linestyle='--',
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    axes_object.set_xlim(
        numpy.min(plot_longitudes_deg_e), numpy.max(plot_longitudes_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(plot_latitudes_deg_n), numpy.max(plot_latitudes_deg_n)
    )


def plot_borders(
        border_latitudes_deg_n, border_longitudes_deg_e, axes_object,
        line_colour=DEFAULT_BORDER_COLOUR, line_width=DEFAULT_BORDER_WIDTH,
        z_order=DEFAULT_BORDER_Z_ORDER):
    """Adds borders to plot.

    P = number of points in border set

    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param line_colour: Line colour.
    :param line_width: Line width.
    :param z_order: z-order (lower values put borders near "back" of plot, and
        higher values put borders near "front").
    """

    error_checking.assert_is_numpy_array(
        border_latitudes_deg_n, num_dimensions=1
    )
    error_checking.assert_is_valid_lat_numpy_array(
        border_latitudes_deg_n, allow_nan=True
    )

    expected_dim = numpy.array([len(border_latitudes_deg_n)], dtype=int)
    error_checking.assert_is_numpy_array(
        border_longitudes_deg_e, exact_dimensions=expected_dim
    )
    border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        border_longitudes_deg_e, allow_nan=True
    )

    axes_object.plot(
        border_longitudes_deg_e, border_latitudes_deg_n, color=line_colour,
        linestyle='solid', linewidth=line_width, zorder=z_order
    )
