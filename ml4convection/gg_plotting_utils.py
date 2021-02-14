"""Helper methods for plotting (mostly 2-D georeferenced maps)."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

VERTICAL_CBAR_PADDING = 0.05
HORIZONTAL_CBAR_PADDING = 0.075
DEFAULT_CBAR_ORIENTATION_STRING = 'horizontal'

DEFAULT_FIGURE_WIDTH_INCHES = 15.
DEFAULT_FIGURE_HEIGHT_INCHES = 15.

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def colour_from_numpy_to_tuple(input_colour):
    """Converts colour from numpy array to tuple (if necessary).

    :param input_colour: Colour (possibly length-3 or length-4 numpy array).
    :return: output_colour: Colour (possibly length-3 or length-4 tuple).
    """

    if not isinstance(input_colour, numpy.ndarray):
        return input_colour

    error_checking.assert_is_numpy_array(input_colour, num_dimensions=1)

    num_entries = len(input_colour)
    error_checking.assert_is_geq(num_entries, 3)
    error_checking.assert_is_leq(num_entries, 4)

    return tuple(input_colour.tolist())


def plot_colour_bar(
        axes_object_or_matrix, data_matrix, colour_map_object,
        colour_norm_object, orientation_string=DEFAULT_CBAR_ORIENTATION_STRING,
        padding=None, extend_min=True, extend_max=True,
        fraction_of_axis_length=1., font_size=FONT_SIZE):
    """Plots colour bar.

    :param axes_object_or_matrix: Either one axis handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`) or a numpy array thereof.
    :param data_matrix: numpy array of values to which the colour map applies.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm` or
        similar).
    :param colour_norm_object: Colour normalization (maps from data space to
        colour-bar space, which goes from 0...1).  This should be an instance of
        `matplotlib.colors.Normalize`.
    :param orientation_string: Orientation ("vertical" or "horizontal").
    :param padding: Padding between colour bar and main plot (in range 0...1).
        To use the default (there are different defaults for vertical and horiz
        colour bars), leave this alone.
    :param extend_min: Boolean flag.  If True, values below the minimum
        specified by `colour_norm_object` are possible, so the colour bar will
        be plotted with an arrow at the bottom.
    :param extend_max: Boolean flag.  If True, values above the max specified by
        `colour_norm_object` are possible, so the colour bar will be plotted
        with an arrow at the top.
    :param fraction_of_axis_length: The colour bar will take up this fraction of
        the axis length (x-axis if orientation_string = "horizontal", y-axis if
        orientation_string = "vertical").
    :param font_size: Font size for tick marks on colour bar.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).
    """

    error_checking.assert_is_real_numpy_array(data_matrix)
    error_checking.assert_is_boolean(extend_min)
    error_checking.assert_is_boolean(extend_max)
    error_checking.assert_is_greater(fraction_of_axis_length, 0.)
    # error_checking.assert_is_leq(fraction_of_axis_length, 1.)

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(data_matrix)

    if extend_min and extend_max:
        extend_arg = 'both'
    elif extend_min:
        extend_arg = 'min'
    elif extend_max:
        extend_arg = 'max'
    else:
        extend_arg = 'neither'

    if padding is None:
        if orientation_string == 'horizontal':
            padding = HORIZONTAL_CBAR_PADDING
        else:
            padding = VERTICAL_CBAR_PADDING

    # error_checking.assert_is_geq(padding, 0.)
    # error_checking.assert_is_leq(padding, 1.)
    error_checking.assert_is_real_number(padding)

    if isinstance(axes_object_or_matrix, numpy.ndarray):
        axes_arg = axes_object_or_matrix.ravel().tolist()
    else:
        axes_arg = axes_object_or_matrix

    colour_bar_object = pyplot.colorbar(
        ax=axes_arg, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_arg,
        shrink=fraction_of_axis_length
    )

    colour_bar_object.ax.tick_params(labelsize=font_size)

    if orientation_string == 'horizontal':
        colour_bar_object.ax.set_xticklabels(
            colour_bar_object.ax.get_xticklabels(), rotation=90
        )

    return colour_bar_object


def plot_linear_colour_bar(
        axes_object_or_matrix, data_matrix, colour_map_object, min_value,
        max_value, orientation_string=DEFAULT_CBAR_ORIENTATION_STRING,
        padding=None, extend_min=True, extend_max=True,
        fraction_of_axis_length=1., font_size=FONT_SIZE):
    """Plots colour bar with linear scale.

    :param axes_object_or_matrix: See doc for `plot_colour_bar`.
    :param data_matrix: Same.
    :param colour_map_object: Same.
    :param min_value: Minimum value in colour bar.
    :param max_value: Max value in colour bar.
    :param orientation_string: See doc for `plot_colour_bar`.
    :param padding: Same.
    :param extend_min: Same.
    :param extend_max: Same.
    :param fraction_of_axis_length: Same.
    :param font_size: Same.
    :return: colour_bar_object: Same.
    """

    error_checking.assert_is_greater(max_value, min_value)
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_value, vmax=max_value, clip=False
    )

    return plot_colour_bar(
        axes_object_or_matrix=axes_object_or_matrix, data_matrix=data_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=orientation_string, padding=padding,
        extend_min=extend_min, extend_max=extend_max,
        fraction_of_axis_length=fraction_of_axis_length, font_size=font_size
    )


def label_axes(
        axes_object, label_string, font_size=50, font_colour=numpy.full(3, 0.),
        x_coord_normalized=0., y_coord_normalized=1.):
    """Adds text label to axes.

    :param axes_object: Axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param label_string: Label.
    :param font_size: Font size.
    :param font_colour: Font colour.
    :param x_coord_normalized: Normalized x-coordinate (from 0...1, where 1 is
        the right side).
    :param y_coord_normalized: Normalized y-coordinate (from 0...1, where 1 is
        the top).
    """

    error_checking.assert_is_string(label_string)
    # error_checking.assert_is_geq(x_coord_normalized, 0.)
    # error_checking.assert_is_leq(x_coord_normalized, 1.)
    # error_checking.assert_is_geq(y_coord_normalized, 0.)
    # error_checking.assert_is_leq(y_coord_normalized, 1.)

    axes_object.text(
        x_coord_normalized, y_coord_normalized, label_string,
        fontsize=font_size, color=font_colour,
        horizontalalignment='right', verticalalignment='bottom',
        transform=axes_object.transAxes
    )


def create_paneled_figure(
        num_rows, num_columns, figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        horizontal_spacing=0.075, vertical_spacing=0., shared_x_axis=False,
        shared_y_axis=False, keep_aspect_ratio=True):
    """Creates paneled figure.

    This method only initializes the panels.  It does not plot anything.

    J = number of panel rows
    K = number of panel columns

    :param num_rows: J in the above discussion.
    :param num_columns: K in the above discussion.
    :param figure_width_inches: Width of the entire figure (including all
        panels).
    :param figure_height_inches: Height of the entire figure (including all
        panels).
    :param horizontal_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel columns.
    :param vertical_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel rows.
    :param shared_x_axis: Boolean flag.  If True, all panels will share the same
        x-axis.
    :param shared_y_axis: Boolean flag.  If True, all panels will share the same
        y-axis.
    :param keep_aspect_ratio: Boolean flag.  If True, the aspect ratio of each
        panel will be preserved (reflect the aspect ratio of the data plotted
        therein).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: J-by-K numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    error_checking.assert_is_geq(horizontal_spacing, 0.)
    error_checking.assert_is_less_than(horizontal_spacing, 1.)
    error_checking.assert_is_geq(vertical_spacing, 0.)
    error_checking.assert_is_less_than(vertical_spacing, 1.)
    error_checking.assert_is_boolean(shared_x_axis)
    error_checking.assert_is_boolean(shared_y_axis)
    error_checking.assert_is_boolean(keep_aspect_ratio)

    figure_object, axes_object_matrix = pyplot.subplots(
        num_rows, num_columns, sharex=shared_x_axis, sharey=shared_y_axis,
        figsize=(figure_width_inches, figure_height_inches)
    )

    if num_rows == num_columns == 1:
        axes_object_matrix = numpy.full(
            (1, 1), axes_object_matrix, dtype=object
        )

    if num_rows == 1 or num_columns == 1:
        axes_object_matrix = numpy.reshape(
            axes_object_matrix, (num_rows, num_columns)
        )

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95,
        hspace=horizontal_spacing, wspace=vertical_spacing
    )

    if not keep_aspect_ratio:
        return figure_object, axes_object_matrix

    for i in range(num_rows):
        for j in range(num_columns):
            axes_object_matrix[i][j].set(aspect='equal')

    return figure_object, axes_object_matrix
