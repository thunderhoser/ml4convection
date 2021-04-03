"""Plotting methods for model evaluation."""

import os
import sys
import numpy
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import gg_plotting_utils

POLYGON_OPACITY = 0.5
CSI_LEVELS = numpy.linspace(0, 1, num=11, dtype=float)
PEIRCE_SCORE_LEVELS = numpy.linspace(0, 1, num=11, dtype=float)

ROC_CURVE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
PERF_DIAGRAM_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

FREQ_BIAS_COLOUR = numpy.full(3, 152. / 255)
FREQ_BIAS_WIDTH = 2.
FREQ_BIAS_STRING_FORMAT = '%.2f'
FREQ_BIAS_PADDING = 10
FREQ_BIAS_LEVELS = numpy.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5])

RELIABILITY_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

CLIMO_LINE_COLOUR = numpy.full(3, 152. / 255)
CLIMO_LINE_WIDTH = 2.

ZERO_SKILL_LINE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
ZERO_SKILL_LINE_WIDTH = 2.
POSITIVE_SKILL_AREA_OPACITY = 0.2

HISTOGRAM_FACE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 2.
HISTOGRAM_FONT_SIZE = 30

FONT_SIZE = 40
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _get_positive_skill_area(mean_value_in_training, min_value_in_plot,
                             max_value_in_plot):
    """Returns positive-skill area (where BSS > 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords_left: length-5 numpy array of x-coordinates for left part
        of positive-skill area.
    :return: y_coords_left: Same but for y-coordinates.
    :return: x_coords_right: length-5 numpy array of x-coordinates for right
        part of positive-skill area.
    :return: y_coords_right: Same but for y-coordinates.
    """

    x_coords_left = numpy.array([
        min_value_in_plot, mean_value_in_training, mean_value_in_training,
        min_value_in_plot, min_value_in_plot
    ])
    y_coords_left = numpy.array([
        min_value_in_plot, min_value_in_plot, mean_value_in_training,
        (min_value_in_plot + mean_value_in_training) / 2, min_value_in_plot
    ])

    x_coords_right = numpy.array([
        mean_value_in_training, max_value_in_plot, max_value_in_plot,
        mean_value_in_training, mean_value_in_training
    ])
    y_coords_right = numpy.array([
        mean_value_in_training,
        (max_value_in_plot + mean_value_in_training) / 2,
        max_value_in_plot, max_value_in_plot, mean_value_in_training
    ])

    return x_coords_left, y_coords_left, x_coords_right, y_coords_right


def _get_zero_skill_line(mean_value_in_training, min_value_in_plot,
                         max_value_in_plot):
    """Returns zero-skill line (where BSS = 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords: length-2 numpy array of x-coordinates.
    :return: y_coords: Same but for y-coordinates.
    """

    x_coords = numpy.array([min_value_in_plot, max_value_in_plot], dtype=float)
    y_coords = 0.5 * (mean_value_in_training + x_coords)

    return x_coords, y_coords


def _plot_attr_diagram_background(
        axes_object, mean_value_in_training, min_value_in_plot,
        max_value_in_plot):
    """Plots background (reference lines and polygons) of attributes diagram.

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    """

    x_coords_left, y_coords_left, x_coords_right, y_coords_right = (
        _get_positive_skill_area(
            mean_value_in_training=mean_value_in_training,
            min_value_in_plot=min_value_in_plot,
            max_value_in_plot=max_value_in_plot
        )
    )

    skill_area_colour = matplotlib.colors.to_rgba(
        ZERO_SKILL_LINE_COLOUR, POSITIVE_SKILL_AREA_OPACITY
    )

    left_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_left, y_coords_left
    )))
    left_patch_object = matplotlib.patches.Polygon(
        left_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(left_patch_object)

    right_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_right, y_coords_right
    )))
    right_patch_object = matplotlib.patches.Polygon(
        right_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(right_patch_object)

    no_skill_x_coords, no_skill_y_coords = _get_zero_skill_line(
        mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_in_plot,
        max_value_in_plot=max_value_in_plot
    )

    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords, color=ZERO_SKILL_LINE_COLOUR,
        linestyle='solid', linewidth=ZERO_SKILL_LINE_WIDTH
    )

    climo_x_coords = numpy.full(2, mean_value_in_training)
    climo_y_coords = numpy.array([min_value_in_plot, max_value_in_plot])
    axes_object.plot(
        climo_x_coords, climo_y_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )

    axes_object.plot(
        climo_y_coords, climo_x_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )


def _plot_inset_histogram(
        figure_object, bin_centers, bin_counts, has_predictions,
        bar_colour=HISTOGRAM_FACE_COLOUR):
    """Plots histogram as inset in attributes diagram.

    B = number of bins

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_counts: length-B numpy array with number of examples in each bin.
        These values will be plotted on the y-axis.
    :param has_predictions: Boolean flag.  If True, histogram will contain
        prediction frequencies.  If False, will contain observation frequencies.
    :param bar_colour: Bar colour (in any format accepted by matplotlib).
    """

    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    if has_predictions:
        inset_axes_object = figure_object.add_axes([0.2, 0.55, 0.3, 0.3])
    else:
        inset_axes_object = figure_object.add_axes([0.575, 0.2, 0.3, 0.3])

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
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)

    inset_axes_object.set_title(
        'Prediction frequency' if has_predictions else 'Observation frequency',
        fontsize=HISTOGRAM_FONT_SIZE
    )


def _get_pofd_pod_grid(pofd_spacing=0.01, pod_spacing=0.01):
    """Creates grid in POFD-POD space.

    POFD = probability of false detection
    POD = probability of detection

    M = number of rows (unique POD values) in grid
    N = number of columns (unique POFD values) in grid

    :param pofd_spacing: Spacing between grid cells in adjacent columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: pofd_matrix: M-by-N numpy array of POFD values.
    :return: pod_matrix: M-by-N numpy array of POD values.
    """

    num_pofd_values = int(numpy.ceil(1. / pofd_spacing))
    num_pod_values = int(numpy.ceil(1. / pod_spacing))

    unique_pofd_values = numpy.linspace(
        0, 1, num=num_pofd_values + 1, dtype=float
    )
    unique_pofd_values = unique_pofd_values[:-1] + pofd_spacing / 2

    unique_pod_values = numpy.linspace(
        0, 1, num=num_pod_values + 1, dtype=float
    )
    unique_pod_values = unique_pod_values[:-1] + pod_spacing / 2

    return numpy.meshgrid(unique_pofd_values, unique_pod_values[::-1])


def _get_peirce_colour_scheme():
    """Returns colour scheme for Peirce score.

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.get_cmap('Blues')
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        PEIRCE_SCORE_LEVELS
    ))
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _get_sr_pod_grid(success_ratio_spacing=0.01, pod_spacing=0.01):
    """Creates grid in SR-POD space

    SR = success ratio
    POD = probability of detection

    M = number of rows (unique POD values) in grid
    N = number of columns (unique success ratios) in grid

    :param success_ratio_spacing: Spacing between adjacent success ratios
        (x-values) in grid.
    :param pod_spacing: Spacing between adjacent POD values (y-values) in grid.
    :return: success_ratio_matrix: M-by-N numpy array of success ratios.
        Success ratio increases while traveling right along a row.
    :return: pod_matrix: M-by-N numpy array of POD values.  POD increases while
        traveling up a column.
    """

    num_success_ratios = int(numpy.ceil(1. / success_ratio_spacing))
    num_pod_values = int(numpy.ceil(1. / pod_spacing))

    unique_success_ratios = numpy.linspace(
        0, 1, num=num_success_ratios + 1, dtype=float
    )
    unique_success_ratios = (
        unique_success_ratios[:-1] + success_ratio_spacing / 2
    )

    unique_pod_values = numpy.linspace(
        0, 1, num=num_pod_values + 1, dtype=float
    )
    unique_pod_values = unique_pod_values[:-1] + pod_spacing / 2

    return numpy.meshgrid(unique_success_ratios, unique_pod_values[::-1])


def _csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.

    POD = probability of detection

    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: csi_array: numpy array (same shape) of CSI values.
    """

    return (success_ratio_array ** -1 + pod_array ** -1 - 1.) ** -1


def _bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.

    POD = probability of detection

    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: frequency_bias_array: numpy array (same shape) of frequency biases.
    """

    return pod_array / success_ratio_array


def _get_csi_colour_scheme(use_grey_scheme):
    """Returns colour scheme for CSI (critical success index).

    :param use_grey_scheme: Boolean flag.  If True (False), will use grey (blue)
        colour scheme.
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.get_cmap(
        'gist_yarg' if use_grey_scheme else 'Blues'
    )
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, this_colour_map_object.N
    )

    if use_grey_scheme:
        rgba_matrix = this_colour_map_object(this_colour_norm_object(
            CSI_LEVELS * 0.9
        ))
    else:
        rgba_matrix = this_colour_map_object(this_colour_norm_object(
            CSI_LEVELS
        ))

    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def confidence_interval_to_polygon(
        x_value_matrix, y_value_matrix, confidence_level, same_order):
    """Turns confidence interval into polygon.

    P = number of points
    R = number of bootstrap replicates
    V = number of vertices in resulting polygon = 2 * P + 1

    :param x_value_matrix: R-by-P numpy array of x-values.
    :param y_value_matrix: R-by-P numpy array of y-values.
    :param confidence_level: Confidence level (in range 0...1).
    :param same_order: Boolean flag.  If True (False), minimum x-values will be
        matched with minimum (maximum) y-values.
    :return: polygon_coord_matrix: V-by-2 numpy array of coordinates
        (x-coordinates in first column, y-coords in second).
    """

    error_checking.assert_is_numpy_array(x_value_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(y_value_matrix, num_dimensions=2)

    expected_dim = numpy.array([
        x_value_matrix.shape[0], y_value_matrix.shape[1]
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        y_value_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_leq(confidence_level, 1.)
    error_checking.assert_is_boolean(same_order)

    min_percentile = 50 * (1. - confidence_level)
    max_percentile = 50 * (1. + confidence_level)

    x_values_bottom = numpy.nanpercentile(
        x_value_matrix, min_percentile, axis=0, interpolation='linear'
    )
    x_values_top = numpy.nanpercentile(
        x_value_matrix, max_percentile, axis=0, interpolation='linear'
    )
    y_values_bottom = numpy.nanpercentile(
        y_value_matrix, min_percentile, axis=0, interpolation='linear'
    )
    y_values_top = numpy.nanpercentile(
        y_value_matrix, max_percentile, axis=0, interpolation='linear'
    )

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(x_values_bottom), numpy.isnan(y_values_bottom)
    )))[0]

    if len(real_indices) == 0:
        return None

    x_values_bottom = x_values_bottom[real_indices]
    x_values_top = x_values_top[real_indices]
    y_values_bottom = y_values_bottom[real_indices]
    y_values_top = y_values_top[real_indices]

    x_vertices = numpy.concatenate((
        x_values_top, x_values_bottom[::-1], x_values_top[[0]]
    ))

    if same_order:
        y_vertices = numpy.concatenate((
            y_values_top, y_values_bottom[::-1], y_values_top[[0]]
        ))
    else:
        y_vertices = numpy.concatenate((
            y_values_bottom, y_values_top[::-1], y_values_bottom[[0]]
        ))

    return numpy.transpose(numpy.vstack((
        x_vertices, y_vertices
    )))


def plot_reliability_curve(
        axes_object, mean_prediction_matrix, mean_observation_matrix,
        min_value_to_plot, max_value_to_plot, confidence_level=0.95,
        line_colour=RELIABILITY_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH, plot_background=True):
    """Plots reliability curve.

    B = number of bins
    R = number of bootstrap replicates

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_prediction_matrix: R-by-B numpy array of mean predicted values.
    :param mean_observation_matrix: R-by-B numpy array of mean observed values.
    :param min_value_to_plot: See doc for `plot_attributes_diagram`.
    :param max_value_to_plot: Same.
    :param confidence_level: Same.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :param plot_background: Boolean flag.  If True, will plot background
        (reference line).
    :return: main_line_handle: Handle for main line (reliability curve).
    """

    # TODO(thunderhoser): Input-checking here applies to both regression and
    # classification.

    # Check input args.
    error_checking.assert_is_numpy_array(
        mean_prediction_matrix, num_dimensions=2
    )
    error_checking.assert_is_numpy_array(
        mean_observation_matrix,
        exact_dimensions=numpy.array(mean_prediction_matrix.shape, dtype=int)
    )

    error_checking.assert_is_geq(max_value_to_plot, min_value_to_plot)
    if max_value_to_plot == min_value_to_plot:
        max_value_to_plot = min_value_to_plot + 1.

    error_checking.assert_is_boolean(plot_background)

    if plot_background:
        perfect_x_coords = numpy.array([min_value_to_plot, max_value_to_plot])
        perfect_y_coords = numpy.array([min_value_to_plot, max_value_to_plot])

        axes_object.plot(
            perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
            linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
        )

    mean_predictions = numpy.nanmean(mean_prediction_matrix, axis=0)
    mean_observations = numpy.nanmean(mean_observation_matrix, axis=0)
    nan_flags = numpy.logical_or(
        numpy.isnan(mean_predictions), numpy.isnan(mean_observations)
    )

    if numpy.all(nan_flags):
        main_line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        main_line_handle = axes_object.plot(
            mean_predictions[real_indices], mean_observations[real_indices],
            color=line_colour, linestyle=line_style, linewidth=line_width
        )[0]

    num_bootstrap_reps = mean_prediction_matrix.shape[0]

    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = confidence_interval_to_polygon(
            x_value_matrix=mean_prediction_matrix,
            y_value_matrix=mean_observation_matrix,
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(line_colour, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(min_value_to_plot, max_value_to_plot)
    axes_object.set_ylim(min_value_to_plot, max_value_to_plot)

    return main_line_handle


def plot_roc_curve(
        axes_object, pod_matrix, pofd_matrix, confidence_level=0.95,
        line_colour=ROC_CURVE_COLOUR, plot_background=True):
    """Plots ROC (receiver operating characteristic) curve.

    T = number of probability thresholds
    R = number of bootstrap replicates

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_matrix: R-by-T numpy array of POD (probability of detection)
        values.
    :param pofd_matrix: R-by-T numpy array of POFD (probability of false
        detection) values.
    :param confidence_level: Will plot confidence interval at this level.
    :param line_colour: Line colour.
    :param plot_background: Boolean flag.  If True, will plot background
        (reference line and Peirce-score contours).
    :return: line_handle: Line handle for ROC curve.
    """

    error_checking.assert_is_numpy_array(pod_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(pod_matrix, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(pod_matrix, 1., allow_nan=True)

    error_checking.assert_is_numpy_array(
        pofd_matrix,
        exact_dimensions=numpy.array(pod_matrix.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(pofd_matrix, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(pofd_matrix, 1., allow_nan=True)

    error_checking.assert_is_boolean(plot_background)

    if plot_background:
        this_pofd_matrix, this_pod_matrix = _get_pofd_pod_grid()
        peirce_score_matrix = this_pod_matrix - this_pofd_matrix

        this_colour_map_object, this_colour_norm_object = (
            _get_peirce_colour_scheme()
        )

        pyplot.contourf(
            this_pofd_matrix, this_pod_matrix, peirce_score_matrix,
            PEIRCE_SCORE_LEVELS, cmap=this_colour_map_object,
            norm=this_colour_norm_object, vmin=0., vmax=1., axes=axes_object
        )

        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=peirce_score_matrix,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=False,
            font_size=FONT_SIZE
        )
        colour_bar_object.set_label('Peirce score (POD minus POFD)')

        random_x_coords = numpy.array([0, 1], dtype=float)
        random_y_coords = numpy.array([0, 1], dtype=float)
        axes_object.plot(
            random_x_coords, random_y_coords, color=REFERENCE_LINE_COLOUR,
            linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
        )

    pofd_by_threshold = numpy.nanmean(pofd_matrix, axis=0)
    pod_by_threshold = numpy.nanmean(pod_matrix, axis=0)
    nan_flags = numpy.logical_or(
        numpy.isnan(pofd_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if numpy.all(nan_flags):
        line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        line_handle = axes_object.plot(
            pofd_by_threshold[real_indices], pod_by_threshold[real_indices],
            color=line_colour, linestyle='solid', linewidth=DEFAULT_LINE_WIDTH
        )[0]

    num_bootstrap_reps = pod_matrix.shape[0]

    if num_bootstrap_reps > 1:
        polygon_coord_matrix = confidence_interval_to_polygon(
            x_value_matrix=pofd_matrix, y_value_matrix=pod_matrix,
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(line_colour, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    axes_object.set_xlabel('POFD (probability of false detection)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return line_handle


def plot_performance_diagram(
        axes_object, pod_matrix, success_ratio_matrix, confidence_level=0.95,
        line_colour=PERF_DIAGRAM_COLOUR, plot_background=True,
        plot_csi_in_grey=False):
    """Plots performance diagram.

    T = number of probability thresholds
    R = number of bootstrap replicates

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_matrix: R-by-T numpy array of POD (probability of detection)
        values.
    :param success_ratio_matrix: R-by-T numpy array of success ratios.
    :param confidence_level: Will plot confidence interval at this level.
    :param line_colour: Line colour.
    :param plot_background: Boolean flag.  If True, will plot background
        (frequency-bias and CSI contours).
    :param plot_csi_in_grey: Boolean flag.  If True (False), will plot CSI in
        grey (blue) colour scheme.
    :return: line_handle: Line handle for ROC curve.
    """

    error_checking.assert_is_numpy_array(pod_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(pod_matrix, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(pod_matrix, 1., allow_nan=True)

    error_checking.assert_is_numpy_array(
        success_ratio_matrix,
        exact_dimensions=numpy.array(pod_matrix.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        success_ratio_matrix, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        success_ratio_matrix, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(plot_background)
    error_checking.assert_is_boolean(plot_csi_in_grey)

    if plot_background:
        this_success_ratio_matrix, this_pod_matrix = _get_sr_pod_grid()
        csi_matrix = _csi_from_sr_and_pod(
            success_ratio_array=this_success_ratio_matrix,
            pod_array=this_pod_matrix
        )
        frequency_bias_matrix = _bias_from_sr_and_pod(
            success_ratio_array=this_success_ratio_matrix,
            pod_array=this_pod_matrix
        )

        this_colour_map_object, this_colour_norm_object = (
            _get_csi_colour_scheme(use_grey_scheme=plot_csi_in_grey)
        )
        pyplot.contourf(
            this_success_ratio_matrix, this_pod_matrix, csi_matrix, CSI_LEVELS,
            cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
            vmax=1., axes=axes_object
        )

        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=csi_matrix,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=False,
            font_size=FONT_SIZE
        )
        colour_bar_object.set_label('CSI (critical success index)')

        if plot_csi_in_grey:
            bias_colour_tuple = (0., 0., 0.)
        else:
            bias_colour_tuple = tuple(FREQ_BIAS_COLOUR.tolist())

        bias_colours_2d_tuple = ()
        for _ in range(len(FREQ_BIAS_LEVELS)):
            bias_colours_2d_tuple += (bias_colour_tuple,)

        bias_contour_object = pyplot.contour(
            this_success_ratio_matrix, this_pod_matrix, frequency_bias_matrix,
            FREQ_BIAS_LEVELS, colors=bias_colours_2d_tuple,
            linewidths=FREQ_BIAS_WIDTH, linestyles='dashed', axes=axes_object
        )
        axes_object.clabel(
            bias_contour_object, inline=True, inline_spacing=FREQ_BIAS_PADDING,
            fmt=FREQ_BIAS_STRING_FORMAT, fontsize=FONT_SIZE
        )

    success_ratio_by_threshold = numpy.nanmean(success_ratio_matrix, axis=0)
    pod_by_threshold = numpy.nanmean(pod_matrix, axis=0)
    nan_flags = numpy.logical_or(
        numpy.isnan(success_ratio_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if numpy.all(nan_flags):
        line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        line_handle = axes_object.plot(
            success_ratio_by_threshold[real_indices],
            pod_by_threshold[real_indices], color=line_colour,
            linestyle='solid', linewidth=DEFAULT_LINE_WIDTH
        )[0]

    num_bootstrap_reps = pod_matrix.shape[0]

    if num_bootstrap_reps > 1:
        polygon_coord_matrix = confidence_interval_to_polygon(
            x_value_matrix=success_ratio_matrix, y_value_matrix=pod_matrix,
            confidence_level=confidence_level, same_order=True
        )

        polygon_colour = matplotlib.colors.to_rgba(line_colour, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    axes_object.set_xlabel('Success ratio (1 - FAR)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return line_handle


def plot_attributes_diagram(
        figure_object, axes_object, mean_prediction_matrix,
        mean_observation_matrix, example_counts, mean_value_in_training,
        min_value_to_plot, max_value_to_plot, confidence_level=0.95,
        line_colour=RELIABILITY_LINE_COLOUR,
        line_style='solid', line_width=DEFAULT_LINE_WIDTH,
        inv_mean_observations=None, inv_example_counts=None):
    """Plots attributes diagram.

    If `inv_mean_observations is None` and `inv_example_counts is None`, this
    method will plot only the histogram of predicted values, not the histogram
    of observed values.

    B = number of bins
    R = number of bootstrap replicates

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_prediction_matrix: R-by-B numpy array of mean predicted values.
    :param mean_observation_matrix: R-by-B numpy array of mean observed values.
    :param example_counts: length-B numpy array with number of examples in each
        bin.
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_to_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_to_plot: Max value in plot (for both x- and y-axes).
        If None, will be determined automatically.
    :param confidence_level: Will plot confidence interval at this level.
    :param line_colour: See doc for `plot_reliability_curve`.
    :param line_width: Same.
    :param line_style: Same.
    :param inv_mean_observations: length-B numpy array of mean observed values
        for inverted reliability curve.
    :param inv_example_counts: length-B numpy array of example counts for
        inverted reliability curve.
    :return: main_line_handle: See doc for `plot_reliability_curve`.
    """

    # TODO(thunderhoser): Input-checking here applies to both regression and
    # classification.

    # Check input args.
    error_checking.assert_is_numpy_array(
        mean_prediction_matrix, num_dimensions=2
    )
    error_checking.assert_is_numpy_array(
        mean_observation_matrix,
        exact_dimensions=numpy.array(mean_prediction_matrix.shape, dtype=int)
    )

    num_bins = mean_prediction_matrix.shape[1]

    error_checking.assert_is_integer_numpy_array(example_counts)
    error_checking.assert_is_geq_numpy_array(example_counts, 0)
    error_checking.assert_is_numpy_array(
        example_counts,
        exact_dimensions=numpy.array([num_bins], dtype=int)
    )

    error_checking.assert_is_not_nan(mean_value_in_training)
    error_checking.assert_is_geq(max_value_to_plot, min_value_to_plot)
    if max_value_to_plot == min_value_to_plot:
        max_value_to_plot = min_value_to_plot + 1.

    plot_obs_histogram = not(
        inv_mean_observations is None and inv_example_counts is None
    )

    if plot_obs_histogram:
        error_checking.assert_is_numpy_array(
            inv_mean_observations,
            exact_dimensions=numpy.array([num_bins], dtype=int)
        )

        error_checking.assert_is_integer_numpy_array(inv_example_counts)
        error_checking.assert_is_geq_numpy_array(inv_example_counts, 0)
        error_checking.assert_is_numpy_array(
            inv_example_counts,
            exact_dimensions=numpy.array([num_bins], dtype=int)
        )

    # Do actual stuff.
    _plot_attr_diagram_background(
        axes_object=axes_object, mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_to_plot, max_value_in_plot=max_value_to_plot
    )

    _plot_inset_histogram(
        figure_object=figure_object,
        bin_centers=numpy.nanmean(mean_prediction_matrix, axis=0),
        bin_counts=example_counts, has_predictions=True, bar_colour=line_colour
    )

    if plot_obs_histogram:
        _plot_inset_histogram(
            figure_object=figure_object, bin_centers=inv_mean_observations,
            bin_counts=inv_example_counts, has_predictions=False,
            bar_colour=line_colour
        )

    return plot_reliability_curve(
        axes_object=axes_object, mean_prediction_matrix=mean_prediction_matrix,
        mean_observation_matrix=mean_observation_matrix,
        min_value_to_plot=min_value_to_plot,
        max_value_to_plot=max_value_to_plot,
        confidence_level=confidence_level,
        line_colour=line_colour, line_style=line_style, line_width=line_width
    )
