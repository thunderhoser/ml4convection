"""Plots scores on hyperparam grid for aggregated loss-function experiment.

"Aggregated loss-function experiment" = LF Experiments 4-6
"""

import os
import sys
import glob
import argparse
from PIL import Image
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import gg_model_evaluation as gg_model_eval
import gg_plotting_utils
import imagemagick_utils
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NAN = numpy.nan
MAX_MAX_RESOLUTION_DEG = 1e10

WAVELET_TRANSFORM_FLAGS = numpy.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0
], dtype=bool)

FOURIER_TRANSFORM_FLAGS = numpy.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
], dtype=bool)

MIN_RESOLUTIONS_DEG = numpy.array([
    0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0, 0, 0, 0, 0.05, 0.1, 0.2, 0.4,
    0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0, 0, 0, 0, 0.05, 0.1, 0.2, 0.4,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN
])

MAX_RESOLUTIONS_DEG = numpy.array([
    0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, numpy.inf, 0.05, 0.1, 0.2, 0.4, numpy.inf, numpy.inf, numpy.inf, numpy.inf,
    0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, numpy.inf, 0.05, 0.1, 0.2, 0.4, numpy.inf, numpy.inf, numpy.inf, numpy.inf,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN
])

NEIGH_HALF_WINDOW_SIZES_PX = numpy.array([
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
    0, 1, 2, 3, 4, 6, 8, 12
])

SUBEXPERIMENT_ENUMS = numpy.array([
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6
], dtype=int)

FILTER_NAMES = [
    r'FT 0-0.0125$^{\circ}$',
    r'FT 0.0125-0.025$^{\circ}$',
    r'FT 0.025-0.05$^{\circ}$',
    r'FT 0.05-0.1$^{\circ}$',
    r'FT 0.1-0.2$^{\circ}$',
    r'FT 0.2-0.4$^{\circ}$',
    r'FT 0.4-0.8$^{\circ}$',
    r'FT 0.8-$\infty^{\circ}$',
    r'FT 0-0.05$^{\circ}$',
    r'FT 0-0.1$^{\circ}$',
    r'FT 0-0.2$^{\circ}$',
    r'FT 0-0.4$^{\circ}$',
    r'FT 0.05-$\infty^{\circ}$',
    r'FT 0.1-$\infty^{\circ}$',
    r'FT 0.2-$\infty^{\circ}$',
    r'FT 0.4-$\infty^{\circ}$',
    r'WT 0-0.0125$^{\circ}$',
    r'WT 0.0125-0.025$^{\circ}$',
    r'WT 0.025-0.05$^{\circ}$',
    r'WT 0.05-0.1$^{\circ}$',
    r'WT 0.1-0.2$^{\circ}$',
    r'WT 0.2-0.4$^{\circ}$',
    r'WT 0.4-0.8$^{\circ}$',
    r'WT 0.8-$\infty^{\circ}$',
    r'WT 0-0.05$^{\circ}$',
    r'WT 0-0.1$^{\circ}$',
    r'WT 0-0.2$^{\circ}$',
    r'WT 0-0.4$^{\circ}$',
    r'WT 0.05-$\infty^{\circ}$',
    r'WT 0.1-$\infty^{\circ}$',
    r'WT 0.2-$\infty^{\circ}$',
    r'WT 0.4-$\infty^{\circ}$',
    '1 x 1 neigh',
    '3 x 3 neigh',
    '5 x 5 neigh',
    '7 x 7 neigh',
    '9 x 9 neigh',
    '13 x 13 neigh',
    '17 x 17 neigh',
    '25 x 25 neigh'
]

LOSS_FUNCTION_NAMES = [
    'brier', 'fss', 'iou', 'all-class-iou', 'dice', 'csi', 'heidke',
    'gerrity', 'peirce'
]
LOSS_FUNCTION_NAMES_FANCY = [
    'Brier', 'FSS', r'IOU$_{pos}$', r'IOU$_{all}$', 'Dice', 'CSI', 'Heidke',
    'Gerrity', 'Peirce'
]

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.3
MARKER_COLOUR = numpy.full(3, 0.)

DEFAULT_FONT_SIZE = 20
COLOUR_BAR_FONT_SIZE = 25

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
BSS_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

ALL_EXPERIMENT_DIR_ARG_NAME = 'input_all_experiment_dir_name'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_px'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ALL_EXPERIMENT_DIR_HELP_STRING = (
    'Name of directory containing results for all relevant subexperiments '
    '(Experiments 4-6).'
)
MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance for neighbourhood evaluation (pixels).  Will plot scores'
    ' at this matching distance.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ALL_EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=ALL_EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=True,
    help=MATCHING_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_scores_one_model(
        subexperiment_dir_name, loss_function_name, wavelet_transform_flag,
        fourier_transform_flag, min_resolution_deg, max_resolution_deg,
        neigh_half_window_size_px, matching_distance_px):
    """Reads scores for one model.

    :param subexperiment_dir_name: Name of directory with all models for
        subexperiment (Experiment 4, 5, or 6).
    :param loss_function_name: Name of loss function.
    :param wavelet_transform_flag: Boolean flag, indicating whether or not
        wavelet transform was used to filter data before loss function.
    :param fourier_transform_flag: Boolean flag, indicating whether or not
        Fourier transform was used to filter data before loss function.
    :param min_resolution_deg: Minimum resolution permitted through
        wavelet/Fourier filter.
    :param max_resolution_deg: Max resolution permitted through wavelet/Fourier
        filter.
    :param neigh_half_window_size_px: Size of half-window (pixels) for
        neighbourhood-based loss function.  If
        `wavelet_transform_flag == fourier_transform_flag == False`, then loss
        function is neighbourhood-based.
    :param matching_distance_px: See documentation at top of file.
    :return: advanced_score_table_xarray: xarray table returned by
        `evaluation.read_advanced_score_file`.
    """

    if wavelet_transform_flag or fourier_transform_flag:
        max_resolution_deg = min([
            max_resolution_deg, MAX_MAX_RESOLUTION_DEG
        ])

        this_string = (
            '{0:s}_wavelets{1:d}_min-resolution-deg={2:.4f}_'
            'max-resolution-deg={3:.4f}'
        ).format(
            loss_function_name, int(wavelet_transform_flag),
            min_resolution_deg, max_resolution_deg
        )
    else:
        this_string = '{0:s}-neigh{1:d}'.format(
            loss_function_name, int(numpy.round(neigh_half_window_size_px))
        )

    score_file_pattern = (
        '{0:s}/{1:s}/model*/validation_best_validation_loss/full_grids/'
        'evaluation/matching_distance_px={2:.6f}/advanced_scores_gridded=0.p'
    ).format(
        subexperiment_dir_name, this_string, matching_distance_px
    )

    all_file_names = glob.glob(score_file_pattern)
    if len(all_file_names) == 0:
        return None

    assert len(all_file_names) == 1
    score_file_name = all_file_names[0]

    print('Reading data from: "{0:s}"...'.format(score_file_name))
    return evaluation.read_advanced_score_file(score_file_name)


def _plot_grid_one_score(score_matrix, min_colour_value, max_colour_value,
                         colour_map_object):
    """Plots grid for one score.

    L = number of loss functions
    F = number of filters

    :param score_matrix: L-by-F numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    max_colour_value = max([
        max_colour_value, min_colour_value + 1e-6
    ])

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(x_tick_values, FILTER_NAMES, rotation=90.)
    pyplot.yticks(y_tick_values, LOSS_FUNCTION_NAMES_FANCY)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    L = number of loss functions
    F = number of filters

    :param score_matrix: L-by-F numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix) + 0.
    scores_1d[numpy.isnan(scores_1d)] = -numpy.inf
    sort_indices_1d = numpy.argsort(-scores_1d)

    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        j = j_sort_indices[k]

        print((
            '{0:d}th-highest {1:s} = {2:.4g} ... base loss function = {3:s} '
            '... filter = {4:s}'
        ).format(
            k + 1, score_name, score_matrix[i, j],
            LOSS_FUNCTION_NAMES[i], FILTER_NAMES[j]
        ))


def _add_markers(figure_object, axes_object, best_marker_indices):
    """Adds markers to figure.

    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param best_marker_indices: length-2 numpy array of array indices for best
        model.
    """

    figure_width_px = figure_object.get_size_inches()[0] * figure_object.dpi
    marker_size_px = figure_width_px * (
        BEST_MARKER_SIZE_GRID_CELLS / len(FILTER_NAMES)
    )

    axes_object.plot(
        best_marker_indices[1], best_marker_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )


def _add_colour_bar(
        figure_file_name, colour_map_object, min_colour_value, max_colour_value,
        temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param temporary_dir_name: Name of temporary output directory.
    """

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )
    dummy_values = numpy.array([min_colour_value, max_colour_value])

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=1.25, font_size=COLOUR_BAR_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_file_name = '{0:s}/extra_colour_bar.jpg'.format(temporary_dir_name)
    print('Saving colour bar to: "{0:s}"...'.format(extra_file_name))

    extra_figure_object.savefig(
        extra_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, extra_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(extra_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(all_experiment_dir_name, matching_distance_px, output_dir_name):
    """Plots scores on hyperparam grid for aggregated loss-function experiment.

    This is effectively the main method.

    :param all_experiment_dir_name: See documentation at top of file.
    :param matching_distance_px: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_loss_functions = len(LOSS_FUNCTION_NAMES)
    num_filters = len(MIN_RESOLUTIONS_DEG)
    dimensions = (num_loss_functions, num_filters)

    aupd_matrix = numpy.full(dimensions, numpy.nan)
    max_csi_matrix = numpy.full(dimensions, numpy.nan)
    fss_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(num_loss_functions):
        for j in range(num_filters):
            t = _read_scores_one_model(
                subexperiment_dir_name='{0:s}/lf_experiment{1:02d}'.format(
                    all_experiment_dir_name, SUBEXPERIMENT_ENUMS[j]
                ),
                loss_function_name=LOSS_FUNCTION_NAMES[i],
                wavelet_transform_flag=WAVELET_TRANSFORM_FLAGS[j],
                fourier_transform_flag=FOURIER_TRANSFORM_FLAGS[j],
                min_resolution_deg=MIN_RESOLUTIONS_DEG[j],
                max_resolution_deg=MAX_RESOLUTIONS_DEG[j],
                neigh_half_window_size_px=NEIGH_HALF_WINDOW_SIZES_PX[j],
                matching_distance_px=matching_distance_px
            )

            if t is None:
                continue

            these_pod = numpy.nanmean(t[evaluation.POD_KEY].values, axis=0)
            these_success_ratios = numpy.nanmean(
                t[evaluation.SUCCESS_RATIO_KEY].values, axis=0
            )
            bad_flags = numpy.logical_or(
                numpy.isnan(these_pod), numpy.isnan(these_success_ratios)
            )
            good_indices = numpy.where(numpy.invert(bad_flags))[0]

            if len(good_indices) >= 2:
                aupd_matrix[i, j] = gg_model_eval.get_area_under_perf_diagram(
                    pod_by_threshold=these_pod,
                    success_ratio_by_threshold=these_success_ratios
                )

            max_csi_matrix[i, j] = numpy.nanmean(
                numpy.nanmax(t[evaluation.CSI_KEY].values, axis=1)
            )
            fss_matrix[i, j] = numpy.nanmean(t[evaluation.FSS_KEY].values)
            bss_matrix[i, j] = numpy.nanmean(
                t[evaluation.BRIER_SKILL_SCORE_KEY].values
            )

    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=aupd_matrix, score_name='AUPD')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=max_csi_matrix, score_name='max CSI')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=fss_matrix, score_name='FSS')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=bss_matrix, score_name='BSS')
    print(SEPARATOR_STRING)

    this_index = numpy.nanargmax(numpy.ravel(aupd_matrix))
    max_aupd_indices = numpy.unravel_index(this_index, aupd_matrix.shape)

    this_index = numpy.nanargmax(numpy.ravel(max_csi_matrix))
    max_csi_indices = numpy.unravel_index(this_index, max_csi_matrix.shape)

    this_index = numpy.nanargmax(numpy.ravel(fss_matrix))
    max_fss_indices = numpy.unravel_index(this_index, fss_matrix.shape)

    this_index = numpy.nanargmax(numpy.ravel(bss_matrix))
    max_bss_indices = numpy.unravel_index(this_index, bss_matrix.shape)

    # Plot AUPD.
    figure_object, axes_object = _plot_grid_one_score(
        score_matrix=aupd_matrix,
        min_colour_value=numpy.nanpercentile(aupd_matrix, 0),
        max_colour_value=numpy.nanpercentile(aupd_matrix, 100),
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_aupd_indices
    )
    axes_object.set_ylabel('Score for model''s loss function')
    axes_object.set_xlabel('Filter for model''s loss function')
    axes_object.set_title('Area under performance diagram')

    figure_file_name = '{0:s}/aupd_grid.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=numpy.nanpercentile(aupd_matrix, 0),
        max_colour_value=numpy.nanpercentile(aupd_matrix, 100),
        temporary_dir_name=output_dir_name
    )

    # Plot max CSI.
    figure_object, axes_object = _plot_grid_one_score(
        score_matrix=max_csi_matrix,
        min_colour_value=numpy.nanpercentile(max_csi_matrix, 0),
        max_colour_value=numpy.nanpercentile(max_csi_matrix, 100),
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_csi_indices
    )
    axes_object.set_ylabel('Score for model''s loss function')
    axes_object.set_xlabel('Filter for model''s loss function')
    axes_object.set_title('Critical success index')

    figure_file_name = '{0:s}/csi_grid.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=numpy.nanpercentile(max_csi_matrix, 0),
        max_colour_value=numpy.nanpercentile(max_csi_matrix, 100),
        temporary_dir_name=output_dir_name
    )

    # Plot FSS.
    figure_object, axes_object = _plot_grid_one_score(
        score_matrix=fss_matrix,
        min_colour_value=numpy.nanpercentile(fss_matrix, 0),
        max_colour_value=numpy.nanpercentile(fss_matrix, 100),
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_fss_indices
    )
    axes_object.set_ylabel('Score for model''s loss function')
    axes_object.set_xlabel('Filter for model''s loss function')
    axes_object.set_title('Fractions skill score')

    figure_file_name = '{0:s}/fss_grid.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=numpy.nanpercentile(fss_matrix, 0),
        max_colour_value=numpy.nanpercentile(fss_matrix, 100),
        temporary_dir_name=output_dir_name
    )

    # Plot BSS.
    this_max_value = numpy.nanpercentile(numpy.absolute(bss_matrix), 99.)
    this_max_value = min([this_max_value, 1.])
    this_min_value = -1 * this_max_value

    figure_object, axes_object = _plot_grid_one_score(
        score_matrix=bss_matrix,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        colour_map_object=BSS_COLOUR_MAP_OBJECT
    )
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_bss_indices
    )
    axes_object.set_ylabel('Score for model''s loss function')
    axes_object.set_xlabel('Filter for model''s loss function')
    axes_object.set_title('Brier skill score')

    figure_file_name = '{0:s}/bss_grid.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=BSS_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        temporary_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        all_experiment_dir_name=getattr(
            INPUT_ARG_OBJECT, ALL_EXPERIMENT_DIR_ARG_NAME
        ),
        matching_distance_px=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
