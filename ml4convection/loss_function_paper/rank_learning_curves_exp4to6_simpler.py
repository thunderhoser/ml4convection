"""Ranks learning-curve scores for aggregated experiment."""

import os
import glob
import argparse
from PIL import Image
import numpy
from scipy.stats import rankdata
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4convection.utils import learning_curves

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOLERANCE = 1e-6
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
    'ft-0.0000-0.0125',
    'ft-0.0125-0.0250',
    'ft-0.0250-0.0500',
    'ft-0.0500-0.1000',
    'ft-0.1000-0.2000',
    'ft-0.2000-0.4000',
    'ft-0.4000-0.8000',
    'ft-0.8000-inf',
    'ft-0.0000-0.0500',
    'ft-0.0000-0.1000',
    'ft-0.0000-0.2000',
    'ft-0.0000-0.4000',
    'ft-0.0500-inf',
    'ft-0.1000-inf',
    'ft-0.2000-inf',
    'ft-0.4000-inf',
    'wt-0.0000-0.0125',
    'wt-0.0125-0.0250',
    'wt-0.0250-0.0500',
    'wt-0.0500-0.1000',
    'wt-0.1000-0.2000',
    'wt-0.2000-0.4000',
    'wt-0.4000-0.8000',
    'wt-0.8000-inf',
    'wt-0.0000-0.0500',
    'wt-0.0000-0.1000',
    'wt-0.0000-0.2000',
    'wt-0.0000-0.4000',
    'wt-0.0500-inf',
    'wt-0.1000-inf',
    'wt-0.2000-inf',
    'wt-0.4000-inf',
    'neigh0',
    'neigh1',
    'neigh2',
    'neigh3',
    'neigh4',
    'neigh6',
    'neigh8',
    'neigh12'
]

FILTER_NAMES_FANCY = [
    r'FD 0-0.025$^{\circ}$',
    r'FD 0.025-0.05$^{\circ}$',
    r'FD 0.05-0.1$^{\circ}$',
    r'FD 0.1-0.2$^{\circ}$',
    r'FD 0.2-0.4$^{\circ}$',
    r'FD 0.4-0.8$^{\circ}$',
    r'FD 0.8-1.6$^{\circ}$',
    r'FD 1.6-$\infty^{\circ}$',
    r'FD 0-0.1$^{\circ}$',
    r'FD 0-0.2$^{\circ}$',
    r'FD 0-0.4$^{\circ}$',
    r'FD 0-0.8$^{\circ}$',
    r'FD 0.1-$\infty^{\circ}$',
    r'FD 0.2-$\infty^{\circ}$',
    r'FD 0.4-$\infty^{\circ}$',
    r'FD 0.8-$\infty^{\circ}$',
    r'WD 0-0.025$^{\circ}$',
    r'WD 0.025-0.05$^{\circ}$',
    r'WD 0.05-0.1$^{\circ}$',
    r'WD 0.1-0.2$^{\circ}$',
    r'WD 0.2-0.4$^{\circ}$',
    r'WD 0.4-0.8$^{\circ}$',
    r'WD 0.8-1.6$^{\circ}$',
    r'WD 1.6-$\infty^{\circ}$',
    r'WD 0-0.1$^{\circ}$',
    r'WD 0-0.2$^{\circ}$',
    r'WD 0-0.4$^{\circ}$',
    r'WD 0-0.8$^{\circ}$',
    r'WD 0.1-$\infty^{\circ}$',
    r'WD 0.2-$\infty^{\circ}$',
    r'WD 0.4-$\infty^{\circ}$',
    r'WD 0.8-$\infty^{\circ}$',
    '1-by-1 neigh',
    '3-by-3 neigh',
    '5-by-5 neigh',
    '7-by-7 neigh',
    '9-by-9 neigh',
    '13-by-13 neigh',
    '17-by-17 neigh',
    '25-by-25 neigh'
]

REFERENCE_LINE_COLOUR = numpy.full(3, 0.)
REFERENCE_LINE_WIDTH = 5
REFERENCE_LINE_X_COORDS = numpy.array([15.5, 31.5])

LOSS_FUNCTION_NAMES = [
    'fss', 'iou', 'csi', 'heidke', 'gerrity', 'peirce', 'brier', 'dice'
]
LOSS_FUNCTION_NAMES_FANCY = [
    'FSS', 'IOU', 'CSI', 'Heidke score', 'Gerrity score', 'Peirce score',
    'Brier score', 'Dice coeff'
]
NEGATIVELY_ORIENTED_FLAGS = numpy.array(
    [1, 0, 0, 0, 0, 0, 0, 0], dtype=bool
)
MODEL_NAME_INDICES_TO_PLOT = numpy.array([0, 6], dtype=int)
EVAL_FILTER_INDICES_TO_PLOT = numpy.array(
    [9, 13, 25, 29, 32, 36, 39], dtype=int
)

LOSS_FUNCTION_KEYS_NEIGH = [
    learning_curves.NEIGH_BRIER_SCORE_KEY, learning_curves.NEIGH_FSS_KEY,
    learning_curves.NEIGH_IOU_KEY, learning_curves.NEIGH_DICE_COEFF_KEY,
    learning_curves.NEIGH_CSI_KEY, None, None, None
]
LOSS_FUNCTION_KEYS_FOURIER = [
    learning_curves.FOURIER_BRIER_SCORE_KEY, learning_curves.FOURIER_FSS_KEY,
    learning_curves.FOURIER_IOU_KEY, learning_curves.FOURIER_DICE_COEFF_KEY,
    learning_curves.FOURIER_CSI_KEY,
    learning_curves.FOURIER_HEIDKE_SCORE_KEY,
    learning_curves.FOURIER_GERRITY_SCORE_KEY,
    learning_curves.FOURIER_PEIRCE_SCORE_KEY
]
LOSS_FUNCTION_KEYS_WAVELET = [
    learning_curves.WAVELET_BRIER_SCORE_KEY, learning_curves.WAVELET_FSS_KEY,
    learning_curves.WAVELET_IOU_KEY, learning_curves.WAVELET_DICE_COEFF_KEY,
    learning_curves.WAVELET_CSI_KEY,
    learning_curves.WAVELET_HEIDKE_SCORE_KEY,
    learning_curves.WAVELET_GERRITY_SCORE_KEY,
    learning_curves.WAVELET_PEIRCE_SCORE_KEY
]

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.3
MARKER_COLOUR = numpy.full(3, 0.)

COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
DEFAULT_FONT_SIZE = 20
COLOUR_BAR_FONT_SIZE = 25

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

ALL_EXPERIMENT_DIR_ARG_NAME = 'input_all_experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ALL_EXPERIMENT_DIR_HELP_STRING = (
    'Name of directory containing results for all relevant subexperiments '
    '(Experiments 4-6).'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_scores_one_model(
        subexperiment_dir_name, loss_function_name, wavelet_transform_flag,
        fourier_transform_flag, min_resolution_deg, max_resolution_deg,
        neigh_half_window_size_px):
    """Reads scores for one model.

    L = number of loss functions
    F = number of filters

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
    :return: score_matrix: L-by-F numpy array of scores.
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
        '{0:s}/{1:s}/model*/validation_best_validation_loss/partial_grids/'
        'learning_curves/advanced_scores.nc'
    ).format(
        subexperiment_dir_name, this_string
    )
    score_file_names = glob.glob(score_file_pattern)

    if len(score_file_names) == 0:
        return None

    model_subdir_names = [f.split('/')[-5] for f in score_file_names]
    validation_loss_strings = [
        d.split('_')[-1] for d in model_subdir_names
    ]

    for this_string in validation_loss_strings:
        assert this_string.startswith('val-loss=')

    validation_losses = numpy.array([
        float(s.replace('val-loss=', '')) for s in validation_loss_strings
    ])
    min_index = numpy.nanargmin(validation_losses)
    score_file_name = score_file_names[min_index]

    print('Reading data from: "{0:s}"...'.format(score_file_name))
    advanced_score_table_xarray = learning_curves.read_scores(score_file_name)
    a = advanced_score_table_xarray

    table_neigh_distances_px = (
        a.coords[learning_curves.NEIGH_DISTANCE_DIM].values
    )
    table_min_resolutions_deg = (
        a.coords[learning_curves.MIN_RESOLUTION_DIM].values
    )
    table_max_resolutions_deg = (
        a.coords[learning_curves.MAX_RESOLUTION_DIM].values
    )
    table_max_resolutions_deg[numpy.isinf(table_max_resolutions_deg)] = (
        MAX_MAX_RESOLUTION_DEG
    )

    num_loss_functions = len(LOSS_FUNCTION_NAMES)
    num_filters = len(FILTER_NAMES_FANCY)
    score_matrix = numpy.full((num_loss_functions, num_filters), numpy.nan)

    for i in range(num_loss_functions):
        for j in range(num_filters):
            if WAVELET_TRANSFORM_FLAGS[j]:
                this_key = LOSS_FUNCTION_KEYS_WAVELET[i]
            elif FOURIER_TRANSFORM_FLAGS[j]:
                this_key = LOSS_FUNCTION_KEYS_FOURIER[i]
            else:
                this_key = LOSS_FUNCTION_KEYS_NEIGH[i]

            if this_key is None:
                continue

            if WAVELET_TRANSFORM_FLAGS[j] or FOURIER_TRANSFORM_FLAGS[j]:
                these_diffs = numpy.absolute(
                    MIN_RESOLUTIONS_DEG[j] - table_min_resolutions_deg
                )

                if numpy.isinf(MAX_RESOLUTIONS_DEG[j]):
                    these_diffs += numpy.absolute(
                        MAX_MAX_RESOLUTION_DEG - table_max_resolutions_deg
                    )
                else:
                    these_diffs += numpy.absolute(
                        MAX_RESOLUTIONS_DEG[j] - table_max_resolutions_deg
                    )
            else:
                these_diffs = numpy.absolute(
                    NEIGH_HALF_WINDOW_SIZES_PX[j] - table_neigh_distances_px
                )

            scale_index = numpy.where(these_diffs <= TOLERANCE)[0][0]
            score_matrix[i, j] = a[this_key].values[scale_index]

    return score_matrix


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

    print('MIN AND MAX COLOUR VALUES = {0:.4f}, {1:.4f}'.format(
        min_colour_value, max_colour_value
    ))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    for this_x_coord in REFERENCE_LINE_X_COORDS:
        these_x_coords = numpy.full(2, this_x_coord)
        these_y_coords = numpy.array([
            -0.5, score_matrix.shape[0] - 0.5
        ])
        axes_object.plot(
            these_x_coords, these_y_coords, linewidth=REFERENCE_LINE_WIDTH,
            linestyle='dotted', color=REFERENCE_LINE_COLOUR
        )

    return figure_object, axes_object


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
        BEST_MARKER_SIZE_GRID_CELLS / len(FILTER_NAMES_FANCY)
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
        colour_bar_file_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param colour_bar_file_name: Path to output file.  Image with colour bar
        will be saved here.
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
        fraction_of_axis_length=1.25
    )

    tick_values = numpy.array([min_colour_value, max_colour_value])
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(['Worst', 'Best'])

    extra_figure_object.savefig(
        colour_bar_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, colour_bar_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(colour_bar_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(all_experiment_dir_name, output_dir_name):
    """Ranks learning-curve scores for aggregated experiment.

    This is effectively the main method.

    :param all_experiment_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_loss_functions = len(LOSS_FUNCTION_NAMES)
    num_filters = len(MIN_RESOLUTIONS_DEG)
    score_matrix = numpy.full(
        (num_loss_functions, num_filters, num_loss_functions, num_filters),
        numpy.nan
    )

    for i in range(num_loss_functions):
        for j in range(num_filters):
            this_matrix = _read_scores_one_model(
                subexperiment_dir_name='{0:s}/lf_experiment{1:02d}'.format(
                    all_experiment_dir_name, SUBEXPERIMENT_ENUMS[j]
                ),
                loss_function_name=LOSS_FUNCTION_NAMES[i],
                wavelet_transform_flag=WAVELET_TRANSFORM_FLAGS[j],
                fourier_transform_flag=FOURIER_TRANSFORM_FLAGS[j],
                min_resolution_deg=MIN_RESOLUTIONS_DEG[j],
                max_resolution_deg=MAX_RESOLUTIONS_DEG[j],
                neigh_half_window_size_px=NEIGH_HALF_WINDOW_SIZES_PX[j]
            )

            if this_matrix is None:
                continue

            score_matrix[i, j, ...] = this_matrix

    rank_matrix = numpy.full(score_matrix.shape, numpy.nan)

    for i in range(num_loss_functions):
        for j in range(num_filters):
            this_score_matrix = score_matrix[..., i, j] + 0.

            if NEGATIVELY_ORIENTED_FLAGS[i]:
                this_score_matrix[numpy.isnan(this_score_matrix)] = numpy.inf
                rank_array = rankdata(-1 * this_score_matrix)
            else:
                this_score_matrix[numpy.isnan(this_score_matrix)] = -numpy.inf
                rank_array = rankdata(this_score_matrix)

            rank_matrix[..., i, j] = numpy.reshape(
                rank_array, this_score_matrix.shape
            )

    print(SEPARATOR_STRING)
    panel_file_names = [''] * num_loss_functions
    panel_letter = chr(ord('a') - 1)

    for i in range(num_loss_functions):
        this_rank_matrix = numpy.mean(
            rank_matrix[MODEL_NAME_INDICES_TO_PLOT, :, i, :], axis=-1
        )
        figure_object, axes_object = _plot_grid_one_score(
            score_matrix=this_rank_matrix,
            min_colour_value=numpy.min(this_rank_matrix),
            max_colour_value=numpy.max(this_rank_matrix),
            colour_map_object=COLOUR_MAP_OBJECT
        )

        y_tick_values = numpy.linspace(
            0, this_rank_matrix.shape[0] - 1, num=this_rank_matrix.shape[0],
            dtype=float
        )
        pyplot.yticks(
            y_tick_values,
            [LOSS_FUNCTION_NAMES_FANCY[k] for k in MODEL_NAME_INDICES_TO_PLOT]
        )
        axes_object.set_ylabel('LF score')

        if i == num_loss_functions - 1:
            x_tick_values = numpy.linspace(
                0, this_rank_matrix.shape[1] - 1, num=this_rank_matrix.shape[1],
                dtype=float
            )
            pyplot.xticks(x_tick_values, FILTER_NAMES_FANCY, rotation=90.)
            axes_object.set_xlabel('LF filter')
        else:
            pyplot.xticks([], [])

        this_index = numpy.nanargmax(numpy.ravel(this_rank_matrix))
        _add_markers(
            figure_object=figure_object, axes_object=axes_object,
            best_marker_indices=
            numpy.unravel_index(this_index, this_rank_matrix.shape)
        )

        panel_letter = chr(ord(panel_letter) + 1)
        score_string = 'Mean ranking on metrics with {0:s}'.format(
            LOSS_FUNCTION_NAMES_FANCY[i]
        )
        title_string = '({0:s}) {1:s}'.format(panel_letter, score_string)
        axes_object.set_title(title_string)

        panel_file_names[i] = '{0:s}/{1:s}_ranking.jpg'.format(
            output_dir_name, LOSS_FUNCTION_NAMES[i]
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[i]))
        figure_object.savefig(
            panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        sort_indices_linear = numpy.argsort(-1 * numpy.ravel(this_rank_matrix))

        for k in range(len(sort_indices_linear)):
            loss_index, filter_index = numpy.unravel_index(
                sort_indices_linear[k], (num_loss_functions, num_filters)
            )
            loss_index = MODEL_NAME_INDICES_TO_PLOT[loss_index]

            model_loss_string = '{0:s} ({1:s})'.format(
                LOSS_FUNCTION_NAMES_FANCY[loss_index],
                FILTER_NAMES_FANCY[filter_index]
            )

            display_string = (
                '{0:d}th-best {1:s} = {2:.1f} ... model trained with {3:s}'
            ).format(
                k + 1, score_string, this_rank_matrix[loss_index, filter_index],
                model_loss_string
            )

            print(display_string)

        print(SEPARATOR_STRING)

    concat_figure_file_name = '{0:s}/ranking_by_score.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=len(panel_file_names), num_panel_columns=1
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    _add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=COLOUR_MAP_OBJECT,
        min_colour_value=1., max_colour_value=2.,
        colour_bar_file_name=
        '{0:s}/ranking_by_score_cbar.jpg'.format(output_dir_name)
    )
    print(SEPARATOR_STRING)

    panel_file_names = []
    panel_letter = chr(ord('a') - 1)

    for j in EVAL_FILTER_INDICES_TO_PLOT:
        this_rank_matrix = numpy.nanmean(
            rank_matrix[MODEL_NAME_INDICES_TO_PLOT, :, :, j], axis=-1
        )
        figure_object, axes_object = _plot_grid_one_score(
            score_matrix=this_rank_matrix,
            min_colour_value=numpy.min(this_rank_matrix),
            max_colour_value=numpy.max(this_rank_matrix),
            colour_map_object=COLOUR_MAP_OBJECT
        )

        y_tick_values = numpy.linspace(
            0, this_rank_matrix.shape[0] - 1, num=this_rank_matrix.shape[0],
            dtype=float
        )
        pyplot.yticks(
            y_tick_values,
            [LOSS_FUNCTION_NAMES_FANCY[k] for k in MODEL_NAME_INDICES_TO_PLOT]
        )
        axes_object.set_ylabel('LF score')

        pyplot.xticks([], [])

        this_index = numpy.nanargmax(numpy.ravel(this_rank_matrix))
        _add_markers(
            figure_object=figure_object, axes_object=axes_object,
            best_marker_indices=
            numpy.unravel_index(this_index, this_rank_matrix.shape)
        )

        panel_letter = chr(ord(panel_letter) + 1)
        score_string = 'Mean ranking on metrics filtered with {0:s}'.format(
            FILTER_NAMES_FANCY[j]
        )
        title_string = '({0:s}) {1:s}'.format(panel_letter, score_string)
        axes_object.set_title(title_string)

        panel_file_names.append(
            '{0:s}/{1:s}_ranking.jpg'.format(output_dir_name, FILTER_NAMES[j])
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
        figure_object.savefig(
            panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        sort_indices_linear = numpy.argsort(-1 * numpy.ravel(this_rank_matrix))

        for k in range(len(sort_indices_linear)):
            loss_index, filter_index = numpy.unravel_index(
                sort_indices_linear[k], (num_loss_functions, num_filters)
            )
            loss_index = MODEL_NAME_INDICES_TO_PLOT[loss_index]

            model_loss_string = '{0:s} ({1:s})'.format(
                LOSS_FUNCTION_NAMES_FANCY[loss_index],
                FILTER_NAMES_FANCY[filter_index]
            )

            display_string = (
                '{0:d}th-best {1:s} = {2:.1f} ... model trained with {3:s}'
            ).format(
                k + 1, score_string, this_rank_matrix[loss_index, filter_index],
                model_loss_string
            )

            print(display_string)

        print(SEPARATOR_STRING)

    this_rank_matrix = numpy.nanmean(
        rank_matrix[MODEL_NAME_INDICES_TO_PLOT, ...], axis=(-2, -1)
    )
    figure_object, axes_object = _plot_grid_one_score(
        score_matrix=this_rank_matrix,
        min_colour_value=numpy.min(this_rank_matrix),
        max_colour_value=numpy.max(this_rank_matrix),
        colour_map_object=COLOUR_MAP_OBJECT
    )

    y_tick_values = numpy.linspace(
        0, this_rank_matrix.shape[0] - 1, num=this_rank_matrix.shape[0],
        dtype=float
    )
    pyplot.yticks(
        y_tick_values,
        [LOSS_FUNCTION_NAMES_FANCY[k] for k in MODEL_NAME_INDICES_TO_PLOT]
    )
    axes_object.set_ylabel('LF score')

    x_tick_values = numpy.linspace(
        0, this_rank_matrix.shape[1] - 1, num=this_rank_matrix.shape[1],
        dtype=float
    )
    pyplot.xticks(x_tick_values, FILTER_NAMES_FANCY, rotation=90.)
    axes_object.set_xlabel('LF filter')

    this_index = numpy.nanargmax(numpy.ravel(this_rank_matrix))
    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=
        numpy.unravel_index(this_index, this_rank_matrix.shape)
    )

    panel_letter = chr(ord(panel_letter) + 1)
    score_string = 'Mean ranking on all metrics'
    title_string = '({0:s}) {1:s}'.format(panel_letter, score_string)
    axes_object.set_title(title_string)

    panel_file_names.append(
        '{0:s}/overall_ranking.jpg'.format(output_dir_name)
    )
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    sort_indices_linear = numpy.argsort(-1 * numpy.ravel(this_rank_matrix))

    for k in range(len(sort_indices_linear)):
        loss_index, filter_index = numpy.unravel_index(
            sort_indices_linear[k], (num_loss_functions, num_filters)
        )
        loss_index = MODEL_NAME_INDICES_TO_PLOT[loss_index]

        model_loss_string = '{0:s} ({1:s})'.format(
            LOSS_FUNCTION_NAMES_FANCY[loss_index],
            FILTER_NAMES_FANCY[filter_index]
        )

        display_string = (
            '{0:d}th-best {1:s} = {2:.1f} ... model trained with {3:s}'
        ).format(
            k + 1, score_string, this_rank_matrix[loss_index, filter_index],
            model_loss_string
        )

        print(display_string)

    print(SEPARATOR_STRING)

    concat_figure_file_name = '{0:s}/ranking_by_filter_and_overall.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=len(panel_file_names), num_panel_columns=1
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    _add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=COLOUR_MAP_OBJECT,
        min_colour_value=1., max_colour_value=2.,
        colour_bar_file_name=
        '{0:s}/ranking_by_filter_and_overall_cbar.jpg'.format(output_dir_name)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        all_experiment_dir_name=getattr(
            INPUT_ARG_OBJECT, ALL_EXPERIMENT_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
