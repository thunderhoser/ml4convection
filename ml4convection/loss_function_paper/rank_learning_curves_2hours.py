"""Ranks learning-curve scores for experiment with 2-hour lead time."""

import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
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
    'brier', 'fss', 'xentropy', 'iou', 'all-class-iou', 'dice', 'csi',
    'heidke', 'gerrity', 'peirce'
]
LOSS_FUNCTION_NAMES_FANCY = [
    'Brier', 'FSS', 'X-entropy', r'IOU$_{pos}$', r'IOU$_{all}$', 'Dice', 'CSI',
    'Heidke', 'Gerrity', 'Peirce'
]
NEGATIVELY_ORIENTED_FLAGS = numpy.array(
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool
)

NUM_MODEL_LOSS_FUNCTIONS = 3

LOSS_FUNCTION_KEYS_NEIGH = [
    learning_curves.NEIGH_BRIER_SCORE_KEY, learning_curves.NEIGH_FSS_KEY,
    learning_curves.NEIGH_XENTROPY_KEY, learning_curves.NEIGH_IOU_KEY,
    learning_curves.NEIGH_ALL_CLASS_IOU_KEY,
    learning_curves.NEIGH_DICE_COEFF_KEY, learning_curves.NEIGH_CSI_KEY,
    None, None, None
]
LOSS_FUNCTION_KEYS_FOURIER = [
    learning_curves.FOURIER_BRIER_SCORE_KEY, learning_curves.FOURIER_FSS_KEY,
    learning_curves.FOURIER_XENTROPY_KEY, learning_curves.FOURIER_IOU_KEY,
    learning_curves.FOURIER_ALL_CLASS_IOU_KEY,
    learning_curves.FOURIER_DICE_COEFF_KEY, learning_curves.FOURIER_CSI_KEY,
    learning_curves.FOURIER_HEIDKE_SCORE_KEY,
    learning_curves.FOURIER_GERRITY_SCORE_KEY,
    learning_curves.FOURIER_PEIRCE_SCORE_KEY
]
LOSS_FUNCTION_KEYS_WAVELET = [
    learning_curves.WAVELET_BRIER_SCORE_KEY, learning_curves.WAVELET_FSS_KEY,
    learning_curves.WAVELET_XENTROPY_KEY, learning_curves.WAVELET_IOU_KEY,
    learning_curves.WAVELET_ALL_CLASS_IOU_KEY,
    learning_curves.WAVELET_DICE_COEFF_KEY, learning_curves.WAVELET_CSI_KEY,
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

EXPERIMENT_DIR_ARG_NAME = 'input_experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = (
    'Name of directory containing results for each model.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_scores_one_model(
        experiment_dir_name, loss_function_name, wavelet_transform_flag,
        fourier_transform_flag, min_resolution_deg, max_resolution_deg,
        neigh_half_window_size_px):
    """Reads scores for one model.

    L = number of loss functions
    F = number of filters

    :param experiment_dir_name: See documentation at top of file.
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
        experiment_dir_name, this_string
    )
    score_file_names = glob.glob(score_file_pattern)

    if len(score_file_names) == 0:
        return None

    if len(score_file_names) == 1:
        min_index = 0
    else:
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
    num_filters = len(FILTER_NAMES)
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
    pyplot.yticks(
        y_tick_values, LOSS_FUNCTION_NAMES_FANCY[:NUM_MODEL_LOSS_FUNCTIONS]
    )

    gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=score_matrix,
        colour_map_object=colour_map_object, min_value=min_colour_value,
        max_value=max_colour_value, orientation_string='vertical',
        extend_min=False, extend_max=False, font_size=COLOUR_BAR_FONT_SIZE,
        fraction_of_axis_length=0.3
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
        BEST_MARKER_SIZE_GRID_CELLS / len(FILTER_NAMES)
    )

    axes_object.plot(
        best_marker_indices[1], best_marker_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )


def _run(experiment_dir_name, output_dir_name):
    """Ranks learning-curve scores for aggregated experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_eval_loss_functions = len(LOSS_FUNCTION_NAMES)
    num_filters = len(MIN_RESOLUTIONS_DEG)
    dimensions = (
        NUM_MODEL_LOSS_FUNCTIONS, num_filters,
        num_eval_loss_functions, num_filters
    )

    score_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(NUM_MODEL_LOSS_FUNCTIONS):
        for j in range(num_filters):
            this_matrix = _read_scores_one_model(
                experiment_dir_name=experiment_dir_name,
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

    print(SEPARATOR_STRING)

    for i in range(num_eval_loss_functions):
        for j in range(num_filters):
            if numpy.all(numpy.isnan(score_matrix[..., i, j])):
                continue

            figure_object, axes_object = _plot_grid_one_score(
                score_matrix=score_matrix[..., i, j],
                min_colour_value=numpy.nanpercentile(
                    score_matrix[..., i, j], 0
                ),
                max_colour_value=numpy.nanpercentile(
                    score_matrix[..., i, j],
                    50 if LOSS_FUNCTION_NAMES[i] == 'brier' else 100
                ),
                colour_map_object=COLOUR_MAP_OBJECT
            )

            if NEGATIVELY_ORIENTED_FLAGS[i]:
                this_index = numpy.nanargmin(numpy.ravel(
                    score_matrix[..., i, j]
                ))
            else:
                this_index = numpy.nanargmax(numpy.ravel(
                    score_matrix[..., i, j]
                ))

            _add_markers(
                figure_object=figure_object, axes_object=axes_object,
                best_marker_indices=numpy.unravel_index(
                    this_index, score_matrix[..., i, j].shape
                )
            )

            axes_object.set_ylabel('Score for model''s loss function')
            axes_object.set_xlabel('Filter for model''s loss function')

            score_string = '{0:s} ({1:s})'.format(
                LOSS_FUNCTION_NAMES_FANCY[i], FILTER_NAMES[j]
            )
            title_string = score_string + ' for different models'

            if FOURIER_TRANSFORM_FLAGS[j] or WAVELET_TRANSFORM_FLAGS[j]:
                output_file_name = (
                    '{0:s}/{1:s}_{2:s}_{3:.4f}-{4:.4f}-deg.jpg'
                ).format(
                    output_dir_name, LOSS_FUNCTION_NAMES[i],
                    'fourier' if FOURIER_TRANSFORM_FLAGS[j] else 'wavelet',
                    MIN_RESOLUTIONS_DEG[j], MAX_RESOLUTIONS_DEG[j]
                )
            else:
                output_file_name = (
                    '{0:s}/{1:s}_neigh{2:02d}.jpg'
                ).format(
                    output_dir_name, LOSS_FUNCTION_NAMES[i],
                    int(numpy.round(NEIGH_HALF_WINDOW_SIZES_PX[j]))
                )

            axes_object.set_title(title_string)

            print('Saving figure to: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            if NEGATIVELY_ORIENTED_FLAGS[i]:
                sort_indices_linear = numpy.argsort(
                    numpy.ravel(score_matrix[..., i, j])
                )
            else:
                sort_indices_linear = numpy.argsort(
                    -1 * numpy.ravel(score_matrix[..., i, j])
                )

            for k in range(len(sort_indices_linear)):
                loss_index, filter_index = numpy.unravel_index(
                    sort_indices_linear[k],
                    (NUM_MODEL_LOSS_FUNCTIONS, num_filters)
                )

                model_loss_string = '{0:s} ({1:s})'.format(
                    LOSS_FUNCTION_NAMES_FANCY[loss_index],
                    FILTER_NAMES[filter_index]
                )

                display_string = (
                    '{0:d}th-best {1:s} = {2:.4f} ... model trained with {3:s}'
                ).format(
                    k + 1, score_string,
                    score_matrix[loss_index, filter_index, i, j],
                    model_loss_string
                )

                print(display_string)

            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(
            INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
