"""Ranks learning curves for Loss-function Experiments 1 and 2."""

import os
import sys
import glob
import argparse
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
import gg_plotting_utils
import imagemagick_utils
import neural_net
import learning_curves

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOLERANCE = 1e-6
MAX_MAX_RESOLUTION_DEG = 1e9

COLOUR_MAP_OBJECT = pyplot.get_cmap('cividis')
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

SCORE_NAME_TO_VERBOSE_DICT = {
    neural_net.BRIER_SCORE_NAME: 'Brier',
    neural_net.FSS_NAME: 'FSS',
    neural_net.CSI_NAME: 'CSI',
    neural_net.DICE_COEFF_NAME: 'Dice',
    neural_net.ALL_CLASS_IOU_NAME: 'IOU'
}

EXP1_LOSS_FUNCTION_NAMES = [
    'brier_neigh0', 'brier_neigh1', 'brier_neigh2', 'brier_neigh3',
    'brier_neigh4', 'brier_neigh6', 'brier_neigh8', 'brier_neigh12',
    'fss_neigh0', 'fss_neigh1', 'fss_neigh2', 'fss_neigh3',
    'fss_neigh4', 'fss_neigh6', 'fss_neigh8', 'fss_neigh12',
    'csi_neigh0', 'csi_neigh1', 'csi_neigh2', 'csi_neigh3',
    'csi_neigh4', 'csi_neigh6', 'csi_neigh8', 'csi_neigh12',
    'dice_neigh0', 'dice_neigh1', 'dice_neigh2', 'dice_neigh3',
    'dice_neigh4', 'dice_neigh6', 'dice_neigh8', 'dice_neigh12'
]

EXP2_LOSS_FUNCTION_NAMES = [
    'all-class-iou_neigh0', 'all-class-iou_neigh1',
    'all-class-iou_neigh2', 'all-class-iou_neigh3',
    'all-class-iou_neigh4', 'all-class-iou_neigh6',
    'all-class-iou_neigh8', 'all-class-iou_neigh12',
    'brier_0.0000d_0.0125d', 'brier_0.0125d_0.0250d',
    'brier_0.0250d_0.0500d', 'brier_0.0500d_0.1000d',
    'brier_0.1000d_0.2000d', 'brier_0.2000d_0.4000d',
    'brier_0.4000d_0.8000d', 'brier_0.8000d_infd',
    'fss_0.0000d_0.0125d', 'fss_0.0125d_0.0250d',
    'fss_0.0250d_0.0500d', 'fss_0.0500d_0.1000d',
    'fss_0.1000d_0.2000d', 'fss_0.2000d_0.4000d',
    'fss_0.4000d_0.8000d', 'fss_0.8000d_infd',
    'csi_0.0000d_0.0125d', 'csi_0.0125d_0.0250d',
    'csi_0.0250d_0.0500d', 'csi_0.0500d_0.1000d',
    'csi_0.1000d_0.2000d', 'csi_0.2000d_0.4000d',
    'csi_0.4000d_0.8000d', 'csi_0.8000d_infd',
    'dice_0.0000d_0.0125d', 'dice_0.0125d_0.0250d',
    'dice_0.0250d_0.0500d', 'dice_0.0500d_0.1000d',
    'dice_0.1000d_0.2000d', 'dice_0.2000d_0.4000d',
    'dice_0.4000d_0.8000d', 'dice_0.8000d_infd',
    'all-class-iou_0.0000d_0.0125d', 'all-class-iou_0.0125d_0.0250d',
    'all-class-iou_0.0250d_0.0500d', 'all-class-iou_0.0500d_0.1000d',
    'all-class-iou_0.1000d_0.2000d', 'all-class-iou_0.2000d_0.4000d',
    'all-class-iou_0.4000d_0.8000d', 'all-class-iou_0.8000d_infd'
]

LOSS_FUNCTION_NAMES = EXP1_LOSS_FUNCTION_NAMES + EXP2_LOSS_FUNCTION_NAMES

UNIQUE_NEIGH_DISTANCES_PX = numpy.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=float)
UNIQUE_MIN_RESOLUTIONS_DEG = numpy.array([
    0, 0.0125, 0.0250, 0.0500, 0.1000, 0.2000, 0.4000, 0.8000
])
UNIQUE_MAX_RESOLUTIONS_DEG = numpy.array([
    0.0125, 0.0250, 0.0500, 0.1000, 0.2000, 0.4000, 0.8000, numpy.inf
])

UNIQUE_NEIGH_SCORE_KEYS = [
    learning_curves.NEIGH_BRIER_SCORE_KEY, learning_curves.NEIGH_CSI_KEY,
    learning_curves.NEIGH_FSS_KEY, learning_curves.NEIGH_DICE_COEFF_KEY
]
UNIQUE_FOURIER_SCORE_KEYS = [
    learning_curves.FOURIER_BRIER_SCORE_KEY, learning_curves.FOURIER_CSI_KEY,
    learning_curves.FOURIER_FSS_KEY,
    learning_curves.FOURIER_DICE_COEFF_KEY, learning_curves.FREQ_MSE_REAL_KEY,
    learning_curves.FREQ_MSE_IMAGINARY_KEY, learning_curves.FREQ_MSE_TOTAL_KEY
]
NEGATIVELY_ORIENTED_KEYS = [
    learning_curves.NEIGH_BRIER_SCORE_KEY,
    learning_curves.FOURIER_BRIER_SCORE_KEY,
    learning_curves.FREQ_MSE_REAL_KEY, learning_curves.FREQ_MSE_IMAGINARY_KEY,
    learning_curves.FREQ_MSE_TOTAL_KEY
]

SCORE_KEY_TO_VERBOSE_DICT = {
    learning_curves.NEIGH_BRIER_SCORE_KEY: 'Brier score',
    learning_curves.NEIGH_CSI_KEY: 'Crtiical success index (CSI)',
    learning_curves.NEIGH_FSS_KEY: 'Fractions skill score (FSS)',
    learning_curves.NEIGH_DICE_COEFF_KEY: 'Dice coefficient',
    learning_curves.FOURIER_BRIER_SCORE_KEY: 'Brier score',
    learning_curves.FOURIER_CSI_KEY: 'Crtiical success index (CSI)',
    learning_curves.FOURIER_FSS_KEY: 'Fractions skill score (FSS)',
    learning_curves.FOURIER_DICE_COEFF_KEY: 'Dice coefficient',
    learning_curves.FREQ_MSE_REAL_KEY: 'MSE for real Fourier spectrum',
    learning_curves.FREQ_MSE_IMAGINARY_KEY:
        'MSE for imaginary Fourier spectrum',
    learning_curves.FREQ_MSE_TOTAL_KEY: 'MSE for total Fourier spectrum'
}

EXPERIMENT1_DIR_ARG_NAME = 'input_experiment1_dir_name'
EXPERIMENT2_DIR_ARG_NAME = 'input_experiment2_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT1_DIR_HELP_STRING = (
    'Name of directory for Experiment 1, containing individual models in '
    'subdirectories.'
)
EXPERIMENT2_DIR_HELP_STRING = (
    'Name of directory for Experiment 2, containing individual models in '
    'subdirectories.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT1_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT1_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT2_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT2_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_scores_one_model(
        experiment_dir_name, loss_function_name, score_keys, neigh_distances_px,
        fourier_min_resolutions_deg, fourier_max_resolutions_deg):
    """Reads learning-curve scores for one model.

    N = number of scores

    :param experiment_dir_name: See documentation at top of file.
    :param loss_function_name: Name of loss function for given model.
    :param score_keys: length-N list of keys for xarray table.
    :param neigh_distances_px: length-N numpy array of neighbourhood distances
        (pixels).
    :param fourier_min_resolutions_deg: length-N numpy array of minimum
        resolutions (degrees) for Fourier band-pass filters.
    :param fourier_max_resolutions_deg: Same but for max resolutions.
    :return: score_values: length-N numpy array of values.
    """

    score_file_pattern = (
        '{0:s}/{1:s}/model_epoch=[0-9][0-9][0-9]_'
        'val-loss=[0-9].[0-9][0-9][0-9][0-9][0-9][0-9]/'
        'validation/partial_grids/learning_curves/advanced_scores.nc'
    ).format(
        experiment_dir_name, loss_function_name.replace('_', '-')
    )

    score_file_names = glob.glob(score_file_pattern)

    if len(score_file_names) == 0:
        error_string = 'Cannot find any files with pattern: {0:s}'.format(
            score_file_pattern
        )
        raise ValueError(error_string)

    score_file_names.sort()

    print('Reading data from: "{0:s}"...'.format(score_file_names[-1]))
    advanced_score_table_xarray = learning_curves.read_scores(
        score_file_names[-1]
    )
    a = advanced_score_table_xarray

    num_scores = len(score_keys)
    score_values = numpy.full(num_scores, numpy.nan)

    for j in range(num_scores):
        if numpy.isnan(neigh_distances_px[j]):
            these_min_resolutions_deg = (
                a.coords[learning_curves.MIN_RESOLUTION_DIM].values
            )
            these_max_resolutions_deg = (
                a.coords[learning_curves.MAX_RESOLUTION_DIM].values
            )
            these_max_resolutions_deg[
                these_max_resolutions_deg >= MAX_MAX_RESOLUTION_DEG
            ] = numpy.inf

            these_diffs = numpy.absolute(
                fourier_min_resolutions_deg[j] - these_min_resolutions_deg
            )
            these_diffs += numpy.absolute(
                fourier_max_resolutions_deg[j] - these_max_resolutions_deg
            )
            these_diffs[numpy.isnan(these_diffs)] = 0.
        else:
            these_diffs = numpy.absolute(
                neigh_distances_px[j] -
                a.coords[learning_curves.NEIGH_DISTANCE_DIM].values
            )

        scale_index = numpy.where(these_diffs <= TOLERANCE)[0][0]
        score_values[j] = a[score_keys[j]].values[scale_index]

    return score_values


def _plot_scores_on_grid(score_values, min_colour_value, max_colour_value,
                         title_string, output_file_name):
    """Plots values of one score on hyperparameter grid.

    L = number of loss functions

    :param score_values: length-L numpy array of values for the given score.
    :param min_colour_value: Minimum value in colour bar.
    :param max_colour_value: Max value in colour bar.
    :param title_string: Figure title (will be printed above figure).
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    param_dicts = [
        neural_net.metric_name_to_params(n) for n in LOSS_FUNCTION_NAMES
    ]

    # Plot neighbourhood-based loss functions.
    neigh_flags = numpy.array(
        [d[neural_net.HALF_WINDOW_SIZE_KEY] is not None for d in param_dicts],
        dtype=bool
    )
    neigh_indices = numpy.where(neigh_flags)[0]
    neigh_param_dicts = [param_dicts[k] for k in neigh_indices]
    neigh_score_values = score_values[neigh_indices]

    neigh_half_widths_px = numpy.array([
        d[neural_net.HALF_WINDOW_SIZE_KEY] for d in neigh_param_dicts
    ])
    neigh_half_widths_px = numpy.round(neigh_half_widths_px).astype(int)

    neigh_score_names = [
        d[neural_net.SCORE_NAME_KEY] for d in neigh_param_dicts
    ]
    neigh_score_names = numpy.array([
        SCORE_NAME_TO_VERBOSE_DICT[n] for n in neigh_score_names
    ])

    num_unique_scores = len(numpy.unique(neigh_score_names))
    num_unique_widths = len(numpy.unique(neigh_half_widths_px))
    neigh_half_width_matrix_px = numpy.reshape(
        neigh_half_widths_px, (num_unique_scores, num_unique_widths)
    )
    neigh_score_name_matrix = numpy.reshape(
        neigh_score_names, (num_unique_scores, num_unique_widths)
    )
    neigh_score_matrix = numpy.reshape(
        neigh_score_values, (num_unique_scores, num_unique_widths)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        neigh_score_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, neigh_score_matrix.shape[1] - 1, num=neigh_score_matrix.shape[1],
        dtype=float
    )
    y_tick_values = numpy.linspace(
        0, neigh_score_matrix.shape[0] - 1, num=neigh_score_matrix.shape[0],
        dtype=float
    )

    x_tick_labels = [
        '{0:d}'.format(2 * w + 1) for w in neigh_half_width_matrix_px[0, :]
    ]
    y_tick_labels = [
        '{0:s}'.format(s) for s in neigh_score_name_matrix[:, 0]
    ]

    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)
    axes_object.set_xlabel('Neighbourhood width for loss function (pixels)')
    axes_object.set_ylabel('Basic score for loss function')
    axes_object.set_title(title_string)

    neigh_figure_file_name = '{0:s}_neigh{1:s}'.format(
        os.path.splitext(output_file_name)[0],
        os.path.splitext(output_file_name)[1]
    )
    figure_object.savefig(
        neigh_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot Fourier-based loss functions.
    fourier_indices = numpy.where(numpy.invert(neigh_flags))[0]
    fourier_param_dicts = [param_dicts[k] for k in fourier_indices]
    fourier_score_values = score_values[fourier_indices]

    fourier_min_resolutions_deg = numpy.array([
        d[neural_net.MIN_RESOLUTION_KEY] for d in fourier_param_dicts
    ])
    fourier_max_resolutions_deg = numpy.array([
        d[neural_net.MAX_RESOLUTION_KEY] for d in fourier_param_dicts
    ])
    fourier_score_names = [
        d[neural_net.SCORE_NAME_KEY] for d in fourier_param_dicts
    ]
    fourier_score_names = numpy.array([
        SCORE_NAME_TO_VERBOSE_DICT[n] for n in fourier_score_names
    ])

    num_unique_scores = len(numpy.unique(fourier_score_names))
    num_unique_bands = len(numpy.unique(fourier_min_resolutions_deg))
    fourier_min_res_matrix_deg = numpy.reshape(
        fourier_min_resolutions_deg, (num_unique_scores, num_unique_bands)
    )
    fourier_max_res_matrix_deg = numpy.reshape(
        fourier_max_resolutions_deg, (num_unique_scores, num_unique_bands)
    )
    fourier_score_name_matrix = numpy.reshape(
        fourier_score_names, (num_unique_scores, num_unique_bands)
    )
    fourier_score_matrix = numpy.reshape(
        fourier_score_values, (num_unique_scores, num_unique_bands)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        fourier_score_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, fourier_score_matrix.shape[1] - 1, num=fourier_score_matrix.shape[1],
        dtype=float
    )
    y_tick_values = numpy.linspace(
        0, fourier_score_matrix.shape[0] - 1, num=fourier_score_matrix.shape[0],
        dtype=float
    )

    x_tick_labels = [
        '[{0:.3g}, {1:.3g}]'.format(a, b) for a, b in
        zip(fourier_min_res_matrix_deg[0, :], fourier_max_res_matrix_deg[0, :])
    ]
    x_tick_labels = [s.replace('inf]', r'$\infty$)') for s in x_tick_labels]
    y_tick_labels = ['{0:s}'.format(s) for s in fourier_score_name_matrix[:, 0]]

    print(x_tick_labels)
    print(y_tick_labels)

    pyplot.xticks(x_tick_values, x_tick_labels, rotation=90.)
    pyplot.yticks(y_tick_values, y_tick_labels)
    axes_object.set_xlabel('Fourier resolution band for loss function (deg)')
    axes_object.set_ylabel('Basic score for loss function')

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )
    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=fourier_score_matrix,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal',
        padding=0.2, extend_min=False, extend_max=False
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    fourier_figure_file_name = '{0:s}_fourier{1:s}'.format(
        os.path.splitext(output_file_name)[0],
        os.path.splitext(output_file_name)[1]
    )
    figure_object.savefig(
        fourier_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Concatenate the two figures.
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[neigh_figure_file_name, fourier_figure_file_name],
        output_file_name=output_file_name,
        num_panel_rows=2, num_panel_columns=1
    )
    os.remove(neigh_figure_file_name)
    os.remove(fourier_figure_file_name)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


def _run(experiment1_dir_name, experiment2_dir_name, output_dir_name):
    """Ranks learning curves for Loss-function Experiments 1 and 2.

    This is effectively the main method.

    :param experiment1_dir_name: See documentation at top of file.
    :param experiment2_dir_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    neigh_score_key_matrix, neigh_distance_matrix_px = numpy.meshgrid(
        numpy.array(UNIQUE_NEIGH_SCORE_KEYS), UNIQUE_NEIGH_DISTANCES_PX
    )
    fourier_score_key_matrix, min_resolution_matrix_deg = numpy.meshgrid(
        numpy.array(UNIQUE_FOURIER_SCORE_KEYS), UNIQUE_MIN_RESOLUTIONS_DEG
    )
    _, max_resolution_matrix_deg = numpy.meshgrid(
        numpy.array(UNIQUE_FOURIER_SCORE_KEYS), UNIQUE_MAX_RESOLUTIONS_DEG
    )

    neigh_score_keys = numpy.ravel(neigh_score_key_matrix)
    neigh_distances_px = numpy.ravel(neigh_distance_matrix_px)
    fourier_score_keys = numpy.ravel(fourier_score_key_matrix)
    fourier_min_resolutions_deg = numpy.ravel(min_resolution_matrix_deg)
    fourier_max_resolutions_deg = numpy.ravel(max_resolution_matrix_deg)

    score_keys = neigh_score_keys.tolist() + fourier_score_keys.tolist()
    neigh_distances_px = numpy.concatenate((
        neigh_distances_px,
        numpy.full(len(fourier_score_keys), numpy.nan)
    ))
    fourier_min_resolutions_deg = numpy.concatenate((
        numpy.full(len(neigh_score_keys), numpy.nan),
        fourier_min_resolutions_deg
    ))
    fourier_max_resolutions_deg = numpy.concatenate((
        numpy.full(len(neigh_score_keys), numpy.nan),
        fourier_max_resolutions_deg
    ))

    num_loss_functions = len(LOSS_FUNCTION_NAMES)
    num_scores = len(score_keys)
    score_matrix = numpy.full((num_loss_functions, num_scores), numpy.nan)

    for i in range(num_loss_functions):
        this_dir_name = (
            experiment1_dir_name
            if LOSS_FUNCTION_NAMES[i] in EXP1_LOSS_FUNCTION_NAMES
            else experiment2_dir_name
        )

        score_matrix[i, :] = _read_scores_one_model(
            experiment_dir_name=this_dir_name,
            loss_function_name=LOSS_FUNCTION_NAMES[i],
            score_keys=score_keys, neigh_distances_px=neigh_distances_px,
            fourier_min_resolutions_deg=fourier_min_resolutions_deg,
            fourier_max_resolutions_deg=fourier_max_resolutions_deg
        )

    print(SEPARATOR_STRING)

    for j in range(num_scores):
        if score_keys[j] in NEGATIVELY_ORIENTED_KEYS:
            sort_indices = numpy.argsort(score_matrix[:, j])
        else:
            sort_indices = numpy.argsort(-1 * score_matrix[:, j])

        for i, k in enumerate(sort_indices):
            if numpy.isnan(neigh_distances_px[j]):
                display_string = (
                    '{0:d}th-best {1:s} from {2:.4f} to {3:.4f} deg = {4:.3g} '
                    '(loss function {5:s})'
                ).format(
                    i + 1, score_keys[j],
                    fourier_min_resolutions_deg[j],
                    fourier_max_resolutions_deg[j],
                    score_matrix[k, j], LOSS_FUNCTION_NAMES[k]
                )
            else:
                display_string = (
                    '{0:d}th-best {1:d}-pixel {2:s} = {3:.3g} '
                    '(loss function {4:s})'
                ).format(
                    i + 1, int(numpy.round(neigh_distances_px[j])),
                    score_keys[j], score_matrix[k, j], LOSS_FUNCTION_NAMES[k]
                )

            print(display_string)

        print(SEPARATOR_STRING)

    for j in range(num_scores):
        if numpy.isnan(neigh_distances_px[j]):
            title_string = '{0:s}, band [{1:.3g}, {2:.3g}]'.format(
                SCORE_KEY_TO_VERBOSE_DICT[score_keys[j]],
                fourier_min_resolutions_deg[j],
                fourier_max_resolutions_deg[j]
            )
            title_string = title_string.replace('inf]', r'$\infty$)')
            title_string = title_string + r'$^{\circ}$'

            pathless_file_name = '{0:s}_{1:.4f}d_{2:.4f}d.jpg'.format(
                score_keys[j], fourier_min_resolutions_deg[j],
                fourier_max_resolutions_deg[j]
            )
        else:
            title_string = '{0:s}, {1:d}-by-{1:d}-pixel neigh'.format(
                SCORE_KEY_TO_VERBOSE_DICT[score_keys[j]],
                int(numpy.round(neigh_distances_px[j]))
            )
            pathless_file_name = '{0:s}_neigh{1:02d}.jpg'.format(
                score_keys[j],
                int(numpy.round(neigh_distances_px[j]))
            )

        _plot_scores_on_grid(
            score_values=score_matrix[:, j],
            min_colour_value=numpy.percentile(score_matrix[:, j], 1.),
            max_colour_value=numpy.percentile(score_matrix[:, j], 99.),
            title_string=title_string,
            output_file_name=
            '{0:s}/{1:s}'.format(output_dir_name, pathless_file_name)
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment1_dir_name=getattr(
            INPUT_ARG_OBJECT, EXPERIMENT1_DIR_ARG_NAME
        ),
        experiment2_dir_name=getattr(
            INPUT_ARG_OBJECT, EXPERIMENT2_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
