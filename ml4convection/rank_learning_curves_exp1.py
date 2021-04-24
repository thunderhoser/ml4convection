"""Ranks learning curves for Loss-function Experiment 1."""

import os
import sys
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import learning_curves

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOLERANCE = 1e-6
MAX_MAX_RESOLUTION_DEG = 1e9

LOSS_FUNCTION_NAMES = [
    'brier_neigh0', 'brier_neigh1', 'brier_neigh2', 'brier_neigh3',
    'brier_neigh4', 'brier_neigh6', 'brier_neigh8', 'brier_neigh12',
    'brier_0.0000d_0.0125d',
    'brier_0.0125d_0.0250d',
    'brier_0.0250d_0.0500d',
    'brier_0.0500d_0.1000d',
    'brier_0.1000d_0.2000d',
    'brier_0.2000d_0.4000d',
    'brier_0.4000d_0.8000d',
    'brier_0.8000d_infd',
    'fss_neigh0', 'fss_neigh1', 'fss_neigh2', 'fss_neigh3',
    'fss_neigh4', 'fss_neigh6', 'fss_neigh8', 'fss_neigh12',
    'fss_0.0000d_0.0125d',
    'fss_0.0125d_0.0250d',
    'fss_0.0250d_0.0500d',
    'fss_0.0500d_0.1000d',
    'fss_0.1000d_0.2000d',
    'fss_0.2000d_0.4000d',
    'fss_0.4000d_0.8000d',
    'fss_0.8000d_infd',
    'csi_neigh0', 'csi_neigh1', 'csi_neigh2', 'csi_neigh3',
    'csi_neigh4', 'csi_neigh6', 'csi_neigh8', 'csi_neigh12',
    'csi_0.0000d_0.0125d',
    'csi_0.0125d_0.0250d',
    'csi_0.0250d_0.0500d',
    'csi_0.0500d_0.1000d',
    'csi_0.1000d_0.2000d',
    'csi_0.2000d_0.4000d',
    'csi_0.4000d_0.8000d',
    'csi_0.8000d_infd',
    'iou_neigh0', 'iou_neigh1', 'iou_neigh2', 'iou_neigh3',
    'iou_neigh4', 'iou_neigh6', 'iou_neigh8', 'iou_neigh12',
    'iou_0.0000d_0.0125d',
    'iou_0.0125d_0.0250d',
    'iou_0.0250d_0.0500d',
    'iou_0.0500d_0.1000d',
    'iou_0.1000d_0.2000d',
    'iou_0.2000d_0.4000d',
    'iou_0.4000d_0.8000d',
    'iou_0.8000d_infd',
    'dice_neigh0', 'dice_neigh1', 'dice_neigh2', 'dice_neigh3',
    'dice_neigh4', 'dice_neigh6', 'dice_neigh8', 'dice_neigh12',
    'dice_0.0000d_0.0125d',
    'dice_0.0125d_0.0250d',
    'dice_0.0250d_0.0500d',
    'dice_0.0500d_0.1000d',
    'dice_0.1000d_0.2000d',
    'dice_0.2000d_0.4000d',
    'dice_0.4000d_0.8000d',
    'dice_0.8000d_infd'
]

UNIQUE_NEIGH_DISTANCES_PX = numpy.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=float)
UNIQUE_MIN_RESOLUTIONS_DEG = numpy.array([
    0, 0.0125, 0.0250, 0.0500, 0.1000, 0.2000, 0.4000, 0.8000
])
UNIQUE_MAX_RESOLUTIONS_DEG = numpy.array([
    0.0125, 0.0250, 0.0500, 0.1000, 0.2000, 0.4000, 0.8000, numpy.inf
])

UNIQUE_NEIGH_SCORE_KEYS = [
    learning_curves.NEIGH_BRIER_SCORE_KEY, learning_curves.NEIGH_CSI_KEY,
    learning_curves.NEIGH_FSS_KEY, learning_curves.NEIGH_IOU_KEY,
    learning_curves.NEIGH_DICE_COEFF_KEY
]
UNIQUE_FOURIER_SCORE_KEYS = [
    learning_curves.FOURIER_BRIER_SCORE_KEY, learning_curves.FOURIER_CSI_KEY,
    learning_curves.FOURIER_FSS_KEY, learning_curves.FOURIER_IOU_KEY,
    learning_curves.FOURIER_DICE_COEFF_KEY, learning_curves.FREQ_MSE_REAL_KEY,
    learning_curves.FREQ_MSE_IMAGINARY_KEY, learning_curves.FREQ_MSE_TOTAL_KEY
]
NEGATIVELY_ORIENTED_KEYS = [
    learning_curves.NEIGH_BRIER_SCORE_KEY,
    learning_curves.FOURIER_BRIER_SCORE_KEY,
    learning_curves.FREQ_MSE_REAL_KEY, learning_curves.FREQ_MSE_IMAGINARY_KEY,
    learning_curves.FREQ_MSE_TOTAL_KEY
]

EXPERIMENT_DIR_ARG_NAME = 'input_experiment_dir_name'
EXPERIMENT_DIR_HELP_STRING = (
    'Name of experiment directory, containing individual models in '
    'subdirectories.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
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


def _run(experiment_dir_name):
    """Ranks learning curves for Loss-function Experiment 1.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    """

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
        score_matrix[i, :] = _read_scores_one_model(
            experiment_dir_name=experiment_dir_name,
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
            display_string = (
                '{0:d}th-best {1:s} = {2:.3g} (loss function {3:s})'
            ).format(
                i + 1, score_keys[j], score_matrix[k, j], LOSS_FUNCTION_NAMES[k]
            )

            print(display_string)

    print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME)
    )
