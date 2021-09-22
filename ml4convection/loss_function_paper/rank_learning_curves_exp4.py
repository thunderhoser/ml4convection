"""Ranks learning curves for Loss-function Experiment 4."""

import glob
import argparse
import numpy
from ml4convection.utils import learning_curves
from ml4convection.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOLERANCE = 1e-6
MAX_MAX_RESOLUTION_DEG = 1e10

LOSS_FUNCTION_NAMES = [
    'brier_0.0000d_0.0125d_wavelets0', 'brier_0.0125d_0.0250d_wavelets0',
    'brier_0.0250d_0.0500d_wavelets0', 'brier_0.0500d_0.1000d_wavelets0',
    'brier_0.1000d_0.2000d_wavelets0', 'brier_0.2000d_0.4000d_wavelets0',
    'brier_0.4000d_0.8000d_wavelets0', 'brier_0.8000d_infd_wavelets0',
    'fss_0.0000d_0.0125d_wavelets0', 'fss_0.0125d_0.0250d_wavelets0',
    'fss_0.0250d_0.0500d_wavelets0', 'fss_0.0500d_0.1000d_wavelets0',
    'fss_0.1000d_0.2000d_wavelets0', 'fss_0.2000d_0.4000d_wavelets0',
    'fss_0.4000d_0.8000d_wavelets0', 'fss_0.8000d_infd_wavelets0',
    'csi_0.0000d_0.0125d_wavelets0', 'csi_0.0125d_0.0250d_wavelets0',
    'csi_0.0250d_0.0500d_wavelets0', 'csi_0.0500d_0.1000d_wavelets0',
    'csi_0.1000d_0.2000d_wavelets0', 'csi_0.2000d_0.4000d_wavelets0',
    'csi_0.4000d_0.8000d_wavelets0', 'csi_0.8000d_infd_wavelets0',
    'iou_0.0000d_0.0125d_wavelets0', 'iou_0.0125d_0.0250d_wavelets0',
    'iou_0.0250d_0.0500d_wavelets0', 'iou_0.0500d_0.1000d_wavelets0',
    'iou_0.1000d_0.2000d_wavelets0', 'iou_0.2000d_0.4000d_wavelets0',
    'iou_0.4000d_0.8000d_wavelets0', 'iou_0.8000d_infd_wavelets0',
    'all-class-iou_0.0000d_0.0125d_wavelets0',
    'all-class-iou_0.0125d_0.0250d_wavelets0',
    'all-class-iou_0.0250d_0.0500d_wavelets0',
    'all-class-iou_0.0500d_0.1000d_wavelets0',
    'all-class-iou_0.1000d_0.2000d_wavelets0',
    'all-class-iou_0.2000d_0.4000d_wavelets0',
    'all-class-iou_0.4000d_0.8000d_wavelets0',
    'all-class-iou_0.8000d_infd_wavelets0',
    'dice_0.0000d_0.0125d_wavelets0', 'dice_0.0125d_0.0250d_wavelets0',
    'dice_0.0250d_0.0500d_wavelets0', 'dice_0.0500d_0.1000d_wavelets0',
    'dice_0.1000d_0.2000d_wavelets0', 'dice_0.2000d_0.4000d_wavelets0',
    'dice_0.4000d_0.8000d_wavelets0', 'dice_0.8000d_infd_wavelets0',
    'brier_0.0000d_0.0125d_wavelets1', 'brier_0.0125d_0.0250d_wavelets1',
    'brier_0.0250d_0.0500d_wavelets1', 'brier_0.0500d_0.1000d_wavelets1',
    'brier_0.1000d_0.2000d_wavelets1', 'brier_0.2000d_0.4000d_wavelets1',
    'brier_0.4000d_0.8000d_wavelets1', 'brier_0.8000d_infd_wavelets1',
    'fss_0.0000d_0.0125d_wavelets1', 'fss_0.0125d_0.0250d_wavelets1',
    'fss_0.0250d_0.0500d_wavelets1', 'fss_0.0500d_0.1000d_wavelets1',
    'fss_0.1000d_0.2000d_wavelets1', 'fss_0.2000d_0.4000d_wavelets1',
    'fss_0.4000d_0.8000d_wavelets1', 'fss_0.8000d_infd_wavelets1',
    'csi_0.0000d_0.0125d_wavelets1', 'csi_0.0125d_0.0250d_wavelets1',
    'csi_0.0250d_0.0500d_wavelets1', 'csi_0.0500d_0.1000d_wavelets1',
    'csi_0.1000d_0.2000d_wavelets1', 'csi_0.2000d_0.4000d_wavelets1',
    'csi_0.4000d_0.8000d_wavelets1', 'csi_0.8000d_infd_wavelets1',
    'iou_0.0000d_0.0125d_wavelets1', 'iou_0.0125d_0.0250d_wavelets1',
    'iou_0.0250d_0.0500d_wavelets1', 'iou_0.0500d_0.1000d_wavelets1',
    'iou_0.1000d_0.2000d_wavelets1', 'iou_0.2000d_0.4000d_wavelets1',
    'iou_0.4000d_0.8000d_wavelets1', 'iou_0.8000d_infd_wavelets1',
    'all-class-iou_0.0000d_0.0125d_wavelets1',
    'all-class-iou_0.0125d_0.0250d_wavelets1',
    'all-class-iou_0.0250d_0.0500d_wavelets1',
    'all-class-iou_0.0500d_0.1000d_wavelets1',
    'all-class-iou_0.1000d_0.2000d_wavelets1',
    'all-class-iou_0.2000d_0.4000d_wavelets1',
    'all-class-iou_0.4000d_0.8000d_wavelets1',
    'all-class-iou_0.8000d_infd_wavelets1',
    'dice_0.0000d_0.0125d_wavelets1', 'dice_0.0125d_0.0250d_wavelets1',
    'dice_0.0250d_0.0500d_wavelets1', 'dice_0.0500d_0.1000d_wavelets1',
    'dice_0.1000d_0.2000d_wavelets1', 'dice_0.2000d_0.4000d_wavelets1',
    'dice_0.4000d_0.8000d_wavelets1', 'dice_0.8000d_infd_wavelets1'
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
    learning_curves.FOURIER_DICE_COEFF_KEY,
    learning_curves.FOURIER_COEFF_MSE_REAL_KEY,
    learning_curves.FOURIER_COEFF_MSE_IMAGINARY_KEY,
    learning_curves.FOURIER_COEFF_MSE_TOTAL_KEY
]
UNIQUE_WAVELET_SCORE_KEYS = [
    learning_curves.WAVELET_BRIER_SCORE_KEY, learning_curves.WAVELET_CSI_KEY,
    learning_curves.WAVELET_FSS_KEY, learning_curves.WAVELET_IOU_KEY,
    learning_curves.WAVELET_DICE_COEFF_KEY,
    learning_curves.WAVELET_COEFF_MSE_MEAN_KEY,
    learning_curves.WAVELET_COEFF_MSE_DETAIL_KEY
]
NEGATIVELY_ORIENTED_KEYS = [
    learning_curves.NEIGH_BRIER_SCORE_KEY,
    learning_curves.FOURIER_BRIER_SCORE_KEY,
    learning_curves.FOURIER_COEFF_MSE_REAL_KEY,
    learning_curves.FOURIER_COEFF_MSE_IMAGINARY_KEY,
    learning_curves.FOURIER_COEFF_MSE_TOTAL_KEY,
    learning_curves.WAVELET_BRIER_SCORE_KEY,
    learning_curves.WAVELET_COEFF_MSE_MEAN_KEY,
    learning_curves.WAVELET_COEFF_MSE_DETAIL_KEY
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
        fourier_min_resolutions_deg, fourier_max_resolutions_deg,
        wavelet_min_resolutions_deg, wavelet_max_resolutions_deg):
    """Reads learning-curve scores for one model.

    N = number of scores

    :param experiment_dir_name: See documentation at top of file.
    :param loss_function_name: Name of loss function for given model.
    :param score_keys: length-N list of keys for xarray table.
    :param neigh_distances_px: length-N numpy array of neighbourhood distances
        (pixels).
    :param fourier_min_resolutions_deg: length-N numpy array of minimum
        resolutions (degrees) for Fourier band-pass filters.
    :param fourier_max_resolutions_deg: Same as above but for max resolutions.
    :param wavelet_max_resolutions_deg: Same as `fourier_min_resolutions_deg`
        but for wavelet band-pass filter.
    :param wavelet_max_resolutions_deg: Same as above but for max resolutions.
    :return: score_values: length-N numpy array of values.
    """

    param_dict = neural_net.metric_name_to_params(loss_function_name)
    if numpy.isinf(param_dict[neural_net.MAX_RESOLUTION_KEY]):
        param_dict[neural_net.MAX_RESOLUTION_KEY] = MAX_MAX_RESOLUTION_DEG

    this_string = (
        '{0:s}_wavelets{1:d}_min-resolution-deg={2:.4f}_'
        'max-resolution-deg={3:.4f}'
    ).format(
        loss_function_name.split('_')[0],
        int(param_dict[neural_net.USE_WAVELETS_KEY]),
        param_dict[neural_net.MIN_RESOLUTION_KEY],
        param_dict[neural_net.MAX_RESOLUTION_KEY]
    )

    score_file_pattern = (
        '{0:s}/{1:s}/model*/validation/partial_grids/learning_curves/'
        'advanced_scores.nc'
    ).format(
        experiment_dir_name, this_string
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

    table_neigh_distances_px = (
        a.coords[learning_curves.NEIGH_DISTANCE_DIM].values
    )
    table_min_resolutions_deg = (
        a.coords[learning_curves.MIN_RESOLUTION_DIM].values
    )
    table_max_resolutions_deg = (
        a.coords[learning_curves.MAX_RESOLUTION_DIM].values
    )
    table_max_resolutions_deg[
        table_max_resolutions_deg >= MAX_MAX_RESOLUTION_DEG
    ] = numpy.inf

    num_scores = len(score_keys)
    score_values = numpy.full(num_scores, numpy.nan)

    for j in range(num_scores):
        if not numpy.isnan(neigh_distances_px[j]):
            these_diffs = numpy.absolute(
                neigh_distances_px[j] - table_neigh_distances_px
            )
        elif not numpy.isnan(fourier_min_resolutions_deg[j]):
            these_diffs = numpy.absolute(
                fourier_min_resolutions_deg[j] - table_min_resolutions_deg
            )
            these_diffs += numpy.absolute(
                fourier_max_resolutions_deg[j] - table_max_resolutions_deg
            )
            these_diffs[numpy.isnan(these_diffs)] = 0.
        else:
            these_diffs = numpy.absolute(
                wavelet_min_resolutions_deg[j] - table_min_resolutions_deg
            )
            these_diffs += numpy.absolute(
                wavelet_max_resolutions_deg[j] - table_max_resolutions_deg
            )
            these_diffs[numpy.isnan(these_diffs)] = 0.

        scale_index = numpy.where(these_diffs <= TOLERANCE)[0][0]
        score_values[j] = a[score_keys[j]].values[scale_index]

    return score_values


def _run(experiment_dir_name):
    """Ranks learning curves for Loss-function Experiment 4.

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

    wavelet_score_key_matrix, min_resolution_matrix_deg = numpy.meshgrid(
        numpy.array(UNIQUE_WAVELET_SCORE_KEYS), UNIQUE_MIN_RESOLUTIONS_DEG
    )
    _, max_resolution_matrix_deg = numpy.meshgrid(
        numpy.array(UNIQUE_WAVELET_SCORE_KEYS), UNIQUE_MAX_RESOLUTIONS_DEG
    )

    wavelet_score_keys = numpy.ravel(wavelet_score_key_matrix)
    wavelet_min_resolutions_deg = numpy.ravel(min_resolution_matrix_deg)
    wavelet_max_resolutions_deg = numpy.ravel(max_resolution_matrix_deg)

    score_keys = numpy.concatenate((
        neigh_score_keys, fourier_score_keys, wavelet_score_keys
    ))

    neigh_distances_px = numpy.concatenate((
        neigh_distances_px,
        numpy.full(len(fourier_score_keys), numpy.nan),
        numpy.full(len(wavelet_score_keys), numpy.nan)
    ))
    fourier_min_resolutions_deg = numpy.concatenate((
        numpy.full(len(neigh_score_keys), numpy.nan),
        fourier_min_resolutions_deg,
        numpy.full(len(wavelet_score_keys), numpy.nan)
    ))
    fourier_max_resolutions_deg = numpy.concatenate((
        numpy.full(len(neigh_score_keys), numpy.nan),
        fourier_max_resolutions_deg,
        numpy.full(len(wavelet_score_keys), numpy.nan)
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
            fourier_max_resolutions_deg=fourier_max_resolutions_deg,
            wavelet_min_resolutions_deg=wavelet_min_resolutions_deg,
            wavelet_max_resolutions_deg=wavelet_max_resolutions_deg
        )

    print(SEPARATOR_STRING)

    for j in range(num_scores):
        if score_keys[j] in NEGATIVELY_ORIENTED_KEYS:
            sort_indices = numpy.argsort(score_matrix[:, j])
        else:
            sort_indices = numpy.argsort(-1 * score_matrix[:, j])

        for i, k in enumerate(sort_indices):
            if not numpy.isnan(fourier_min_resolutions_deg[j]):
                display_string = (
                    '{0:d}th-best {1:s} with Fourier BPF from {2:.4f} to '
                    '{3:.4f} deg = {4:.3g} (loss function {5:s})'
                ).format(
                    i + 1, score_keys[j],
                    fourier_min_resolutions_deg[j],
                    fourier_max_resolutions_deg[j],
                    score_matrix[k, j], LOSS_FUNCTION_NAMES[k]
                )
            elif not numpy.isnan(wavelet_min_resolutions_deg[j]):
                display_string = (
                    '{0:d}th-best {1:s} with wavelet BPF from {2:.4f} to '
                    '{3:.4f} deg = {4:.3g} (loss function {5:s})'
                ).format(
                    i + 1, score_keys[j],
                    wavelet_min_resolutions_deg[j],
                    wavelet_max_resolutions_deg[j],
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


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME)
    )
