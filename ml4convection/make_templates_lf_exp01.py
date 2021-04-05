"""Makes U-net templates for Loss-function Experiment 1."""

import sys
import copy
import os.path
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import radar_io
import u_net_architecture
import neural_net
import custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/lf_experiment01/templates'.format(
    HOME_DIR_NAME
)

FULL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/radar_mask_100km_omit-north.nc'
).format(HOME_DIR_NAME)

PARTIAL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/'
    'radar_mask_100km_omit-north_partial.nc'
).format(HOME_DIR_NAME)

METRIC_NAMES = [
    'brier_neigh0', 'brier_neigh1', 'brier_neigh2', 'brier_neigh3',
    'brier_neigh4', 'brier_neigh6', 'brier_neigh8', 'brier_neigh12',
    'brier_band_0.0000000000deg_0.0125000000deg',
    'brier_band_0.0125000000deg_0.0250000000deg',
    'brier_band_0.0250000000deg_0.0500000000deg',
    'brier_band_0.0500000000deg_0.1000000000deg',
    'brier_band_0.1000000000deg_0.2000000000deg',
    'brier_band_0.2000000000deg_0.4000000000deg',
    'brier_band_0.4000000000deg_0.8000000000deg',
    'brier_band_0.8000000000deg_infdeg',
    'fss_neigh0', 'fss_neigh1', 'fss_neigh2', 'fss_neigh3',
    'fss_neigh4', 'fss_neigh6', 'fss_neigh8', 'fss_neigh12',
    'fss_band_0.0000000000deg_0.0125000000deg',
    'fss_band_0.0125000000deg_0.0250000000deg',
    'fss_band_0.0250000000deg_0.0500000000deg',
    'fss_band_0.0500000000deg_0.1000000000deg',
    'fss_band_0.1000000000deg_0.2000000000deg',
    'fss_band_0.2000000000deg_0.4000000000deg',
    'fss_band_0.4000000000deg_0.8000000000deg',
    'fss_band_0.8000000000deg_infdeg',
    'csi_neigh0', 'csi_neigh1', 'csi_neigh2', 'csi_neigh3',
    'csi_neigh4', 'csi_neigh6', 'csi_neigh8', 'csi_neigh12',
    'csi_band_0.0000000000deg_0.0125000000deg',
    'csi_band_0.0125000000deg_0.0250000000deg',
    'csi_band_0.0250000000deg_0.0500000000deg',
    'csi_band_0.0500000000deg_0.1000000000deg',
    'csi_band_0.1000000000deg_0.2000000000deg',
    'csi_band_0.2000000000deg_0.4000000000deg',
    'csi_band_0.4000000000deg_0.8000000000deg',
    'csi_band_0.8000000000deg_infdeg',
    'iou_neigh0', 'iou_neigh1', 'iou_neigh2', 'iou_neigh3',
    'iou_neigh4', 'iou_neigh6', 'iou_neigh8', 'iou_neigh12',
    'iou_band_0.0000000000deg_0.0125000000deg',
    'iou_band_0.0125000000deg_0.0250000000deg',
    'iou_band_0.0250000000deg_0.0500000000deg',
    'iou_band_0.0500000000deg_0.1000000000deg',
    'iou_band_0.1000000000deg_0.2000000000deg',
    'iou_band_0.2000000000deg_0.4000000000deg',
    'iou_band_0.4000000000deg_0.8000000000deg',
    'iou_band_0.8000000000deg_infdeg',
    'dice_neigh0', 'dice_neigh1', 'dice_neigh2', 'dice_neigh3',
    'dice_neigh4', 'dice_neigh6', 'dice_neigh8', 'dice_neigh12',
    'dice_band_0.0000000000deg_0.0125000000deg',
    'dice_band_0.0125000000deg_0.0250000000deg',
    'dice_band_0.0250000000deg_0.0500000000deg',
    'dice_band_0.0500000000deg_0.1000000000deg',
    'dice_band_0.1000000000deg_0.2000000000deg',
    'dice_band_0.2000000000deg_0.4000000000deg',
    'dice_band_0.4000000000deg_0.8000000000deg',
    'dice_band_0.8000000000deg_infdeg',
    'freqmsereal_band_0.0000000000deg_0.0125000000deg',
    'freqmsereal_band_0.0125000000deg_0.0250000000deg',
    'freqmsereal_band_0.0250000000deg_0.0500000000deg',
    'freqmsereal_band_0.0500000000deg_0.1000000000deg',
    'freqmsereal_band_0.1000000000deg_0.2000000000deg',
    'freqmsereal_band_0.2000000000deg_0.4000000000deg',
    'freqmsereal_band_0.4000000000deg_0.8000000000deg',
    'freqmsereal_band_0.8000000000deg_infdeg',
    'freqmseimag_band_0.0000000000deg_0.0125000000deg',
    'freqmseimag_band_0.0125000000deg_0.0250000000deg',
    'freqmseimag_band_0.0250000000deg_0.0500000000deg',
    'freqmseimag_band_0.0500000000deg_0.1000000000deg',
    'freqmseimag_band_0.1000000000deg_0.2000000000deg',
    'freqmseimag_band_0.2000000000deg_0.4000000000deg',
    'freqmseimag_band_0.4000000000deg_0.8000000000deg',
    'freqmseimag_band_0.8000000000deg_infdeg',
    'freqmse_band_0.0000000000deg_0.0125000000deg',
    'freqmse_band_0.0125000000deg_0.0250000000deg',
    'freqmse_band_0.0250000000deg_0.0500000000deg',
    'freqmse_band_0.0500000000deg_0.1000000000deg',
    'freqmse_band_0.1000000000deg_0.2000000000deg',
    'freqmse_band_0.2000000000deg_0.4000000000deg',
    'freqmse_band_0.4000000000deg_0.8000000000deg',
    'freqmse_band_0.8000000000deg_infdeg'
]

LOSS_FUNCTION_NAMES = METRIC_NAMES[:80]

OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([205, 205, 21], dtype=int),
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.NUM_CONV_LAYERS_KEY: 2,
    u_net_architecture.CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.L2_WEIGHT_KEY: 10 ** -5.5
}


def _run():
    """Makes U-net templates for Loss-function Experiment 1.

    This is effectively the main method.
    """

    print('Reading full mask from: "{0:s}"...'.format(FULL_MASK_FILE_NAME))
    full_mask_matrix = radar_io.read_mask_file(FULL_MASK_FILE_NAME)[
        radar_io.MASK_MATRIX_KEY
    ]

    print('Reading partial mask from: "{0:s}"...'.format(
        PARTIAL_MASK_FILE_NAME
    ))
    partial_mask_matrix = radar_io.read_mask_file(PARTIAL_MASK_FILE_NAME)[
        radar_io.MASK_MATRIX_KEY
    ]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_loss_functions = len(LOSS_FUNCTION_NAMES)

    for i in range(num_loss_functions):
        print(SEPARATOR_STRING)

        this_loss_function = neural_net.get_metrics(
            metric_names=[LOSS_FUNCTION_NAMES[i]],
            mask_matrix=partial_mask_matrix,
            use_as_loss_function=True
        )[0][0]

        this_model_object = u_net_architecture.create_model(
            option_dict=OPTION_DICT, loss_function=this_loss_function,
            mask_matrix=partial_mask_matrix, metric_names=METRIC_NAMES
        )

        this_model_file_name = '{0:s}/model_{1:s}.h5'.format(
            OUTPUT_DIR_NAME, LOSS_FUNCTION_NAMES[i].replace('_', '-')
        )

        print('Writing model to: "{0:s}"...'.format(this_model_file_name))
        this_model_object.save(
            filepath=this_model_file_name, overwrite=True,
            include_optimizer=True
        )

        this_metafile_name = neural_net.find_metafile(
            model_file_name=this_model_file_name,
            raise_error_if_missing=False
        )
        dummy_option_dict = neural_net.DEFAULT_GENERATOR_OPTION_DICT

        print('Writing metadata to: "{0:s}"...'.format(this_metafile_name))
        neural_net._write_metafile(
            dill_file_name=this_metafile_name, use_partial_grids=True,
            num_epochs=100, num_training_batches_per_epoch=100,
            training_option_dict=dummy_option_dict,
            num_validation_batches_per_epoch=100,
            validation_option_dict=dummy_option_dict,
            do_early_stopping=True, plateau_lr_multiplier=0.6,
            loss_function_name=LOSS_FUNCTION_NAMES[i],
            metric_names=METRIC_NAMES,
            mask_matrix=partial_mask_matrix, full_mask_matrix=full_mask_matrix
        )


if __name__ == '__main__':
    _run()
