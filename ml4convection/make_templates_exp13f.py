"""Makes U-net templates for Experiment 13f."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import radar_io
import u_net_architecture
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/experiment13f/templates'.format(
    HOME_DIR_NAME
)

FULL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/radar_mask_100km_omit-north.nc'
).format(HOME_DIR_NAME)

PARTIAL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/'
    'radar_mask_100km_omit-north_partial.nc'
).format(HOME_DIR_NAME)

LOSS_FUNCTION_NAME = 'fss_neigh4'
METRIC_NAMES = [
    'fss_neigh0', 'fss_neigh1', 'fss_neigh2', 'fss_neigh3',
    'csi_neigh0', 'csi_neigh1', 'csi_neigh2', 'csi_neigh3',
    'bias_neigh0', 'bias_neigh1', 'bias_neigh2', 'bias_neigh3',
    'iou_neigh0', 'iou_neigh1', 'iou_neigh2', 'iou_neigh3',
    'dice_neigh0', 'dice_neigh1', 'dice_neigh2', 'dice_neigh3'
]

L2_WEIGHTS = numpy.logspace(-7, -5, num=5)
LAG_TIME_COUNTS = numpy.linspace(1, 13, num=13, dtype=int)

DEFAULT_OPTION_DICT = {
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.NUM_CONV_LAYERS_KEY: 2,
    u_net_architecture.CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.)
}


def _run():
    """Makes U-net templates for Experiment 13f.

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

    loss_function = neural_net.get_metrics(
        metric_names=[LOSS_FUNCTION_NAME], mask_matrix=partial_mask_matrix,
        use_as_loss_function=True
    )[0][0]

    print(loss_function)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_l2_weights = len(L2_WEIGHTS)
    num_lag_time_counts = len(LAG_TIME_COUNTS)

    for i in range(num_l2_weights):
        for j in range(num_lag_time_counts):
            this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
            this_option_dict[u_net_architecture.L2_WEIGHT_KEY] = L2_WEIGHTS[i]
            this_option_dict[u_net_architecture.INPUT_DIMENSIONS_KEY] = (
                numpy.array([205, 205, LAG_TIME_COUNTS[j] * 7], dtype=int)
            )

            print(this_option_dict)
            print(SEPARATOR_STRING)

            this_model_object = u_net_architecture.create_model(
                option_dict=this_option_dict, loss_function=loss_function,
                mask_matrix=partial_mask_matrix, metric_names=METRIC_NAMES
            )

            this_model_file_name = (
                '{0:s}/model_l2-weight={1:.10f}_num-lag-times={2:02d}.h5'
            ).format(
                OUTPUT_DIR_NAME, L2_WEIGHTS[i], LAG_TIME_COUNTS[j]
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

            print('Writing metadata to: "{0:s}"...'.format(
                this_metafile_name
            ))
            neural_net._write_metafile(
                dill_file_name=this_metafile_name, use_partial_grids=True,
                num_epochs=100, num_training_batches_per_epoch=100,
                training_option_dict=dummy_option_dict,
                num_validation_batches_per_epoch=100,
                validation_option_dict=dummy_option_dict,
                do_early_stopping=True, plateau_lr_multiplier=0.6,
                loss_function_name=LOSS_FUNCTION_NAME,
                metric_names=METRIC_NAMES,
                mask_matrix=partial_mask_matrix,
                full_mask_matrix=full_mask_matrix
            )


if __name__ == '__main__':
    _run()
