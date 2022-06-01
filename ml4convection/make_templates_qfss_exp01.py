"""Makes U-net templates for Experiment 1 with quantile-based FSS."""

import sys
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

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4convection_models/qfss_experiment01/templates'
).format(HOME_DIR_NAME)

FULL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/radar_mask_100km_omit-north.nc'
).format(HOME_DIR_NAME)

PARTIAL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/'
    'radar_mask_100km_omit-north_partial.nc'
).format(HOME_DIR_NAME)

FSS_WEIGHTS = numpy.linspace(1, 10, num=10, dtype=float)
CENTRAL_LOSS_FUNCTION_NAMES = [
    'fss_neigh0_weight{0:.10f}'.format(i) for i in FSS_WEIGHTS
]

QUANTILE_LEVEL_SETS = [
    numpy.linspace(0.01, 0.99, num=99),
    numpy.linspace(0.01, 0.99, num=99)[::2],
    numpy.linspace(0.01, 0.99, num=99)[::3],
    numpy.linspace(0.01, 0.99, num=99)[::4],
    numpy.linspace(0.01, 0.99, num=99)[::5],
    numpy.linspace(0.01, 0.99, num=99)[::6],
    numpy.linspace(0.01, 0.99, num=99)[::7],
    numpy.linspace(0.01, 0.99, num=99)[::8],
    numpy.linspace(0.01, 0.99, num=99)[::9]
]

QUANTILE_LEVEL_SETS = [
    numpy.concatenate((
        numpy.array([0.025, 0.25, 0.5, 0.75, 0.975, 0.99]), s
    ))
    for s in QUANTILE_LEVEL_SETS
]

QUANTILE_LEVEL_SETS = [numpy.sort(numpy.unique(s)) for s in QUANTILE_LEVEL_SETS]

DEFAULT_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([205, 205, 21], dtype=int),
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.CONV_LAYER_COUNTS_KEY: numpy.full(6, 2, dtype=int),
    u_net_architecture.OUTPUT_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 6,
    u_net_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 5,
    u_net_architecture.SKIP_DROPOUT_MC_FLAGS_KEY:
        [numpy.full(2, 0, dtype=bool)] * 5,
    u_net_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    u_net_architecture.PENULTIMATE_DROPOUT_MC_FLAG_KEY: False,
    u_net_architecture.OUTPUT_DROPOUT_RATE_KEY: 0.,
    u_net_architecture.OUTPUT_DROPOUT_MC_FLAG_KEY: False,
    u_net_architecture.L2_WEIGHT_KEY: 10 ** -5.5
}


def _run():
    """Makes U-net templates for Experiment 1 with quantile-based FSS.

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

    for i in range(len(CENTRAL_LOSS_FUNCTION_NAMES)):
        for j in range(len(QUANTILE_LEVEL_SETS)):
            print(SEPARATOR_STRING)

            loss_function = neural_net.get_metrics(
                metric_names=[CENTRAL_LOSS_FUNCTION_NAMES[i]],
                mask_matrix=partial_mask_matrix,
                use_as_loss_function=True
            )[0][0]

            this_model_object = u_net_architecture.create_qr_model_fancy(
                option_dict=DEFAULT_OPTION_DICT,
                central_loss_function=loss_function,
                mask_matrix=partial_mask_matrix,
                quantile_levels=QUANTILE_LEVEL_SETS[j],
                qfss_half_window_size_px=0
            )

            this_model_file_name = (
                '{0:s}/fss-weight={1:04.1f}_num-quantile-levels={2:03d}/'
                'model.h5'
            ).format(
                OUTPUT_DIR_NAME, FSS_WEIGHTS[i], len(QUANTILE_LEVEL_SETS[j])
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=this_model_file_name
            )

            print('Writing model to: "{0:s}"...'.format(
                this_model_file_name
            ))
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
                loss_function_name=CENTRAL_LOSS_FUNCTION_NAMES[i],
                metric_names=None, quantile_levels=QUANTILE_LEVEL_SETS[j],
                qfss_half_window_size_px=0,
                mask_matrix=partial_mask_matrix,
                full_mask_matrix=full_mask_matrix
            )


if __name__ == '__main__':
    _run()
