"""Creates U-net with fractions skill score (FSS) as loss function."""

import sys
import os.path
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_architecture
import file_system_utils
import neural_net
import custom_losses

# if 'hfe' in socket.gethostname():
#     HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
# else:
#     HOME_DIR_NAME = os.path.expanduser('~')

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
MODEL_FILE_NAME = (
    '{0:s}/ml4convection_models/first_u_net_with_fss/template/model.h5'
).format(HOME_DIR_NAME)

ARCHITECTURE_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([220, 230, 7], dtype=int),
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.)
}

LOSS_FUNCTION = custom_losses.fractions_skill_score(
    half_window_size_px=3, use_as_loss_function=True
)


def _run():
    """Main method."""

    file_system_utils.mkdir_recursive_if_necessary(file_name=MODEL_FILE_NAME)
    model_object = u_net_architecture.create_model(
        option_dict=ARCHITECTURE_OPTION_DICT, loss_function=LOSS_FUNCTION
    )

    print('Writing model to: "{0:s}"...'.format(MODEL_FILE_NAME))
    model_object.save(
        filepath=MODEL_FILE_NAME, overwrite=True, include_optimizer=True
    )

    metafile_name = neural_net.find_metafile(
        model_file_name=MODEL_FILE_NAME, raise_error_if_missing=False
    )
    dummy_option_dict = neural_net.DEFAULT_GENERATOR_OPTION_DICT

    print('Writing metadata to: "{0:s}"...'.format(metafile_name))
    neural_net._write_metafile(
        dill_file_name=metafile_name, num_epochs=100,
        num_training_batches_per_epoch=100,
        training_option_dict=dummy_option_dict,
        num_validation_batches_per_epoch=100,
        validation_option_dict=dummy_option_dict,
        do_early_stopping=True, plateau_lr_multiplier=0.6,
        class_weights=None, fss_half_window_size_px=3
    )


if __name__ == '__main__':
    _run()
