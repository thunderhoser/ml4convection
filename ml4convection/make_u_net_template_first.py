"""Makes U-net template for first experiment."""

import sys
import os.path
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_architecture
import architecture_utils
import file_system_utils
import neural_net

# if 'hfe' in socket.gethostname():
#     HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
# else:
#     HOME_DIR_NAME = os.path.expanduser('~')

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
MODEL_FILE_NAME = (
    '{0:s}/ml4convection_models/first_u_net/template/model.h5'
).format(HOME_DIR_NAME)

ARCHITECTURE_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([881, 921, 7], dtype=int)
}


def _run():
    """Main method."""

    file_system_utils.mkdir_recursive_if_necessary(file_name=MODEL_FILE_NAME)
    model_object = u_net_architecture.create_model(ARCHITECTURE_OPTION_DICT)

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
        do_early_stopping=True, plateau_lr_multiplier=0.6
    )


if __name__ == '__main__':
    _run()
