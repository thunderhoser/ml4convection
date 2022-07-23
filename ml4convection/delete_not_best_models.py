"""USE ONCE AND DESTROY."""

import os
import glob
import shutil
import argparse
import numpy

EXPERIMENT_DIR_ARG_NAME = 'input_experiment_dir_name'
EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)


def _run(top_experiment_dir_name):
    """USE ONCE AND DESTROY.

    :param top_experiment_dir_name: See documentation at top of file.
    """

    model_dir_names = glob.glob('{0:s}/*'.format(top_experiment_dir_name))

    for this_model_dir_name in model_dir_names:
        model_file_names = glob.glob(
            '{0:s}/model*.h5'.format(this_model_dir_name)
        )
        model_file_names.sort()
        if len(model_file_names) <= 1:
            continue

        validation_losses = numpy.full(len(model_file_names), numpy.nan)

        for i in range(len(model_file_names)):
            pathless_model_file_name = os.path.split(model_file_names[i])[-1]
            extensionless_model_file_name = os.path.splitext(
                pathless_model_file_name
            )[0]

            loss_string = extensionless_model_file_name.split('_')[-1]
            assert loss_string.startswith('val-loss=')
            validation_losses[i] = float(loss_string.replace('val-loss=', ''))

        min_index = numpy.argmin(validation_losses)

        for i in range(len(model_file_names)):
            if i == min_index:
                print(model_file_names[i])
                continue

            # print('Deleting "{0:s}"...'.format(model_file_names[i]))
            # os.remove(model_file_names[i])
            #
            # print('Deleting "{0:s}"...'.format(
            #     model_file_names[i].replace('.h5', '')
            # ))
            # shutil.rmtree(model_file_names[i].replace('.h5', ''))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_experiment_dir_name=getattr(
            INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME
        )
    )
