"""Deletes unused models.

USE ONCE AND DESTROY.
"""

import os
import glob
import shutil
import argparse
import numpy

MODEL_DIR_ARG_NAME = 'model_dir_name'
MODEL_DIR_HELP_STRING = 'Name of directory with model files, one per epoch.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DIR_ARG_NAME, type=str, required=True,
    help=MODEL_DIR_HELP_STRING
)


def _run(model_dir_name):
    """Deletes unused models.

    This is effectively the main method.

    :param model_dir_name: See documentation at top of file.
    """

    model_file_pattern = '{0:s}/model_epoch=*_val-loss=*.h5'.format(
        model_dir_name
    )
    model_file_names = glob.glob(model_file_pattern)

    validation_losses = numpy.array([
        float(f.split('.')[-2].split('=')[-1])
        for f in model_file_names
    ])
    best_model_index = numpy.argmin(validation_losses)
    num_models = len(model_file_names)

    for i in range(num_models):
        if i == best_model_index:
            continue

        print('Deleting file: "{0:s}"...'.format(model_file_names[i]))
        os.remove(model_file_names[i])

    best_model_file_name = '{0:s}/model.h5'.format(model_dir_name)

    print('Renaming file: "{0:s}" to "{1:s}"...'.format(
        model_file_names[best_model_index], best_model_file_name
    ))
    shutil.move(model_file_names[best_model_index], best_model_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_dir_name=getattr(INPUT_ARG_OBJECT, MODEL_DIR_ARG_NAME)
    )
