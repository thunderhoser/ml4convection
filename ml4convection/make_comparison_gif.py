"""USE ONCE AND DESTROY."""

import os
import sys
import glob

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import imagemagick_utils

INPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4convection_models/'
    'experiment11/batch-size=64_l2-weight=0.0000100000_num-lag-times=2/'
    'validation/full_grids/prediction_plots'
)
INPUT_FILE_PATTERN = '{0:s}/predictions_2017-06-03-*00.jpg'.format(
    INPUT_DIR_NAME
)
OUTPUT_DIR_NAME = '{0:s}/gif'.format(INPUT_DIR_NAME)


def _run():
    """USE ONCE AND DESTROY."""

    orig_image_file_names = glob.glob(INPUT_FILE_PATTERN)
    orig_image_file_names.sort()

    cropped_image_file_names = [
        f.replace(INPUT_DIR_NAME, OUTPUT_DIR_NAME)
        for f in orig_image_file_names
    ]

    num_times = len(orig_image_file_names)

    for i in range(num_times):
        this_command_string = (
            'convert -crop 1444x2161+1103+788 {0:s} {1:s}'
        ).format(orig_image_file_names[i], cropped_image_file_names[i])

        print('Cropping to: "{0:s}"...'.format(cropped_image_file_names[i]))
        os.system(this_command_string)

    gif_file_name = '{0:s}/predictions_2017-06-03.gif'.format(OUTPUT_DIR_NAME)
    print('Making GIF: "{0:s}"...'.format(gif_file_name))

    imagemagick_utils.create_gif(
        input_file_names=cropped_image_file_names,
        output_file_name=gif_file_name,
        num_seconds_per_frame=1., resize_factor=0.5
    )


if __name__ == '__main__':
    _run()
