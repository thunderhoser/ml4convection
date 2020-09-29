"""Finds event (convection) frequency for given dilation distance."""

import argparse
import numpy
from ml4convection.io import radar_io
from ml4convection.io import example_io
from ml4convection.io import twb_satellite_io
from ml4convection.utils import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

TARGET_DIR_ARG_NAME = 'input_target_dir_name'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_px'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'

TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with target values.  Files therein will be '
    'found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
DILATION_DISTANCE_HELP_STRING = 'Dilation distance for convective pixels.'
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will include grids for all days in the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=True,
    help=DILATION_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)


def _run(top_target_dir_name, first_date_string, last_date_string,
         dilation_distance_px):
    """Finds event (convection) frequency for given dilation distance.

    This is effectively the main method.

    :param top_target_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param dilation_distance_px: Same.
    """

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string, raise_error_if_any_missing=False
    )

    first_target_dict = example_io.read_target_file(target_file_names[0])
    mask_file_name = first_target_dict[example_io.MASK_FILE_KEY]

    print('Reading mask from: "{0:s}"...'.format(mask_file_name))
    mask_dict = radar_io.read_mask_file(mask_file_name)
    mask_dict = radar_io.expand_to_satellite_grid(any_radar_dict=mask_dict)

    num_target_latitudes = len(first_target_dict[example_io.LATITUDES_KEY])
    num_full_latitudes = len(twb_satellite_io.GRID_LATITUDES_DEG_N)
    downsampling_factor = int(numpy.floor(
        float(num_full_latitudes) / num_target_latitudes
    ))

    if downsampling_factor > 1:
        print('Downsampling mask to {0:d}x spatial resolution...'.format(
            downsampling_factor
        ))
        mask_dict = radar_io.downsample_in_space(
            any_radar_dict=mask_dict, downsampling_factor=4
        )

    print('Eroding mask with erosion distance = {0:f} pixels...'.format(
        dilation_distance_px
    ))
    mask_matrix = evaluation.erode_binary_matrix(
        binary_matrix=mask_dict[radar_io.MASK_MATRIX_KEY],
        buffer_distance_px=dilation_distance_px
    )
    mask_matrix = mask_matrix.astype(bool)

    print(SEPARATOR_STRING)

    num_pixels = 0
    num_convective_pixels = 0

    for this_file_name in target_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_target_dict = example_io.read_target_file(this_file_name)
        this_num_times = len(this_target_dict[example_io.VALID_TIMES_KEY])

        for i in range(this_num_times):
            this_target_matrix = evaluation.dilate_binary_matrix(
                binary_matrix=
                this_target_dict[example_io.TARGET_MATRIX_KEY][i, ...],
                buffer_distance_px=dilation_distance_px
            )

            this_target_matrix[mask_matrix == False] = -1
            num_pixels += numpy.sum(this_target_matrix >= 0)
            num_convective_pixels += numpy.sum(this_target_matrix == 1)

            print((
                'Number of convective pixels = {0:d} of {1:d} = {2:10f}'
            ).format(
                num_convective_pixels, num_pixels,
                float(num_convective_pixels) / num_pixels
            ))

        if this_file_name == target_file_names[-1]:
            print(SEPARATOR_STRING)
        else:
            print(MINOR_SEPARATOR_STRING)

    print((
        'Number of convective pixels = {0:d} of {1:d} = {2:10f}'
    ).format(
        num_convective_pixels, num_pixels,
        float(num_convective_pixels) / num_pixels
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        dilation_distance_px=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME
        )
    )
