"""Creates mask for radar data.

This mask censors out regions with too few radar observations.
"""

import copy
import glob
import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4convection.scripts import count_radar_observations as count_obs

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

COUNT_DIR_ARG_NAME = 'input_count_dir_name'
MAX_HEIGHT_ARG_NAME = 'max_height_m_asl'
MIN_OBSERVATIONS_ARG_NAME = 'min_observations'
MIN_HEIGHT_FRACTION_ARG_NAME = 'min_height_fraction'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

COUNT_DIR_HELP_STRING = (
    'Name of directory with count files.  Names must be in format "counts*.nc".'
    '  Files will be read by `count_radar_observations.read_count_file`.'
)
MAX_HEIGHT_HELP_STRING = (
    'Max height (metres above sea level).  Radar observations above this height'
    ' will not be considered.'
)
MIN_OBSERVATIONS_HELP_STRING = (
    'Minimum number of observations.  Each grid column (horizontal location) '
    'will be included if it contains >= N observations at >= fraction f of '
    'heights up to `{0:s}`, where N = `{1:s}` and f = `{2:s}`.'
).format(
    MAX_HEIGHT_ARG_NAME, MIN_OBSERVATIONS_ARG_NAME, MIN_HEIGHT_FRACTION_ARG_NAME
)
MIN_HEIGHT_FRACTION_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_OBSERVATIONS_ARG_NAME
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Mask will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + COUNT_DIR_ARG_NAME, type=str, required=True,
    help=COUNT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_HEIGHT_ARG_NAME, type=float, required=False, default=9000,
    help=MAX_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_OBSERVATIONS_ARG_NAME, type=int, required=False, default=10000,
    help=MIN_OBSERVATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HEIGHT_FRACTION_ARG_NAME, type=float, required=False,
    default=0.5, help=MIN_HEIGHT_FRACTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(count_dir_name, max_height_m_asl, min_observations,
         min_height_fraction, output_file_name):
    """Creates mask for radar data.

    This is effectively the main method.

    :param count_dir_name: See documentation at top of file.
    :param max_height_m_asl: Same.
    :param min_observations: Same.
    :param min_height_fraction: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(max_height_m_asl, 5000.)
    error_checking.assert_is_geq(min_observations, 1)
    error_checking.assert_is_greater(min_height_fraction, 0.)
    error_checking.assert_is_leq(min_height_fraction, 1.)

    count_file_names = glob.glob('{0:s}/counts*.nc'.format(count_dir_name))

    count_matrix = None
    metadata_dict = None
    max_height_index = None

    for this_file_name in count_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_count_dict = count_obs.read_count_file(this_file_name)

        if count_matrix is None:
            count_matrix = this_count_dict[count_obs.OBSERVATION_COUNT_KEY] + 0
            del this_count_dict[count_obs.OBSERVATION_COUNT_KEY]
            metadata_dict = copy.deepcopy(this_count_dict)

            all_heights_m_asl = metadata_dict[count_obs.HEIGHTS_KEY]
            these_indices = numpy.where(
                all_heights_m_asl >= max_height_m_asl
            )[0]

            if len(these_indices) == 0:
                max_height_index = len(all_heights_m_asl) - 1
            else:
                max_height_index = these_indices[0]

            metadata_dict[count_obs.HEIGHTS_KEY] = (
                all_heights_m_asl[:(max_height_index + 1)]
            )
            count_matrix = count_matrix[..., :(max_height_index + 1)]
        else:
            count_matrix += this_count_dict[count_obs.OBSERVATION_COUNT_KEY][
                ..., :(max_height_index + 1)
            ]

    print(SEPARATOR_STRING)

    print('Finding grid cells with >= {0:d} observations...'.format(
        min_observations
    ))
    fractional_exceedance_matrix = numpy.mean(
        count_matrix >= min_observations, axis=-1
    )

    num_heights = count_matrix.shape[2]

    print((
        'Finding grid columns with >= {0:d} observations at >= {1:f} of {2:d} '
        'heights...'
    ).format(
        min_observations, min_height_fraction * num_heights, num_heights
    ))
    mask_matrix = fractional_exceedance_matrix >= min_height_fraction

    print('{0:d} of {1:d} grid columns are unmasked!'.format(
        numpy.sum(mask_matrix), mask_matrix.size
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        count_dir_name=getattr(INPUT_ARG_OBJECT, COUNT_DIR_ARG_NAME),
        max_height_m_asl=getattr(INPUT_ARG_OBJECT, MAX_HEIGHT_ARG_NAME),
        min_observations=getattr(INPUT_ARG_OBJECT, MIN_OBSERVATIONS_ARG_NAME),
        min_height_fraction=getattr(
            INPUT_ARG_OBJECT, MIN_HEIGHT_FRACTION_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
