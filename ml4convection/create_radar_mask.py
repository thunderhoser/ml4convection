"""Creates mask for radar data.

This mask censors out regions with too few radar observations.
"""

import os
import sys
import copy
import glob
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import radar_io
import count_radar_observations as count_obs

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


def _increment_counts_one_file(count_file_name, count_matrix, max_height_m_asl):
    """Increments counts, based on one count file.

    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid

    :param count_file_name: Path to input file (will be read by
        `count_radar_observations.read_count_file`).
    :param count_matrix: M-by-N-by-H numpy array of observation counts.
        If None, will be created on the fly.
    :param max_height_m_asl: [used only if `count_matrix is None`]
        See documentation at top of file.
    :return: count_matrix: Same as input but with new (incremented) values.
    :return: metadata_dict: Dictionary returned by
        `count_radar_observations.read_count_file`, excluding counts.
    """

    print('Reading data from: "{0:s}"...'.format(count_file_name))
    count_dict = count_obs.read_count_file(count_file_name)

    new_count_matrix = count_dict[count_obs.OBSERVATION_COUNT_KEY]
    del count_dict[count_obs.OBSERVATION_COUNT_KEY]
    metadata_dict = copy.deepcopy(count_dict)

    if count_matrix is None:
        these_indices = numpy.where(
            metadata_dict[count_obs.HEIGHTS_KEY] >= max_height_m_asl
        )[0]

        if len(these_indices) == 0:
            max_height_index = len(metadata_dict[count_obs.HEIGHTS_KEY]) - 1
        else:
            max_height_index = these_indices[0]
    else:
        max_height_index = count_matrix.shape[-1] - 1

    new_count_matrix = new_count_matrix[..., :(max_height_index + 1)]

    if count_matrix is None:
        count_matrix = new_count_matrix + 0
    else:
        count_matrix += new_count_matrix

    return count_matrix, metadata_dict


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

    option_dict = {
        radar_io.MAX_MASK_HEIGHT_KEY: max_height_m_asl,
        radar_io.MIN_OBSERVATIONS_KEY: min_observations,
        radar_io.MIN_HEIGHT_FRACTION_FOR_MASK_KEY: min_height_fraction
    }
    radar_io.check_mask_options(option_dict)

    count_file_names = glob.glob('{0:s}/counts*.nc'.format(count_dir_name))
    count_file_names.sort()

    count_matrix = None
    metadata_dict = None

    for this_file_name in count_file_names:
        count_matrix, metadata_dict = _increment_counts_one_file(
            count_file_name=this_file_name, count_matrix=count_matrix,
            max_height_m_asl=max_height_m_asl
        )

    print(SEPARATOR_STRING)
    num_heights = count_matrix.shape[2]

    print('Finding grid cells with >= {0:d} observations...'.format(
        min_observations
    ))
    fractional_exceedance_matrix = numpy.mean(
        count_matrix >= min_observations, axis=-1
    )

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

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    radar_io.write_mask_file(
        netcdf_file_name=output_file_name, mask_matrix=mask_matrix,
        latitudes_deg_n=metadata_dict[count_obs.LATITUDES_KEY],
        longitudes_deg_e=metadata_dict[count_obs.LONGITUDES_KEY],
        option_dict=option_dict
    )


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
