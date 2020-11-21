"""Runs echo classification to separate convective from non-convective."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import echo_classification as echo_classifn
import radar_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT_FOR_MESSAGES = '%Y-%m-%d-%H%M%S'

# TODO(thunderhoser): 1000-metre height spacing!!

INPUT_DIR_ARG_NAME = 'input_radar_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
PEAKEDNESS_NEIGH_ARG_NAME = 'peakedness_neigh_metres'
MAX_PEAKEDNESS_HEIGHT_ARG_NAME = 'max_peakedness_height_m_asl'
MIN_HEIGHT_FRACTION_ARG_NAME = 'min_height_fraction_for_peakedness'
THIN_HEIGHT_GRID_ARG_NAME = 'thin_height_grid'
MIN_ECHO_TOP_ARG_NAME = 'min_echo_top_m_asl'
ECHO_TOP_LEVEL_ARG_NAME = 'echo_top_level_dbz'
MIN_REFL_CRITERION1_ARG_NAME = 'min_refl_criterion1_dbz'
MIN_REFL_CRITERION5_ARG_NAME = 'min_refl_criterion5_dbz'
MIN_REFL_AML_ARG_NAME = 'min_reflectivity_aml_dbz'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with radar data.  Files therein will be '
    'found by `radar_io.find_file` and read by '
    '`radar_io.read_reflectivity_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will do echo classification for all days in the'
    ' period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

PEAKEDNESS_NEIGH_HELP_STRING = (
    'Neighbourhood radius for peakedness calculations.'
)
MAX_PEAKEDNESS_HEIGHT_HELP_STRING = (
    'Max height (metres above sea level) for peakedness calculations.'
)
MIN_HEIGHT_FRACTION_HELP_STRING = (
    'Minimum fraction of heights that exceed peakedness threshold.  At each '
    'horizontal location, at least this fraction of heights must exceed the '
    'threshold.'
)
THIN_HEIGHT_GRID_HELP_STRING = (
    'Boolean flag.  If 1, will thin height grid to 1000-metre spacing '
    'throughout.  If 0, will keep extra heights near the surface.'
)
MIN_ECHO_TOP_HELP_STRING = (
    'Minimum echo top (metres above sea level), used for criterion 3.'
)
ECHO_TOP_LEVEL_HELP_STRING = (
    'Critical reflectivity (used to compute echo top for criterion 3).'
)
MIN_REFL_CRITERION1_HELP_STRING = (
    'Minimum composite (column-max) reflectivity for criterion 1.  To exclude '
    'this criterion, make the value negative.'
)
MIN_REFL_CRITERION5_HELP_STRING = (
    'Minimum composite reflectivity for criterion 5.'
)
MIN_REFL_AML_HELP_STRING = (
    'Minimum composite reflectivity above melting level, used for criterion 2.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Echo-classification files will be '
    'written by `radar_io.write_echo_classifn_file`, to exact locations therein'
    ' determined by `radar_io.find_file`.'
)

DEFAULT_OPTION_DICT = echo_classifn.DEFAULT_OPTION_DICT

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PEAKEDNESS_NEIGH_ARG_NAME, type=float, required=False,
    default=DEFAULT_OPTION_DICT[echo_classifn.PEAKEDNESS_NEIGH_KEY],
    help=PEAKEDNESS_NEIGH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PEAKEDNESS_HEIGHT_ARG_NAME, type=float, required=False,
    default=DEFAULT_OPTION_DICT[echo_classifn.MAX_PEAKEDNESS_HEIGHT_KEY],
    help=MAX_PEAKEDNESS_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HEIGHT_FRACTION_ARG_NAME, type=float, required=False,
    default=0.6, help=MIN_HEIGHT_FRACTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + THIN_HEIGHT_GRID_ARG_NAME, type=int, required=False, default=1,
    help=THIN_HEIGHT_GRID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_ECHO_TOP_ARG_NAME, type=int, required=False,
    default=DEFAULT_OPTION_DICT[echo_classifn.MIN_ECHO_TOP_KEY],
    help=MIN_ECHO_TOP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_TOP_LEVEL_ARG_NAME, type=float, required=False,
    default=DEFAULT_OPTION_DICT[echo_classifn.ECHO_TOP_LEVEL_KEY],
    help=ECHO_TOP_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_REFL_CRITERION1_ARG_NAME, type=float, required=False,
    default=DEFAULT_OPTION_DICT[
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION1_KEY
    ],
    help=MIN_REFL_CRITERION1_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_REFL_CRITERION5_ARG_NAME, type=float, required=False,
    default=DEFAULT_OPTION_DICT[
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION5_KEY
    ],
    help=MIN_REFL_CRITERION5_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_REFL_AML_ARG_NAME, type=float, required=False,
    default=DEFAULT_OPTION_DICT[echo_classifn.MIN_COMPOSITE_REFL_AML_KEY],
    help=MIN_REFL_AML_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run_for_one_day(
        radar_file_name, peakedness_neigh_metres, max_peakedness_height_m_asl,
        min_height_fraction_for_peakedness, thin_height_grid,
        min_echo_top_m_asl, echo_top_level_dbz, min_refl_criterion1_dbz,
        min_refl_criterion5_dbz, min_reflectivity_aml_dbz, output_file_name):
    """Runs echo classification for one day.

    :param radar_file_name: Path to input file (will be read by
        `radar_io.read_reflectivity_file`).
    :param peakedness_neigh_metres: See documentation at top of file.
    :param max_peakedness_height_m_asl: Same.
    :param min_height_fraction_for_peakedness: Same.
    :param thin_height_grid: Same.
    :param min_echo_top_m_asl: Same.
    :param echo_top_level_dbz: Same.
    :param min_refl_criterion1_dbz: Same.
    :param min_refl_criterion5_dbz: Same.
    :param min_reflectivity_aml_dbz: Same.
    :param output_file_name: Path to output file (will be read by
        `radar_io.write_echo_classifn_file`).
    :raises: ValueError: if radar file does not contain 3-D data.
    """

    option_dict = {
        echo_classifn.PEAKEDNESS_NEIGH_KEY: peakedness_neigh_metres,
        echo_classifn.MAX_PEAKEDNESS_HEIGHT_KEY: max_peakedness_height_m_asl,
        echo_classifn.MIN_HEIGHT_FRACTION_KEY:
            min_height_fraction_for_peakedness,
        radar_io.THIN_HEIGHT_GRID_KEY: thin_height_grid,
        echo_classifn.HALVE_RESOLUTION_KEY: False,
        echo_classifn.MIN_ECHO_TOP_KEY: min_echo_top_m_asl,
        echo_classifn.ECHO_TOP_LEVEL_KEY: echo_top_level_dbz,
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION1_KEY:
            min_refl_criterion1_dbz,
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION5_KEY:
            min_refl_criterion5_dbz,
        echo_classifn.MIN_COMPOSITE_REFL_AML_KEY: min_reflectivity_aml_dbz
    }

    print('Reading data from: "{0:s}"...'.format(radar_file_name))
    radar_dict = radar_io.read_reflectivity_file(
        netcdf_file_name=radar_file_name, fill_nans=True
    )

    heights_m_asl = radar_dict[radar_io.HEIGHTS_KEY]
    if len(heights_m_asl) == 1:
        raise ValueError('File should contain 3-D data.')

    if thin_height_grid:
        print('Thinning height grid to 1000-metre spacing...')

        remainders = numpy.mod(heights_m_asl, 1000)
        good_indices = numpy.where(remainders < 1)[0]

        radar_dict[radar_io.HEIGHTS_KEY] = (
            radar_dict[radar_io.HEIGHTS_KEY][good_indices]
        )
        radar_dict[radar_io.REFLECTIVITY_KEY] = (
            radar_dict[radar_io.REFLECTIVITY_KEY][..., good_indices]
        )

    latitudes_deg_n = radar_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = radar_dict[radar_io.LONGITUDES_KEY]
    valid_times_unix_sec = radar_dict[radar_io.VALID_TIMES_KEY]

    grid_metadata_dict = {
        echo_classifn.MIN_LATITUDE_KEY: numpy.min(latitudes_deg_n),
        echo_classifn.LATITUDE_SPACING_KEY:
            numpy.diff(latitudes_deg_n[:2])[0],
        echo_classifn.MIN_LONGITUDE_KEY: numpy.min(longitudes_deg_e),
        echo_classifn.LONGITUDE_SPACING_KEY:
            numpy.diff(longitudes_deg_e[:2])[0],
        echo_classifn.HEIGHTS_KEY: radar_dict[radar_io.HEIGHTS_KEY]
    }

    reflectivity_matrix_dbz = radar_dict[radar_io.REFLECTIVITY_KEY]
    convective_flag_matrix = numpy.full(
        reflectivity_matrix_dbz.shape[:3], False, dtype=bool
    )
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        convective_flag_matrix[i, ...], option_dict = (
            echo_classifn.find_convective_pixels(
                reflectivity_matrix_dbz=reflectivity_matrix_dbz[i, ...],
                grid_metadata_dict=grid_metadata_dict,
                valid_time_unix_sec=valid_times_unix_sec[i],
                option_dict=option_dict
            )
        )

        this_time_string = time_conversion.unix_sec_to_string(
            valid_times_unix_sec[i], TIME_FORMAT_FOR_MESSAGES
        )
        print((
            '\nNumber of convective pixels at {0:s} = {1:d} of {2:d}\n'
        ).format(
            this_time_string, numpy.sum(convective_flag_matrix[i, ...]),
            convective_flag_matrix[i, ...].size
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    radar_io.write_echo_classifn_file(
        netcdf_file_name=output_file_name,
        convective_flag_matrix=convective_flag_matrix,
        latitudes_deg_n=latitudes_deg_n, longitudes_deg_e=longitudes_deg_e,
        valid_times_unix_sec=valid_times_unix_sec, option_dict=option_dict
    )

    radar_io.compress_file(output_file_name)
    os.remove(output_file_name)


def _run(top_radar_dir_name, first_date_string, last_date_string,
         peakedness_neigh_metres, max_peakedness_height_m_asl,
         min_height_fraction_for_peakedness, thin_height_grid,
         min_echo_top_m_asl, echo_top_level_dbz, min_refl_criterion1_dbz,
         min_refl_criterion5_dbz, min_reflectivity_aml_dbz,
         top_output_dir_name):
    """Runs echo classification to separate convective from non-convective.

    This is effectively the main method.

    :param top_radar_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param peakedness_neigh_metres: Same.
    :param max_peakedness_height_m_asl: Same.
    :param min_height_fraction_for_peakedness: Same.
    :param thin_height_grid: Same.
    :param min_echo_top_m_asl: Same.
    :param echo_top_level_dbz: Same.
    :param min_refl_criterion1_dbz: Same.
    :param min_refl_criterion5_dbz: Same.
    :param min_reflectivity_aml_dbz: Same.
    :param top_output_dir_name: Same.
    """

    if min_refl_criterion1_dbz <= 0:
        min_refl_criterion1_dbz = None

    radar_file_names = radar_io.find_many_files(
        top_directory_name=top_radar_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        file_type_string=radar_io.REFL_TYPE_STRING,
        raise_error_if_any_missing=False
    )

    for this_input_file_name in radar_file_names:
        this_output_file_name = radar_io.find_file(
            top_directory_name=top_output_dir_name,
            valid_date_string=radar_io.file_name_to_date(this_input_file_name),
            file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        _run_for_one_day(
            radar_file_name=this_input_file_name,
            peakedness_neigh_metres=peakedness_neigh_metres,
            max_peakedness_height_m_asl=max_peakedness_height_m_asl,
            min_height_fraction_for_peakedness=
            min_height_fraction_for_peakedness,
            thin_height_grid=thin_height_grid,
            min_echo_top_m_asl=min_echo_top_m_asl,
            echo_top_level_dbz=echo_top_level_dbz,
            min_refl_criterion1_dbz=min_refl_criterion1_dbz,
            min_refl_criterion5_dbz=min_refl_criterion5_dbz,
            min_reflectivity_aml_dbz=min_reflectivity_aml_dbz,
            output_file_name=this_output_file_name
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        peakedness_neigh_metres=getattr(
            INPUT_ARG_OBJECT, PEAKEDNESS_NEIGH_ARG_NAME
        ),
        max_peakedness_height_m_asl=getattr(
            INPUT_ARG_OBJECT, MAX_PEAKEDNESS_HEIGHT_ARG_NAME
        ),
        min_height_fraction_for_peakedness=getattr(
            INPUT_ARG_OBJECT, MIN_HEIGHT_FRACTION_ARG_NAME
        ),
        thin_height_grid=bool(getattr(
            INPUT_ARG_OBJECT, THIN_HEIGHT_GRID_ARG_NAME
        )),
        min_echo_top_m_asl=getattr(INPUT_ARG_OBJECT, MIN_ECHO_TOP_ARG_NAME),
        echo_top_level_dbz=getattr(INPUT_ARG_OBJECT, ECHO_TOP_LEVEL_ARG_NAME),
        min_refl_criterion1_dbz=getattr(
            INPUT_ARG_OBJECT, MIN_REFL_CRITERION1_ARG_NAME
        ),
        min_refl_criterion5_dbz=getattr(
            INPUT_ARG_OBJECT, MIN_REFL_CRITERION5_ARG_NAME
        ),
        min_reflectivity_aml_dbz=getattr(
            INPUT_ARG_OBJECT, MIN_REFL_AML_ARG_NAME
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
