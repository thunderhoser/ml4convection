"""Counts radar observations at each grid cell."""

import argparse
import numpy
import netCDF4
from gewittergefahr.gg_utils import file_system_utils
from ml4convection.io import radar_io

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
HEIGHT_DIMENSION_KEY = 'height'

LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
HEIGHTS_KEY = 'heights_m_asl'
OBSERVATION_COUNT_KEY = 'count_matrix'

REFLECTIVITY_DIR_ARG_NAME = 'input_reflectivity_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

REFLECTIVITY_DIR_HELP_STRING = (
    'Name of directory with reflectivity data.  Files therein will be found by '
    '`radar_io.find_file` and read by `radar_io.read_reflectivity_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will count observations for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (NetCDF).  Results will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + REFLECTIVITY_DIR_ARG_NAME, type=str, required=True,
    help=REFLECTIVITY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _increment_counts_one_day(
        reflectivity_file_name, count_matrix, expected_latitudes_deg_n,
        expected_longitudes_deg_e, expected_heights_m_asl):
    """Increments observation count at each grid cell, based on one day of data.

    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid

    :param reflectivity_file_name: Path to input file (will be read by
        `radar_io.read_reflectivity_file`).
    :param count_matrix: M-by-N-by-H numpy array of observation counts.  If
        None, a new one will be created.
    :param expected_latitudes_deg_n: length-M numpy array of expected latitudes
        (deg N).  If None, latitudes will not be verified.
    :param expected_longitudes_deg_e: length-N numpy array of expected
        longitudes (deg E).  May be None.
    :param expected_heights_m_asl: length-H numpy array of expected heights
        (metres above sea level).  May be None.
    :return: count_matrix: Updated version of input array.
    :return: latitudes_deg_n: Same as input but read from new file.
    :return: longitudes_deg_e: Same as input but read from new file.
    :return: heights_m_asl: Same as input but read from new file.
    """

    print('Reading data from: "{0:s}"...'.format(reflectivity_file_name))
    reflectivity_dict = radar_io.read_reflectivity_file(
        netcdf_file_name=reflectivity_file_name, fill_nans=False
    )

    if expected_latitudes_deg_n is not None:
        assert numpy.allclose(
            expected_latitudes_deg_n, reflectivity_dict[radar_io.LATITUDES_KEY],
            atol=TOLERANCE
        )

    if expected_longitudes_deg_e is not None:
        assert numpy.allclose(
            expected_longitudes_deg_e,
            reflectivity_dict[radar_io.LONGITUDES_KEY],
            atol=TOLERANCE
        )

    if expected_heights_m_asl is not None:
        assert numpy.allclose(
            expected_heights_m_asl, reflectivity_dict[radar_io.HEIGHTS_KEY],
            atol=TOLERANCE
        )

    new_count_matrix = numpy.invert(
        numpy.isnan(reflectivity_dict[radar_io.REFLECTIVITY_KEY])
    ).astype(int)

    if count_matrix is None:
        count_matrix = new_count_matrix + 0
    else:
        count_matrix = count_matrix + new_count_matrix

    return (
        count_matrix,
        reflectivity_dict[radar_io.LATITUDES_KEY],
        reflectivity_dict[radar_io.LONGITUDES_KEY],
        reflectivity_dict[radar_io.HEIGHTS_KEY]
    )


def _write_count_file(netcdf_file_name, count_matrix, latitudes_deg_n,
                      longitudes_deg_e, heights_m_asl):
    """Writes gridded observation counts to file.

    :param netcdf_file_name: Path to output file.
    :param count_matrix: See doc for `_increment_counts_one_day`.
    :param latitudes_deg_n: Same.
    :param longitudes_deg_e: Same.
    :param heights_m_asl: Same.
    """

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(ROW_DIMENSION_KEY, len(latitudes_deg_n))
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, len(longitudes_deg_e))
    dataset_object.createDimension(HEIGHT_DIMENSION_KEY, len(heights_m_asl))

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32, dimensions=COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = longitudes_deg_e

    dataset_object.createVariable(
        HEIGHTS_KEY, datatype=numpy.float32, dimensions=HEIGHT_DIMENSION_KEY
    )
    dataset_object.variables[HEIGHTS_KEY][:] = heights_m_asl

    these_dim = (ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY, HEIGHT_DIMENSION_KEY)
    dataset_object.createVariable(
        OBSERVATION_COUNT_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[OBSERVATION_COUNT_KEY][:] = count_matrix

    dataset_object.close()


def _run(top_reflectivity_dir_name, first_date_string, last_date_string,
         output_file_name):
    """Counts radar observations at each grid cell.

    This is effectively the main method.

    :param top_reflectivity_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param output_file_name: Same.
    """

    reflectivity_file_names = radar_io.find_many_files(
        top_directory_name=top_reflectivity_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        file_type_string=radar_io.REFL_TYPE_STRING,
        raise_error_if_any_missing=False
    )

    count_matrix = None
    latitudes_deg_n = None
    longitudes_deg_e = None
    heights_m_asl = None

    for this_file_name in reflectivity_file_names:
        count_matrix, latitudes_deg_n, longitudes_deg_e, heights_m_asl = (
            _increment_counts_one_day(
                reflectivity_file_name=this_file_name,
                count_matrix=count_matrix,
                expected_latitudes_deg_n=latitudes_deg_n,
                expected_longitudes_deg_e=longitudes_deg_e,
                expected_heights_m_asl=heights_m_asl
            )
        )

    print(SEPARATOR_STRING)
    num_heights = len(heights_m_asl)

    for k in range(num_heights):
        print((
            'Number of grid cells with > 0 observations at {0:d} metres ASL = '
            '{1:d} of {2:d}'
        ).format(
            int(numpy.round(heights_m_asl[k])),
            numpy.sum(count_matrix[..., k] > 0), count_matrix[..., k].size
        ))

    print(SEPARATOR_STRING)
    print('Writing results to file: "{0:s}"...'.format(output_file_name))

    _write_count_file(
        netcdf_file_name=output_file_name, count_matrix=count_matrix,
        latitudes_deg_n=latitudes_deg_n, longitudes_deg_e=longitudes_deg_e,
        heights_m_asl=heights_m_asl
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_reflectivity_dir_name=getattr(
            INPUT_ARG_OBJECT, REFLECTIVITY_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
