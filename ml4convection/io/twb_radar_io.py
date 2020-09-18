"""IO methods for radar files from Taiwanese Weather Bureau (TWB)."""

import os
import gzip
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking

TIME_INTERVAL_SEC = 600
MIN_REFLECTIVITY_DBZ = -5.

VARIABLE_NAMES = [
    'yyyy', 'mm', 'dd', 'hh', 'mn', 'ss', 'nx', 'ny', 'nz', 'proj',
    'mapscale', 'projlat1', 'projlat2', 'projlon', 'alon', 'alat',
    'xy_scale', 'dx', 'dy', 'dxy_scale'
]
VARIABLE_TYPES = [numpy.int32] * len(VARIABLE_NAMES)
VARIABLE_TYPES[VARIABLE_NAMES.index('proj')] = numpy.uint32
VARIABLE_TYPE_DICT = numpy.dtype({
    'names': VARIABLE_NAMES,
    'formats': VARIABLE_TYPES
})

NUM_ROWS_KEY = 'ny'
NUM_COLUMNS_KEY = 'nx'
NUM_HEIGHTS_KEY = 'nz'
REFERENCE_LATITUDE_KEY = 'alat'
REFERENCE_LONGITUDE_KEY = 'alon'
LATLNG_SCALE_KEY = 'xy_scale'
LATITUDE_SPACING_KEY = 'dy'
LONGITUDE_SPACING_KEY = 'dx'
LATLNG_SPACING_SCALE_KEY = 'dxy_scale'

TIME_FORMAT_IN_MESSAGES = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_IN_DIR_NAMES = '%Y%m%d'
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d.%H%M'


def _read_latlng_from_file(data_object):
    """Reads lat-long coordinates from file.

    M = number of rows in grid
    N = number of columns in grid

    :param data_object: Object created by `numpy.frombuffer`.
    :return: latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :return: longitudes_deg_e: length-N numpy array of longitudes (deg E).
    """

    num_grid_rows = data_object[NUM_ROWS_KEY][0]
    num_grid_columns = data_object[NUM_COLUMNS_KEY][0]

    this_scale_factor = float(data_object[LATLNG_SCALE_KEY][0])
    reference_latitude_deg_n = (
        float(data_object[REFERENCE_LATITUDE_KEY][0]) / this_scale_factor
    )
    reference_longitude_deg_e = (
        float(data_object[REFERENCE_LONGITUDE_KEY][0]) / this_scale_factor
    )

    this_scale_factor = float(data_object[LATLNG_SPACING_SCALE_KEY][0])
    latitude_spacing_deg = (
        float(data_object[LATITUDE_SPACING_KEY][0]) / this_scale_factor
    )
    longitude_spacing_deg = (
        float(data_object[LONGITUDE_SPACING_KEY][0]) / this_scale_factor
    )

    row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )
    latitudes_deg_n = (
        reference_latitude_deg_n - row_indices * latitude_spacing_deg
    )
    latitudes_deg_n = latitudes_deg_n[::-1]

    column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )
    longitudes_deg_e = (
        reference_longitude_deg_e + column_indices * longitude_spacing_deg
    )

    return latitudes_deg_n, longitudes_deg_e


def find_file(
        top_directory_name, valid_time_unix_sec, with_3d=False,
        raise_error_if_missing=True):
    """Finds binary file with radar data.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_time_unix_sec: Valid time.
    :param with_3d: Boolean flag.  If True, will look for file with 3-D data.
        If False, will look for file with 2-D data (composite reflectivity).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: radar_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(with_3d)
    error_checking.assert_is_boolean(raise_error_if_missing)

    radar_file_name = '{0:s}/{1:s}{2:s}/{3:s}.{4:s}.gz'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_DIR_NAMES
        ),
        '' if with_3d else '/compref_mosaic',
        'MREF3D21L' if with_3d else 'COMPREF',
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
        )
    )

    if os.path.isfile(radar_file_name) or not raise_error_if_missing:
        return radar_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        radar_file_name
    )
    raise ValueError(error_string)


def file_name_to_time(radar_file_name):
    """Parses valid time from file name.

    :param radar_file_name: Path to radar file (see `find_file` for naming
        convention).
    :return: valid_time_unix_sec: Valid time.
    """

    error_checking.assert_is_string(radar_file_name)
    pathless_file_name = os.path.split(radar_file_name)[-1]
    extensionless_file_name = pathless_file_name[:-3]

    valid_time_string = '.'.join(extensionless_file_name.split('.')[-2:])

    return time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT_IN_FILE_NAMES
    )


def find_many_files(
        top_directory_name, first_time_unix_sec, last_time_unix_sec,
        with_3d=False, raise_error_if_all_missing=True,
        raise_error_if_any_missing=False, test_mode=False):
    """Finds many binary files with radar data.

    :param top_directory_name: See doc for `find_file`.
    :param first_time_unix_sec: First valid time.
    :param last_time_unix_sec: Last valid time.
    :param with_3d: See doc for `find_file`.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: radar_file_names: 1-D list of paths to radar files.  This list
        does *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    radar_file_names = []

    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = find_file(
            top_directory_name=top_directory_name,
            valid_time_unix_sec=this_time_unix_sec, with_3d=with_3d,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            radar_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(radar_file_names) == 0:
        first_time_string = time_conversion.unix_sec_to_string(
            first_time_unix_sec, TIME_FORMAT_IN_MESSAGES
        )
        last_time_string = time_conversion.unix_sec_to_string(
            last_time_unix_sec, TIME_FORMAT_IN_MESSAGES
        )

        error_string = (
            'Cannot find any file in directory "{0:s}" from times {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_time_string, last_time_string
        )
        raise ValueError(error_string)

    return radar_file_names


def read_file(gzip_file_name):
    """Reads radar data from binary file.

    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid

    :param gzip_file_name: Path to input file.
    :return: reflectivity_matrix_dbz: M-by-N-by-H numpy array of reflectivities.
    :return: latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :return: longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :return: heights_m_asl: length-H numpy array of heights (metres above sea
        level).  If the first output array contains composite reflectivity, this
        array will be [0].
    """

    gzip_file_handle = gzip.open(gzip_file_name, 'rb').read()
    data_object = numpy.frombuffer(
        gzip_file_handle, dtype=VARIABLE_TYPE_DICT, count=1
    )
    byte_offset = data_object.nbytes

    heights_m_asl = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int32,
        count=data_object[NUM_HEIGHTS_KEY][0], offset=byte_offset
    )
    byte_offset += heights_m_asl.nbytes

    z_scale = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int32, count=1, offset=byte_offset
    )
    byte_offset += z_scale.nbytes

    i_bb_mode = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int32, count=1, offset=byte_offset
    )
    byte_offset += i_bb_mode.nbytes

    unkn01 = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int32, count=9, offset=byte_offset
    )
    byte_offset += unkn01.nbytes

    field_name = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.uint8, count=20, offset=byte_offset
    )
    byte_offset += field_name.nbytes

    units = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.uint8, count=6, offset=byte_offset
    )
    byte_offset += units.nbytes

    scale_factor = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int32, count=1, offset=byte_offset
    )
    byte_offset += scale_factor.nbytes

    sentinel_value = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int32, count=1, offset=byte_offset
    )
    byte_offset += sentinel_value.nbytes

    nradar = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int32, count=1, offset=byte_offset
    )
    byte_offset += nradar.nbytes

    mosradar = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.uint8, count=4 * nradar[0],
        offset=byte_offset
    )
    byte_offset += mosradar.nbytes

    reflectivity_matrix_dbz = numpy.frombuffer(
        gzip_file_handle, dtype=numpy.int16, offset=byte_offset
    )
    reflectivity_matrix_dbz = (
        reflectivity_matrix_dbz.astype(numpy.float32) / float(scale_factor[0])
    )
    reflectivity_matrix_dbz[reflectivity_matrix_dbz < MIN_REFLECTIVITY_DBZ] = (
        numpy.nan
    )

    latitudes_deg_n, longitudes_deg_e = _read_latlng_from_file(data_object)

    dimensions = (
        len(heights_m_asl), len(latitudes_deg_n), len(longitudes_deg_e)
    )
    reflectivity_matrix_dbz = numpy.reshape(reflectivity_matrix_dbz, dimensions)
    reflectivity_matrix_dbz = numpy.swapaxes(reflectivity_matrix_dbz, 0, 2)
    reflectivity_matrix_dbz = numpy.swapaxes(reflectivity_matrix_dbz, 0, 1)
    # reflectivity_matrix_dbz = numpy.flip(reflectivity_matrix_dbz, axis=0)

    return (
        reflectivity_matrix_dbz,
        latitudes_deg_n, longitudes_deg_e, heights_m_asl
    )
