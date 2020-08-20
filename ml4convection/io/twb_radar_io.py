"""IO methods for radar files from Taiwanese Weather Bureau (TWB)."""

import os
import gzip
import copy
import shutil
import tempfile
import numpy
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_INTERVAL_SEC = 600
MIN_REFLECTIVITY_DBZ = -5.

ERROR_STRING = (
    '\nUnix command failed (log messages shown above should explain why).'
)

DEFAULT_GFORTRAN_COMPILER_NAME = 'gfortran'
FORTRAN_SCRIPT_NAME = '{0:s}/get_grid_info.f90'.format(os.getcwd())
FORTRAN_EXE_NAME = '{0:s}/get_grid_info.exe'.format(os.getcwd())

TIME_FORMAT_IN_MESSAGES = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_IN_DIR_NAMES = '%Y%m%d'
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d.%H%M'

LATITUDE_COLUMN_INDEX = 0
LONGITUDE_COLUMN_INDEX = 1
REFLECTIVITY_COLUMN_INDEX = 2
LATLNG_PRECISION_DEG = 1e-6


def find_file(
        top_directory_name, valid_time_unix_sec, with_3d=False,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_missing=True):
    """Finds binary file with radar data.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_time_unix_sec: Valid time.
    :param with_3d: Boolean flag.  If True, will look for file with 3-D data.
        If False, will look for file with 2-D data (composite reflectivity).
    :param prefer_zipped: Boolean flag.  If True, will look for zipped file
        first.  If False, will look for unzipped file first.
    :param allow_other_format: Boolean flag.  If True, will allow opposite of
        preferred file format (zipped or unzipped).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: radar_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(with_3d)
    error_checking.assert_is_boolean(prefer_zipped)
    error_checking.assert_is_boolean(allow_other_format)
    error_checking.assert_is_boolean(raise_error_if_missing)

    directory_name = '{0:s}/{1:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_DIR_NAMES
        )
    )

    if not with_3d:
        directory_name += '/compref_mosaic'

    radar_file_name = '{0:s}/{1:s}.{2:s}'.format(
        directory_name,
        'MREF3D21L' if with_3d else 'COMPREF',
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
        )
    )

    if prefer_zipped:
        radar_file_name += '.gz'

    if os.path.isfile(radar_file_name):
        return radar_file_name

    if allow_other_format:
        if prefer_zipped:
            radar_file_name = radar_file_name[:-3]
        else:
            radar_file_name += '.gz'

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

    extensionless_file_name = (
        pathless_file_name[:-3] if pathless_file_name.endswith('.gz')
        else pathless_file_name
    )

    valid_time_string = '.'.join(extensionless_file_name.split('.')[-2:])

    return time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT_IN_FILE_NAMES
    )


def find_many_files(
        top_directory_name, first_time_unix_sec, last_time_unix_sec,
        with_3d=False, prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many binary files with radar data.

    :param top_directory_name: See doc for `find_file`.
    :param first_time_unix_sec: First valid time.
    :param last_time_unix_sec: Last valid time.
    :param with_3d: See doc for `find_file`.
    :param prefer_zipped: Same.
    :param allow_other_format: Same.
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
            prefer_zipped=prefer_zipped, allow_other_format=allow_other_format,
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


def read_2d_file(
        binary_file_name, gfortran_compiler_name=DEFAULT_GFORTRAN_COMPILER_NAME,
        temporary_dir_name=None):
    """Reads 2-D radar data (composite reflectivity) from binary file.

    M = number of rows in grid
    N = number of columns in grid

    :param binary_file_name: Path to input file.
    :param gfortran_compiler_name: Path to gfortran compiler.
    :param temporary_dir_name: Name of temporary directory for text file, which
        will be deleted as soon as it is read.  If None, temporary directory
        will be set to default.
    :return: composite_refl_matrix_dbz: M-by-N numpy array of composite
        (column-maximum) reflectivities.
    :return: latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :return: longitudes_deg_e: length-N numpy array of longitudes (deg E).
    """

    error_checking.assert_file_exists(binary_file_name)
    # error_checking.assert_file_exists(gfortran_compiler_name)

    if not os.path.isfile(FORTRAN_EXE_NAME):
        command_string = '"{0:s}" "{1:s}" -o "{2:s}"'.format(
            gfortran_compiler_name, FORTRAN_SCRIPT_NAME, FORTRAN_EXE_NAME
        )

        exit_code = os.system(command_string)
        if exit_code != 0:
            raise ValueError(ERROR_STRING)

    if temporary_dir_name is not None:
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temporary_dir_name
        )

    is_file_zipped = binary_file_name.endswith('.gz')
    input_file_name = copy.deepcopy(binary_file_name)

    if is_file_zipped:
        gzip_file_object = gzip.open(binary_file_name, 'rb')
        binary_file_object = tempfile.NamedTemporaryFile(
            dir=temporary_dir_name, delete=False
        )
        binary_file_name = binary_file_object.name

        shutil.copyfileobj(gzip_file_object, binary_file_object)
        gzip_file_object.close()
        binary_file_object.close()

    temporary_text_file_name = tempfile.NamedTemporaryFile(
        dir=temporary_dir_name, delete=False
    ).name

    print('Reading data from binary file: "{0:s}"...'.format(input_file_name))
    fortran_exe_dir_name, fortran_exe_pathless_name = (
        os.path.split(FORTRAN_EXE_NAME)
    )
    command_string = 'cd "{0:s}"; ./{1:s} "{2:s}" > "{3:s}"'.format(
        fortran_exe_dir_name, fortran_exe_pathless_name, binary_file_name,
        temporary_text_file_name
    )

    exit_code = os.system(command_string)
    if is_file_zipped:
        os.remove(binary_file_name)
    if exit_code != 0:
        raise ValueError(ERROR_STRING)

    print('Reading data from temporary text file: "{0:s}"...'.format(
        temporary_text_file_name
    ))
    data_matrix = numpy.loadtxt(temporary_text_file_name)
    os.remove(temporary_text_file_name)

    all_latitudes_deg_n = number_rounding.round_to_nearest(
        data_matrix[:, LATITUDE_COLUMN_INDEX], LATLNG_PRECISION_DEG
    )
    all_longitudes_deg_e = number_rounding.round_to_nearest(
        data_matrix[:, LONGITUDE_COLUMN_INDEX], LATLNG_PRECISION_DEG
    )
    num_grid_rows = numpy.sum(all_longitudes_deg_e == all_longitudes_deg_e[0])
    num_grid_columns = numpy.sum(all_latitudes_deg_n == all_latitudes_deg_n[0])

    composite_refl_matrix_dbz = numpy.reshape(
        data_matrix[:, REFLECTIVITY_COLUMN_INDEX],
        (num_grid_rows, num_grid_columns)
    )
    composite_refl_matrix_dbz[
        composite_refl_matrix_dbz < MIN_REFLECTIVITY_DBZ
    ] = numpy.nan

    latitudes_deg_n = numpy.reshape(
        all_latitudes_deg_n, (num_grid_rows, num_grid_columns)
    )[:, 0]

    longitudes_deg_e = numpy.reshape(
        all_longitudes_deg_e, (num_grid_rows, num_grid_columns)
    )[0, :]

    return composite_refl_matrix_dbz, latitudes_deg_n, longitudes_deg_e
