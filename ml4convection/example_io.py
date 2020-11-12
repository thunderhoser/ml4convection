"""Input/output methods for learning examples."""

import os
import sys
import copy
import gzip
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import radar_io
import satellite_io
import twb_satellite_io
import normalization
import radar_utils
import standalone_utils

TOLERANCE = 1e-6

DATE_FORMAT = '%Y%m%d'
GZIP_FILE_EXTENSION = '.gz'

PREDICTOR_MATRIX_UNNORM_KEY = 'predictor_matrix_unnorm'
PREDICTOR_MATRIX_NORM_KEY = 'predictor_matrix_norm'
PREDICTOR_MATRIX_UNIF_NORM_KEY = 'predictor_matrix_unif_norm'
VALID_TIMES_KEY = 'valid_times_unix_sec'
BAND_NUMBERS_KEY = 'band_numbers'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
MASK_MATRIX_KEY = 'mask_matrix'
FULL_MASK_MATRIX_KEY = 'full_mask_matrix'
FULL_LATITUDES_KEY = 'full_latitudes_deg_n'
FULL_LONGITUDES_KEY = 'full_longitudes_deg_e'

ONE_PER_PREDICTOR_TIME_KEYS = [
    PREDICTOR_MATRIX_UNNORM_KEY, PREDICTOR_MATRIX_NORM_KEY,
    PREDICTOR_MATRIX_UNIF_NORM_KEY, VALID_TIMES_KEY
]
ONE_PER_PREDICTOR_PIXEL_KEYS = [
    PREDICTOR_MATRIX_UNNORM_KEY, PREDICTOR_MATRIX_NORM_KEY,
    PREDICTOR_MATRIX_UNIF_NORM_KEY
]
ONE_PER_BAND_NUMBER_KEYS = [
    PREDICTOR_MATRIX_UNNORM_KEY, PREDICTOR_MATRIX_NORM_KEY,
    PREDICTOR_MATRIX_UNIF_NORM_KEY, BAND_NUMBERS_KEY
]

TARGET_MATRIX_KEY = 'target_matrix'
ONE_PER_TARGET_TIME_KEYS = [TARGET_MATRIX_KEY, VALID_TIMES_KEY]

TIME_DIMENSION_KEY = 'time'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
FULL_ROW_DIMENSION_KEY = 'full_grid_row'
FULL_COLUMN_DIMENSION_KEY = 'full_grid_column'
BAND_DIMENSION_KEY = 'band'


def _create_predictors_one_day(
        input_file_name, spatial_downsampling_factor, normalization_dict,
        normalization_file_name):
    """Creates predictor values (from satellite data) for one day.

    E = number of examples per batch
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    :param input_file_name: Path to input file (will be read by
        `satellite_io.read_file`).
    :param spatial_downsampling_factor: See doc for `create_predictors`.
    :param normalization_dict: Dictionary returned by `normalization.read_file`.
        Will use this to normalize data.
    :param normalization_file_name: See doc for `create_predictors`.

    :return: predictor_dict: Dictionary with the following keys.
    predictor_dict['predictor_matrix_unnorm']: E-by-M-by-N-by-C numpy array of
        unnormalized predictor values.
    predictor_dict['predictor_matrix_norm']: E-by-M-by-N-by-C numpy array of
        normalized predictor values.
    predictor_dict['predictor_matrix_unif_norm']: E-by-M-by-N-by-C numpy array
        of uniformized, then normalized, predictor values.
    predictor_dict['valid_times_unix_sec']: length-E numpy array of valid times.
    predictor_dict['latitudes_deg_n']: length-M numpy array of latitudes
        (deg N).
    predictor_dict['longitudes_deg_e']: length-N numpy array of longitudes
        (deg E).
    predictor_dict['band_numbers']: length-C numpy array of spectral bands
        (integers).
    predictor_dict['normalization_file_name']: Same as input (metadata).
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    satellite_dict = satellite_io.read_file(
        netcdf_file_name=input_file_name, fill_nans=True
    )

    if spatial_downsampling_factor > 1:
        satellite_dict = downsample_data_in_space(
            satellite_dict=satellite_dict,
            downsampling_factor=spatial_downsampling_factor,
            change_coordinates=True
        )[0]

    predictor_matrix_unnorm = (
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] + 0.
    )

    satellite_dict = normalization.normalize_data(
        satellite_dict=satellite_dict, normalization_dict=normalization_dict,
        uniformize=False
    )
    predictor_matrix_norm = (
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] + 0.
    )
    satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] = (
        predictor_matrix_unnorm + 0.
    )

    satellite_dict = normalization.normalize_data(
        satellite_dict=satellite_dict, normalization_dict=normalization_dict,
        uniformize=True
    )
    predictor_matrix_unif_norm = (
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] + 0.
    )

    return {
        PREDICTOR_MATRIX_UNNORM_KEY: predictor_matrix_unnorm,
        PREDICTOR_MATRIX_NORM_KEY: predictor_matrix_norm,
        PREDICTOR_MATRIX_UNIF_NORM_KEY: predictor_matrix_unif_norm,
        VALID_TIMES_KEY: satellite_dict[satellite_io.VALID_TIMES_KEY],
        LATITUDES_KEY: satellite_dict[satellite_io.LATITUDES_KEY],
        LONGITUDES_KEY: satellite_dict[satellite_io.LONGITUDES_KEY],
        BAND_NUMBERS_KEY: satellite_dict[satellite_io.BAND_NUMBERS_KEY],
        NORMALIZATION_FILE_KEY: normalization_file_name
    }


def _create_predictors_one_day_partial_grids(
        input_file_name, normalization_dict, normalization_file_name,
        half_grid_size_px):
    """Creates predictor values for one day on partial, radar-centered grids.

    R = number of radar sites

    :param input_file_name: See doc for `_create_predictors_one_day`.
    :param normalization_dict: Same.
    :param normalization_file_name: Same.
    :param half_grid_size_px: Size of half-grid (pixels).  If this number is K,
        the grid will have 2 * K + 1 rows and 2 * K + 1 columns.
    :return: predictor_dicts: length-R list of dictionaries, each in format
        returned by `_create_predictors_one_day`.
    """

    predictor_dict_full_grid = _create_predictors_one_day(
        input_file_name=input_file_name, spatial_downsampling_factor=1,
        normalization_dict=normalization_dict,
        normalization_file_name=normalization_file_name
    )

    center_row_indices, center_column_indices = (
        radar_utils.radar_sites_to_grid_points(
            grid_latitudes_deg_n=predictor_dict_full_grid[LATITUDES_KEY],
            grid_longitudes_deg_e=predictor_dict_full_grid[LONGITUDES_KEY]
        )
    )

    num_radars = len(center_row_indices)
    predictor_dicts = [dict()] * num_radars

    for k in range(num_radars):
        predictor_dicts[k] = subset_grid(
            predictor_or_target_dict=copy.deepcopy(predictor_dict_full_grid),
            first_row=center_row_indices[k] - half_grid_size_px,
            last_row=center_row_indices[k] + half_grid_size_px,
            first_column=center_column_indices[k] - half_grid_size_px,
            last_column=center_column_indices[k] + half_grid_size_px,
        )

    return predictor_dicts


def _create_targets_one_day(
        echo_classifn_file_name, spatial_downsampling_factor, mask_dict):
    """Creates target values for one day.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param echo_classifn_file_name: Path to echo-classification file (will be
        read by `radar_io.read_echo_classifn_file`).
    :param spatial_downsampling_factor: See doc for `create_targets`.
    :param mask_dict: Dictionary in format returned by
        `radar_io.read_mask_file`, used to censor locations with bad radar
        coverage.  If you do not want a mask, leave this alone.

    :return: target_dict: Dictionary with the following keys.
    target_dict['target_matrix']: E-by-M-by-N numpy array of target values
        (0 or 1), indicating when and where convection occurs.
    target_dict['valid_times_unix_sec']: length-E numpy array of valid times.
    target_dict['latitudes_deg_n']: length-M numpy array of latitudes
        (deg N).
    target_dict['full_latitudes_deg_n']: Same.
    target_dict['longitudes_deg_e']: length-N numpy array of longitudes
        (deg E).
    target_dict['full_longitudes_deg_e']: Same.
    target_dict['mask_matrix']: M-by-N numpy array of Boolean flags.  False
        means that the grid cell is masked out.
    target_dict['full_mask_matrix']: Same.
    """

    print('Reading data from: "{0:s}"...'.format(echo_classifn_file_name))
    echo_classifn_dict = radar_io.read_echo_classifn_file(
        echo_classifn_file_name
    )
    echo_classifn_dict = radar_io.expand_to_satellite_grid(
        any_radar_dict=echo_classifn_dict
    )

    if spatial_downsampling_factor > 1:
        echo_classifn_dict = downsample_data_in_space(
            echo_classifn_dict=echo_classifn_dict,
            downsampling_factor=spatial_downsampling_factor,
            change_coordinates=True
        )[1]

    target_matrix = echo_classifn_dict[radar_io.CONVECTIVE_FLAGS_KEY]

    assert numpy.allclose(
        echo_classifn_dict[radar_io.LATITUDES_KEY],
        mask_dict[radar_io.LATITUDES_KEY],
        atol=TOLERANCE
    )

    assert numpy.allclose(
        echo_classifn_dict[radar_io.LONGITUDES_KEY],
        mask_dict[radar_io.LONGITUDES_KEY],
        atol=TOLERANCE
    )

    num_times = len(echo_classifn_dict[radar_io.VALID_TIMES_KEY])
    mask_matrix = mask_dict[radar_io.MASK_MATRIX_KEY].astype(bool)

    for i in range(num_times):
        target_matrix[i, ...] = numpy.logical_and(
            target_matrix[i, ...], mask_matrix
        )

    target_matrix = target_matrix.astype(int)

    print((
        'Number of target values = {0:d} ... event frequency = {1:.2g}'
    ).format(
        target_matrix.size, numpy.mean(target_matrix)
    ))

    return {
        TARGET_MATRIX_KEY: target_matrix,
        VALID_TIMES_KEY: echo_classifn_dict[radar_io.VALID_TIMES_KEY],
        LATITUDES_KEY: echo_classifn_dict[radar_io.LATITUDES_KEY],
        FULL_LATITUDES_KEY: echo_classifn_dict[radar_io.LATITUDES_KEY],
        LONGITUDES_KEY: echo_classifn_dict[radar_io.LONGITUDES_KEY],
        FULL_LONGITUDES_KEY: echo_classifn_dict[radar_io.LONGITUDES_KEY],
        MASK_MATRIX_KEY: mask_matrix,
        FULL_MASK_MATRIX_KEY: mask_matrix
    }


def _create_targets_one_day_partial_grids(
        echo_classifn_file_name, mask_dict, half_grid_size_px):
    """Creates predictor values for one day on partial, radar-centered grids.

    R = number of radar sites
    E = number of examples
    M = number of rows in full grid
    N = number of columns in full grid
    m = number of rows in each partial grid
    n = number of columns in each partial grid

    :param echo_classifn_file_name: See doc for `_create_targets_one_day`.
    :param mask_dict: Same.
    :param half_grid_size_px: Size of half-grid (pixels).  If this number is K,
        the grid will have 2 * K + 1 rows and 2 * K + 1 columns.
    :return: target_dicts: length-R list of dictionaries, each with the
        following keys.

    'target_matrix': E-by-m-by-n numpy array of target values (0 or 1),
        indicating when and where convection occurs.
    'valid_times_unix_sec': length-E numpy array of valid times.
    'latitudes_deg_n': length-m numpy array of latitudes (deg N).
    'full_latitudes_deg_n': length-M numpy array of latitudes (deg N).
    'longitudes_deg_e': length-n numpy array of longitudes (deg E).
    'full_longitudes_deg_e': length-N numpy array of longitudes (deg E).
    'mask_matrix': m-by-n numpy array of Boolean flags.  False means that the
        grid cell is masked out.
    'full_mask_matrix': Same but with dimensions of M x N.
    """

    target_dict_full_grid = _create_targets_one_day(
        echo_classifn_file_name=echo_classifn_file_name,
        spatial_downsampling_factor=1, mask_dict=mask_dict
    )

    center_row_indices, center_column_indices = (
        radar_utils.radar_sites_to_grid_points(
            grid_latitudes_deg_n=target_dict_full_grid[LATITUDES_KEY],
            grid_longitudes_deg_e=target_dict_full_grid[LONGITUDES_KEY]
        )
    )

    num_radars = len(center_row_indices)
    target_dicts = [dict()] * num_radars

    for k in range(num_radars):
        target_dicts[k] = subset_grid(
            predictor_or_target_dict=copy.deepcopy(target_dict_full_grid),
            first_row=center_row_indices[k] - half_grid_size_px,
            last_row=center_row_indices[k] + half_grid_size_px,
            first_column=center_column_indices[k] - half_grid_size_px,
            last_column=center_column_indices[k] + half_grid_size_px,
        )

    return target_dicts


def _write_predictor_file(predictor_dict, netcdf_file_name):
    """Writes predictors to NetCDF file.

    :param predictor_dict: Dictionary created by `_create_predictors_one_day`.
    :param netcdf_file_name: Path to output file.
    :raises: ValueError: if output file is a gzip file.
    """

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError('Output file must not be gzip file.')

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    predictor_matrix_unnorm = predictor_dict[PREDICTOR_MATRIX_UNNORM_KEY]
    num_times = predictor_matrix_unnorm.shape[0]
    num_grid_rows = predictor_matrix_unnorm.shape[1]
    num_grid_columns = predictor_matrix_unnorm.shape[2]
    num_channels = predictor_matrix_unnorm.shape[3]

    dataset_object.setncattr(
        NORMALIZATION_FILE_KEY, predictor_dict[NORMALIZATION_FILE_KEY]
    )

    dataset_object.createDimension(TIME_DIMENSION_KEY, num_times)
    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)
    dataset_object.createDimension(BAND_DIMENSION_KEY, num_channels)

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=TIME_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = (
        predictor_dict[VALID_TIMES_KEY]
    )

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = predictor_dict[LATITUDES_KEY]

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32, dimensions=COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = predictor_dict[LONGITUDES_KEY]

    dataset_object.createVariable(
        BAND_NUMBERS_KEY, datatype=numpy.int32, dimensions=BAND_DIMENSION_KEY
    )
    dataset_object.variables[BAND_NUMBERS_KEY][:] = (
        predictor_dict[BAND_NUMBERS_KEY]
    )

    these_dim = (
        TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
        BAND_DIMENSION_KEY
    )
    dataset_object.createVariable(
        PREDICTOR_MATRIX_UNNORM_KEY, datatype=numpy.float32,
        dimensions=these_dim
    )
    dataset_object.variables[PREDICTOR_MATRIX_UNNORM_KEY][:] = (
        predictor_dict[PREDICTOR_MATRIX_UNNORM_KEY]
    )

    dataset_object.createVariable(
        PREDICTOR_MATRIX_NORM_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[PREDICTOR_MATRIX_NORM_KEY][:] = (
        predictor_dict[PREDICTOR_MATRIX_NORM_KEY]
    )

    dataset_object.createVariable(
        PREDICTOR_MATRIX_UNIF_NORM_KEY, datatype=numpy.float32,
        dimensions=these_dim
    )
    dataset_object.variables[PREDICTOR_MATRIX_UNIF_NORM_KEY][:] = (
        predictor_dict[PREDICTOR_MATRIX_UNIF_NORM_KEY]
    )

    dataset_object.close()


def _read_predictors(dataset_object, read_unnormalized, read_normalized,
                     read_unif_normalized):
    """Reads predictors from NetCDF file.

    This method should be called only from `read_predictor_file`.

    :param dataset_object: Instance of `netCDF4.Dataset`.
    :param read_unnormalized: See doc for `read_predictor_file`.
    :param read_normalized: Same.
    :param read_unif_normalized: Same.
    :return: predictor_dict: Same.
    """

    predictor_dict = {
        PREDICTOR_MATRIX_UNNORM_KEY: None,
        PREDICTOR_MATRIX_NORM_KEY: None,
        PREDICTOR_MATRIX_UNIF_NORM_KEY: None,
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
        BAND_NUMBERS_KEY: dataset_object.variables[BAND_NUMBERS_KEY][:],
        NORMALIZATION_FILE_KEY:
            str(getattr(dataset_object, NORMALIZATION_FILE_KEY))
    }

    if read_unnormalized:
        predictor_dict[PREDICTOR_MATRIX_UNNORM_KEY] = (
            dataset_object.variables[PREDICTOR_MATRIX_UNNORM_KEY][:]
        )

    if read_normalized:
        predictor_dict[PREDICTOR_MATRIX_NORM_KEY] = (
            dataset_object.variables[PREDICTOR_MATRIX_NORM_KEY][:]
        )

    if read_unif_normalized:
        predictor_dict[PREDICTOR_MATRIX_UNIF_NORM_KEY] = (
            dataset_object.variables[PREDICTOR_MATRIX_UNIF_NORM_KEY][:]
        )

    return predictor_dict


def _write_target_file(target_dict, netcdf_file_name):
    """Writes targets to NetCDF file.

    :param target_dict: Dictionary created by `_create_targets_one_day`.
    :param netcdf_file_name: Path to output file.
    :raises: ValueError: if output file is a gzip file.
    """

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError('Output file must not be gzip file.')

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    target_matrix = target_dict[TARGET_MATRIX_KEY]
    num_times = target_matrix.shape[0]
    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]

    full_mask_matrix = target_dict[FULL_MASK_MATRIX_KEY]
    num_full_grid_rows = full_mask_matrix.shape[0]
    num_full_grid_columns = full_mask_matrix.shape[1]

    dataset_object.createDimension(TIME_DIMENSION_KEY, num_times)
    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)
    dataset_object.createDimension(FULL_ROW_DIMENSION_KEY, num_full_grid_rows)
    dataset_object.createDimension(
        FULL_COLUMN_DIMENSION_KEY, num_full_grid_columns
    )

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=TIME_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = target_dict[VALID_TIMES_KEY]

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = target_dict[LATITUDES_KEY]

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32, dimensions=COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = target_dict[LONGITUDES_KEY]

    dataset_object.createVariable(
        MASK_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[MASK_MATRIX_KEY][:] = (
        target_dict[MASK_MATRIX_KEY].astype(int)
    )

    dataset_object.createVariable(
        FULL_LATITUDES_KEY, datatype=numpy.float32,
        dimensions=FULL_ROW_DIMENSION_KEY
    )
    dataset_object.variables[FULL_LATITUDES_KEY][:] = (
        target_dict[FULL_LATITUDES_KEY]
    )

    dataset_object.createVariable(
        FULL_LONGITUDES_KEY, datatype=numpy.float32,
        dimensions=FULL_COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[FULL_LONGITUDES_KEY][:] = (
        target_dict[FULL_LONGITUDES_KEY]
    )

    dataset_object.createVariable(
        FULL_MASK_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(FULL_ROW_DIMENSION_KEY, FULL_COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[FULL_MASK_MATRIX_KEY][:] = (
        target_dict[FULL_MASK_MATRIX_KEY].astype(int)
    )

    these_dim = (TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    dataset_object.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[TARGET_MATRIX_KEY][:] = (
        target_dict[TARGET_MATRIX_KEY]
    )

    dataset_object.close()


def _read_targets(dataset_object):
    """Reads targets from NetCDF file.

    This method should be called only from `read_target_file`.

    :param dataset_object: Instance of `netCDF4.Dataset`.
    :return: target_dict: See doc for `read_target_file`.
    """

    target_dict = {
        TARGET_MATRIX_KEY: dataset_object.variables[TARGET_MATRIX_KEY][:],
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:]
    }

    if MASK_MATRIX_KEY in dataset_object.variables:
        target_dict[MASK_MATRIX_KEY] = (
            dataset_object.variables[MASK_MATRIX_KEY][:].astype(bool)
        )
    else:
        mask_file_name = str(getattr(dataset_object, 'mask_file_name'))
        mask_dict = radar_io.read_mask_file(mask_file_name)
        mask_dict = radar_io.expand_to_satellite_grid(any_radar_dict=mask_dict)

        num_target_latitudes = len(target_dict[LATITUDES_KEY])
        num_full_latitudes = len(twb_satellite_io.GRID_LATITUDES_DEG_N)
        downsampling_factor = int(numpy.floor(
            float(num_full_latitudes) / num_target_latitudes
        ))

        if downsampling_factor > 1:
            mask_dict = radar_io.downsample_in_space(
                any_radar_dict=mask_dict, downsampling_factor=4
            )

        target_dict[MASK_MATRIX_KEY] = (
            mask_dict[radar_io.MASK_MATRIX_KEY].astype(bool)
        )

    if FULL_MASK_MATRIX_KEY in dataset_object.variables:
        target_dict[FULL_MASK_MATRIX_KEY] = (
            dataset_object.variables[FULL_MASK_MATRIX_KEY][:].astype(bool)
        )
        target_dict[FULL_LATITUDES_KEY] = (
            dataset_object.variables[FULL_LATITUDES_KEY][:]
        )
        target_dict[FULL_LONGITUDES_KEY] = (
            dataset_object.variables[FULL_LONGITUDES_KEY][:]
        )
    else:
        target_dict[FULL_MASK_MATRIX_KEY] = copy.deepcopy(
            target_dict[MASK_MATRIX_KEY]
        )
        target_dict[FULL_LATITUDES_KEY] = target_dict[LATITUDES_KEY] + 0.
        target_dict[FULL_LONGITUDES_KEY] = target_dict[LONGITUDES_KEY] + 0.

    if numpy.any(numpy.diff(target_dict[LATITUDES_KEY]) < 0):
        target_dict[LATITUDES_KEY] = target_dict[LATITUDES_KEY][::-1]
        target_dict[TARGET_MATRIX_KEY] = numpy.flip(
            target_dict[TARGET_MATRIX_KEY], axis=1
        )
        target_dict[MASK_MATRIX_KEY] = numpy.flip(
            target_dict[MASK_MATRIX_KEY], axis=0
        )

    if numpy.any(numpy.diff(target_dict[FULL_LATITUDES_KEY]) < 0):
        target_dict[FULL_LATITUDES_KEY] = target_dict[FULL_LATITUDES_KEY][::-1]
        target_dict[FULL_MASK_MATRIX_KEY] = numpy.flip(
            target_dict[FULL_MASK_MATRIX_KEY], axis=0
        )

    return target_dict


def downsample_data_in_space(downsampling_factor, change_coordinates=False,
                             satellite_dict=None, echo_classifn_dict=None):
    """Downsamples satellite and/or radar data in space.

    At least one of `satellite_dict` and `echo_classifn_dict` must be specified
    (not None).

    :param downsampling_factor: Downsampling factor (integer).
    :param change_coordinates: Boolean flag.  If True (False), will (not) change
        coordinates in dictionaries to reflect downsampling.
    :param satellite_dict: Dictionary in format returned by
        `satellite_io.read_file`.
    :param echo_classifn_dict: Dictionary in format returned by
        `radar_io.read_echo_classifn_file`.
    :return: satellite_dict: Same as input but maybe with coarser spatial
        resolution.
    :return: echo_classifn_dict: Same as input but maybe with coarser spatial
        resolution.
    :raises: ValueError: if
        `satellite_dict is None and echo_classifn_dict is None`.
    """

    error_checking.assert_is_integer(downsampling_factor)
    error_checking.assert_is_greater(downsampling_factor, 1)
    error_checking.assert_is_boolean(change_coordinates)

    if satellite_dict is None and echo_classifn_dict is None:
        raise ValueError(
            'satellite_dict and echo_classifn_dict cannot both be None.'
        )

    if satellite_dict is not None:
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] = (
            standalone_utils.do_2d_pooling(
                feature_matrix=satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY],
                window_size_px=downsampling_factor, do_max_pooling=False
            )
        )

    if echo_classifn_dict is not None:
        convective_flag_matrix = numpy.expand_dims(
            echo_classifn_dict[radar_io.CONVECTIVE_FLAGS_KEY].astype(float),
            axis=-1
        )
        convective_flag_matrix = standalone_utils.do_2d_pooling(
            feature_matrix=convective_flag_matrix,
            window_size_px=downsampling_factor, do_max_pooling=True
        )
        echo_classifn_dict[radar_io.CONVECTIVE_FLAGS_KEY] = (
            convective_flag_matrix[..., 0] >= 0.99
        )

    if not change_coordinates:
        return satellite_dict, echo_classifn_dict

    if satellite_dict is None:
        latitude_matrix_deg_n = numpy.expand_dims(
            echo_classifn_dict[radar_io.LATITUDES_KEY], axis=0
        )
    else:
        latitude_matrix_deg_n = numpy.expand_dims(
            satellite_dict[satellite_io.LATITUDES_KEY], axis=0
        )

    latitude_matrix_deg_n = numpy.expand_dims(latitude_matrix_deg_n, axis=-1)
    latitude_matrix_deg_n = standalone_utils.do_1d_pooling(
        feature_matrix=latitude_matrix_deg_n,
        window_size_px=downsampling_factor, do_max_pooling=False
    )
    latitudes_deg_n = latitude_matrix_deg_n[0, :, 0]

    if satellite_dict is not None:
        satellite_dict[satellite_io.LATITUDES_KEY] = latitudes_deg_n + 0.
    if echo_classifn_dict is not None:
        echo_classifn_dict[radar_io.LATITUDES_KEY] = latitudes_deg_n + 0.

    if satellite_dict is None:
        longitude_matrix_deg_e = numpy.expand_dims(
            echo_classifn_dict[radar_io.LONGITUDES_KEY], axis=0
        )
    else:
        longitude_matrix_deg_e = numpy.expand_dims(
            satellite_dict[satellite_io.LONGITUDES_KEY], axis=0
        )

    # TODO(thunderhoser): Careful: this will not work with wrap-around at the
    # date line.
    longitude_matrix_deg_e = numpy.expand_dims(longitude_matrix_deg_e, axis=-1)
    longitude_matrix_deg_e = standalone_utils.do_1d_pooling(
        feature_matrix=longitude_matrix_deg_e,
        window_size_px=downsampling_factor, do_max_pooling=False
    )
    longitudes_deg_e = longitude_matrix_deg_e[0, :, 0]

    if satellite_dict is not None:
        satellite_dict[satellite_io.LONGITUDES_KEY] = longitudes_deg_e + 0.
    if echo_classifn_dict is not None:
        echo_classifn_dict[radar_io.LONGITUDES_KEY] = longitudes_deg_e + 0.

    return satellite_dict, echo_classifn_dict


def create_predictors(
        top_input_dir_name, spatial_downsampling_factor,
        first_date_string, last_date_string, normalization_file_name,
        top_output_dir_name, raise_error_if_all_missing=True):
    """Creates predictor values from satellite data.

    :param top_input_dir_name: Name of top-level directory with satellite
        data.  Files therein will be found by `satellite_io.find_file` and read
        by `satellite_io.read_file`.
    :param spatial_downsampling_factor: Downsampling factor (integer), used to
        coarsen spatial resolution.  If you do not want to coarsen spatial
        resolution, make this 1.
    :param first_date_string: First day (format "yyyymmdd").  Will create
        predictors for all days in `first_date_string`...`last_date_string`.
    :param last_date_string: See above.
    :param normalization_file_name: Path to file with normalization parameters
        (will be read by `normalization.read_file`).
    :param top_output_dir_name: Name of top-level output directory.  Files will
        be written here by `_write_predictor_file`, to exact locations
        determined by `find_predictor_file`.
    :param raise_error_if_all_missing: Boolean flag.  If all input files are
        missing and `raise_error_if_all_missing == True`, will throw error.
    """

    error_checking.assert_is_integer(spatial_downsampling_factor)
    error_checking.assert_is_geq(spatial_downsampling_factor, 1)
    error_checking.assert_is_string(normalization_file_name)

    input_file_names = satellite_io.find_many_files(
        top_directory_name=top_input_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        raise_error_if_all_missing=raise_error_if_all_missing,
        raise_error_if_any_missing=False
    )

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    normalization_dict = normalization.read_file(normalization_file_name)

    for this_input_file_name in input_file_names:
        print('\n')

        this_predictor_dict = _create_predictors_one_day(
            input_file_name=this_input_file_name,
            spatial_downsampling_factor=spatial_downsampling_factor,
            normalization_dict=normalization_dict,
            normalization_file_name=normalization_file_name
        )

        this_output_file_name = find_predictor_file(
            top_directory_name=top_output_dir_name,
            date_string=satellite_io.file_name_to_date(this_input_file_name),
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Writing predictors to: "{0:s}"...'.format(this_output_file_name))
        _write_predictor_file(
            predictor_dict=this_predictor_dict,
            netcdf_file_name=this_output_file_name
        )

        compress_file(this_output_file_name)
        os.remove(this_output_file_name)


def create_predictors_partial_grids(
        top_input_dir_name, first_date_string, last_date_string,
        normalization_file_name, half_grid_size_px, top_output_dir_name,
        raise_error_if_all_missing=True):
    """Creates predictor values on partial, radar-centered grids.

    :param top_input_dir_name: See doc for `create_predictors`.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param normalization_file_name: Same.
    :param half_grid_size_px: Size of half-grid (pixels).  If this number is K,
        the grid will have 2 * K + 1 rows and 2 * K + 1 columns.
    :param top_output_dir_name: See doc for `create_predictors`.
    :param raise_error_if_all_missing: Same.
    """

    error_checking.assert_is_string(normalization_file_name)
    error_checking.assert_is_integer(half_grid_size_px)
    error_checking.assert_is_greater(half_grid_size_px, 0)

    input_file_names = satellite_io.find_many_files(
        top_directory_name=top_input_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        raise_error_if_all_missing=raise_error_if_all_missing,
        raise_error_if_any_missing=False
    )

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    normalization_dict = normalization.read_file(normalization_file_name)

    for this_input_file_name in input_file_names:
        print('\n')

        these_predictor_dicts = _create_predictors_one_day_partial_grids(
            input_file_name=this_input_file_name,
            half_grid_size_px=half_grid_size_px,
            normalization_dict=normalization_dict,
            normalization_file_name=normalization_file_name
        )

        for k in range(len(these_predictor_dicts)):
            this_output_file_name = find_predictor_file(
                top_directory_name=top_output_dir_name,
                date_string=
                satellite_io.file_name_to_date(this_input_file_name),
                radar_number=k, prefer_zipped=False, allow_other_format=False,
                raise_error_if_missing=False
            )

            print('Writing predictors to: "{0:s}"...'.format(
                this_output_file_name
            ))

            _write_predictor_file(
                predictor_dict=these_predictor_dicts[k],
                netcdf_file_name=this_output_file_name
            )

            compress_file(this_output_file_name)
            os.remove(this_output_file_name)


def create_targets(
        top_echo_classifn_dir_name, spatial_downsampling_factor,
        first_date_string, last_date_string, top_output_dir_name,
        mask_file_name, raise_error_if_all_missing=True):
    """Creates target values.

    :param top_echo_classifn_dir_name: Name of top-level directory with
        echo-classification data.  Files therein will be found by
        `radar_io.find_file` and read by `radar_io.read_echo_classifn_file`.
    :param spatial_downsampling_factor: See doc for `create_predictors`.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param top_output_dir_name: Name of top-level output directory.  Files will
        be written by `_write_target_file`, to locations therein determined by
        `find_target_file`.
    :param mask_file_name: Path to file with mask (used to censor locations with
        bad radar coverage).  Will be read by `radar_io.read_mask_file`.
    :param raise_error_if_all_missing: Boolean flag.  If all input files are
        missing and `raise_error_if_all_missing == True`, will throw error.
    """

    error_checking.assert_is_integer(spatial_downsampling_factor)
    error_checking.assert_is_geq(spatial_downsampling_factor, 1)

    echo_classifn_file_names = radar_io.find_many_files(
        top_directory_name=top_echo_classifn_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
        raise_error_if_all_missing=raise_error_if_all_missing,
        raise_error_if_any_missing=False
    )

    print('Reading mask from: "{0:s}"...'.format(mask_file_name))
    mask_dict = radar_io.read_mask_file(mask_file_name)
    mask_dict = radar_io.expand_to_satellite_grid(any_radar_dict=mask_dict)

    if spatial_downsampling_factor > 1:
        mask_dict = radar_io.downsample_in_space(
            any_radar_dict=mask_dict,
            downsampling_factor=spatial_downsampling_factor
        )

    for i in range(len(echo_classifn_file_names)):
        this_target_dict = _create_targets_one_day(
            echo_classifn_file_name=echo_classifn_file_names[i],
            spatial_downsampling_factor=spatial_downsampling_factor,
            mask_dict=mask_dict
        )

        this_output_file_name = find_target_file(
            top_directory_name=top_output_dir_name,
            date_string=radar_io.file_name_to_date(echo_classifn_file_names[i]),
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Writing targets to: "{0:s}"...'.format(this_output_file_name))
        _write_target_file(
            target_dict=this_target_dict, netcdf_file_name=this_output_file_name
        )

        compress_file(this_output_file_name)
        os.remove(this_output_file_name)

        if i != len(echo_classifn_file_names) - 1:
            print('\n')


def create_targets_partial_grids(
        top_echo_classifn_dir_name, half_grid_size_px,
        first_date_string, last_date_string, top_output_dir_name,
        mask_file_name, raise_error_if_all_missing=True):
    """Creates target values on partial, radar-centered grids.

    :param top_echo_classifn_dir_name: See doc for `create_targets`.
    :param half_grid_size_px: Size of half-grid (pixels).  If this number is K,
        the grid will have 2 * K + 1 rows and 2 * K + 1 columns.
    :param first_date_string: See doc for `create_targets`.
    :param last_date_string: Same.
    :param top_output_dir_name: Same.
    :param mask_file_name: Same.
    :param raise_error_if_all_missing: Same.
    """

    error_checking.assert_is_integer(half_grid_size_px)
    error_checking.assert_is_greater(half_grid_size_px, 0)

    echo_classifn_file_names = radar_io.find_many_files(
        top_directory_name=top_echo_classifn_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
        raise_error_if_all_missing=raise_error_if_all_missing,
        raise_error_if_any_missing=False
    )

    print('Reading mask from: "{0:s}"...'.format(mask_file_name))
    mask_dict = radar_io.read_mask_file(mask_file_name)
    mask_dict = radar_io.expand_to_satellite_grid(any_radar_dict=mask_dict)

    for i in range(len(echo_classifn_file_names)):
        print('\n')

        these_target_dicts = _create_targets_one_day_partial_grids(
            echo_classifn_file_name=echo_classifn_file_names[i],
            half_grid_size_px=half_grid_size_px,
            mask_dict=mask_dict
        )

        for k in range(len(these_target_dicts)):
            this_output_file_name = find_target_file(
                top_directory_name=top_output_dir_name,
                date_string=
                radar_io.file_name_to_date(echo_classifn_file_names[i]),
                radar_number=k, prefer_zipped=False, allow_other_format=False,
                raise_error_if_missing=False
            )

            print('Writing targets to: "{0:s}"...'.format(
                this_output_file_name
            ))

            _write_target_file(
                target_dict=these_target_dicts[k],
                netcdf_file_name=this_output_file_name
            )

            compress_file(this_output_file_name)
            os.remove(this_output_file_name)


def find_predictor_file(
        top_directory_name, date_string, radar_number=None, prefer_zipped=True,
        allow_other_format=True, raise_error_if_missing=True):
    """Finds NetCDF file with predictors.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param date_string: Date (format "yyyymmdd").
    :param radar_number: Radar number (non-negative integer).  If you are
        looking for data on the full grid, leave this alone.
    :param prefer_zipped: Boolean flag.  If True, will look for zipped file
        first.  If False, will look for unzipped file first.
    :param allow_other_format: Boolean flag.  If True, will allow opposite of
        preferred file format (zipped or unzipped).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: predictor_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    _ = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(prefer_zipped)
    error_checking.assert_is_boolean(allow_other_format)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if radar_number is not None:
        error_checking.assert_is_integer(radar_number)
        error_checking.assert_is_geq(radar_number, 0)

    predictor_file_name = '{0:s}/{1:s}/predictors_{2:s}{3:s}.nc{4:s}'.format(
        top_directory_name, date_string[:4], date_string,
        '' if radar_number is None else '_radar{0:d}'.format(radar_number),
        GZIP_FILE_EXTENSION if prefer_zipped else ''
    )

    if os.path.isfile(predictor_file_name):
        return predictor_file_name

    if allow_other_format:
        if prefer_zipped:
            predictor_file_name = (
                predictor_file_name[:-len(GZIP_FILE_EXTENSION)]
            )
        else:
            predictor_file_name += GZIP_FILE_EXTENSION

    if os.path.isfile(predictor_file_name) or not raise_error_if_missing:
        return predictor_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        predictor_file_name
    )
    raise ValueError(error_string)


def file_name_to_date(predictor_file_name):
    """Parses date from name of predictor or target file.

    :param predictor_file_name: Path to predictor or target file (see
        `find_predictor_file` or `find_target_file` for naming convention).
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    error_checking.assert_is_string(predictor_file_name)
    pathless_file_name = os.path.split(predictor_file_name)[-1]

    valid_date_string = pathless_file_name.split('.')[0].split('_')[-1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_many_predictor_files(
        top_directory_name, first_date_string, last_date_string,
        radar_number=None, prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with predictors.

    :param top_directory_name: See doc for `find_predictor_file`.
    :param first_date_string: First date (format "yyyymmdd").
    :param last_date_string: Last date (format "yyyymmdd").
    :param radar_number: See doc for `find_predictor_file`.
    :param prefer_zipped: Same.
    :param allow_other_format: Same.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: predictor_file_names: 1-D list of paths to target files.  This list
        does *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    predictor_file_names = []

    for this_date_string in date_strings:
        this_file_name = find_predictor_file(
            top_directory_name=top_directory_name, date_string=this_date_string,
            radar_number=radar_number,
            prefer_zipped=prefer_zipped, allow_other_format=allow_other_format,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            predictor_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(predictor_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_date_string, last_date_string
        )
        raise ValueError(error_string)

    return predictor_file_names


def read_predictor_file(netcdf_file_name, read_unnormalized, read_normalized,
                        read_unif_normalized):
    """Reads predictors from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param read_unnormalized: Boolean flag.  If True, will read unnormalized
        predictors.  If False, key `predictor_matrix_unnorm` in the output
        dictionary will be None.
    :param read_normalized: Boolean flag.  If True, will read normalized
        predictors.  If False, key `predictor_matrix_norm` in the output
        dictionary will be None.
    :param read_unif_normalized: Boolean flag.  If True, will read
        uniformized/normalized predictors.  If False, key
        `predictor_matrix_unif_norm` in the output dictionary will be None.
    :return: predictor_dict: Dictionary in format returned by
        `_create_predictors_one_day`.
    """

    error_checking.assert_is_boolean(read_unnormalized)
    error_checking.assert_is_boolean(read_normalized)
    error_checking.assert_is_boolean(read_unif_normalized)

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        with gzip.open(netcdf_file_name) as gzip_handle:
            with netCDF4.Dataset(
                    'dummy', mode='r', memory=gzip_handle.read()
            ) as dataset_object:
                return _read_predictors(
                    dataset_object=dataset_object,
                    read_unnormalized=read_unnormalized,
                    read_normalized=read_normalized,
                    read_unif_normalized=read_unif_normalized
                )

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    predictor_dict = _read_predictors(
        dataset_object=dataset_object,
        read_unnormalized=read_unnormalized,
        read_normalized=read_normalized,
        read_unif_normalized=read_unif_normalized
    )

    dataset_object.close()
    return predictor_dict


def find_target_file(
        top_directory_name, date_string, radar_number=None, prefer_zipped=True,
        allow_other_format=True, raise_error_if_missing=True):
    """Finds NetCDF file with targets.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param date_string: Date (format "yyyymmdd").
    :param radar_number: Radar number (non-negative integer).  If you are
        looking for data on the full grid, leave this alone.
    :param prefer_zipped: Boolean flag.  If True, will look for zipped file
        first.  If False, will look for unzipped file first.
    :param allow_other_format: Boolean flag.  If True, will allow opposite of
        preferred file format (zipped or unzipped).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: target_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    _ = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(prefer_zipped)
    error_checking.assert_is_boolean(allow_other_format)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if radar_number is not None:
        error_checking.assert_is_integer(radar_number)
        error_checking.assert_is_geq(radar_number, 0)

    target_file_name = '{0:s}/{1:s}/targets_{2:s}{3:s}.nc{4:s}'.format(
        top_directory_name, date_string[:4], date_string,
        '' if radar_number is None else '_radar{0:d}'.format(radar_number),
        GZIP_FILE_EXTENSION if prefer_zipped else ''
    )

    if os.path.isfile(target_file_name):
        return target_file_name

    if allow_other_format:
        if prefer_zipped:
            target_file_name = target_file_name[:-len(GZIP_FILE_EXTENSION)]
        else:
            target_file_name += GZIP_FILE_EXTENSION

    if os.path.isfile(target_file_name) or not raise_error_if_missing:
        return target_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        target_file_name
    )
    raise ValueError(error_string)


def find_many_target_files(
        top_directory_name, first_date_string, last_date_string,
        radar_number=None, prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with targets.

    :param top_directory_name: See doc for `find_target_file`.
    :param first_date_string: First date (format "yyyymmdd").
    :param last_date_string: Last date (format "yyyymmdd").
    :param radar_number: See doc for `find_target_file`.
    :param prefer_zipped: Same.
    :param allow_other_format: Same.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: target_file_names: 1-D list of paths to target files.  This list
        does *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    target_file_names = []

    for this_date_string in date_strings:
        this_file_name = find_target_file(
            top_directory_name=top_directory_name, date_string=this_date_string,
            radar_number=radar_number,
            prefer_zipped=prefer_zipped, allow_other_format=allow_other_format,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            target_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(target_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_date_string, last_date_string
        )
        raise ValueError(error_string)

    return target_file_names


def read_target_file(netcdf_file_name):
    """Reads targets from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: target_dict: Dictionary in format returned by
        `_create_targets_one_day`.
    """

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        with gzip.open(netcdf_file_name) as gzip_handle:
            with netCDF4.Dataset(
                    'dummy', mode='r', memory=gzip_handle.read()
            ) as dataset_object:
                return _read_targets(dataset_object)

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    target_dict = _read_targets(dataset_object)
    dataset_object.close()
    return target_dict


def compress_file(netcdf_file_name):
    """Compresses file with predictor or target data (turns it into gzip file).

    :param netcdf_file_name: Path to NetCDF file with predictor or target data.
    :raises: ValueError: if file is already gzipped.
    """

    radar_io.compress_file(netcdf_file_name)


def subset_predictors_by_band(predictor_dict, band_numbers):
    """Subsets predictor (satellite) data by spectral band.

    :param predictor_dict: See doc for `read_predictor_file`.
    :param band_numbers: 1-D numpy array of desired band numbers (integers).
    :return: predictor_dict: Same as input but maybe with fewer bands.
    """

    error_checking.assert_is_integer_numpy_array(band_numbers)
    error_checking.assert_is_greater_numpy_array(band_numbers, 0)

    indices_to_keep = numpy.array([
        numpy.where(predictor_dict[BAND_NUMBERS_KEY] == n)[0][0]
        for n in band_numbers
    ], dtype=int)

    for this_key in ONE_PER_BAND_NUMBER_KEYS:
        if predictor_dict[this_key] is None:
            continue

        predictor_dict[this_key] = (
            predictor_dict[this_key][..., indices_to_keep]
        )

    return predictor_dict


def subset_grid(
        predictor_or_target_dict, first_row, last_row, first_column,
        last_column):
    """Subsets predictor or target data by grid points.

    :param predictor_or_target_dict: Dictionary with keys listed in either
        `read_predictor_file` or `read_target_file`.
    :param first_row: First row to keep (integer index).
    :param last_row: Last row to keep (integer index).
    :param first_column: First column to keep (integer index).
    :param last_column: Last column to keep (integer index).
    :return: predictor_or_target_dict: Same as input but with smaller grid.
    """

    num_rows = len(predictor_or_target_dict[LATITUDES_KEY])
    num_columns = len(predictor_or_target_dict[LONGITUDES_KEY])

    error_checking.assert_is_integer(first_row)
    error_checking.assert_is_geq(first_row, 0)
    error_checking.assert_is_integer(last_row)
    error_checking.assert_is_greater(last_row, first_row)
    error_checking.assert_is_less_than(last_row, num_rows)

    error_checking.assert_is_integer(first_column)
    error_checking.assert_is_geq(first_column, 0)
    error_checking.assert_is_integer(last_column)
    error_checking.assert_is_greater(last_column, first_column)
    error_checking.assert_is_less_than(last_column, num_columns)

    row_indices = numpy.linspace(
        first_row, last_row, num=last_row - first_row + 1, dtype=int
    )
    column_indices = numpy.linspace(
        first_column, last_column, num=last_column - first_column + 1, dtype=int
    )

    predictor_or_target_dict[LATITUDES_KEY] = (
        predictor_or_target_dict[LATITUDES_KEY][row_indices]
    )
    predictor_or_target_dict[LONGITUDES_KEY] = (
        predictor_or_target_dict[LONGITUDES_KEY][column_indices]
    )

    if TARGET_MATRIX_KEY in predictor_or_target_dict:
        predictor_or_target_dict[TARGET_MATRIX_KEY] = numpy.take(
            predictor_or_target_dict[TARGET_MATRIX_KEY],
            axis=1, indices=row_indices
        )
        predictor_or_target_dict[TARGET_MATRIX_KEY] = numpy.take(
            predictor_or_target_dict[TARGET_MATRIX_KEY],
            axis=2, indices=column_indices
        )

        predictor_or_target_dict[MASK_MATRIX_KEY] = numpy.take(
            predictor_or_target_dict[MASK_MATRIX_KEY],
            axis=0, indices=row_indices
        )
        predictor_or_target_dict[MASK_MATRIX_KEY] = numpy.take(
            predictor_or_target_dict[MASK_MATRIX_KEY],
            axis=1, indices=column_indices
        )

        return predictor_or_target_dict

    for this_key in ONE_PER_PREDICTOR_PIXEL_KEYS:
        predictor_or_target_dict[this_key] = numpy.take(
            predictor_or_target_dict[this_key], axis=1, indices=row_indices
        )
        predictor_or_target_dict[this_key] = numpy.take(
            predictor_or_target_dict[this_key], axis=2, indices=column_indices
        )

    return predictor_or_target_dict


def subset_by_time(predictor_or_target_dict, desired_times_unix_sec):
    """Subsets predictor or target data by time.

    T = number of desired times

    :param predictor_or_target_dict: Dictionary with keys listed in either
        `read_predictor_file` or `read_target_file`.
    :param desired_times_unix_sec: length-T numpy array of desired times.
    :return: predictor_or_target_dict: Same as input but maybe with fewer
        examples.
    :return: desired_indices: length-T numpy array with indices corresponding to
        desired times.
    """

    error_checking.assert_is_numpy_array(
        desired_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)

    desired_indices = numpy.array([
        numpy.where(predictor_or_target_dict[VALID_TIMES_KEY] == t)[0][0]
        for t in desired_times_unix_sec
    ], dtype=int)

    predictor_or_target_dict = subset_by_index(
        predictor_or_target_dict=predictor_or_target_dict,
        desired_indices=desired_indices
    )

    return predictor_or_target_dict, desired_indices


def subset_by_index(predictor_or_target_dict, desired_indices):
    """Subsets predictor or target data by index.

    :param predictor_or_target_dict: Dictionary with keys listed in either
        `read_predictor_file` or `read_target_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: predictor_or_target_dict: Same as input but maybe with fewer
        examples.
    """

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(predictor_or_target_dict[VALID_TIMES_KEY])
    )

    if PREDICTOR_MATRIX_UNNORM_KEY in predictor_or_target_dict:
        expected_keys = ONE_PER_PREDICTOR_TIME_KEYS
    else:
        expected_keys = ONE_PER_TARGET_TIME_KEYS

    for this_key in expected_keys:
        if predictor_or_target_dict[this_key] is None:
            continue

        predictor_or_target_dict[this_key] = (
            predictor_or_target_dict[this_key][desired_indices, ...]
        )

    return predictor_or_target_dict


def concat_predictor_data(predictor_dicts):
    """Concatenates many dictionaries with predictor data into one.

    :param predictor_dicts: List of dictionaries, each in the format returned by
        `read_predictor_file`.
    :return: predictor_dict: Single dictionary, also in the format returned by
        `read_predictor_file`.
    :raises: ValueError: if any two dictionaries have different band numbers,
        latitudes, or longitudes.
    """

    predictor_dict = copy.deepcopy(predictor_dicts[0])
    keys_to_match = [
        BAND_NUMBERS_KEY, LATITUDES_KEY, LONGITUDES_KEY, NORMALIZATION_FILE_KEY
    ]

    for i in range(1, len(predictor_dicts)):
        for this_key in keys_to_match:
            if this_key == BAND_NUMBERS_KEY:
                if numpy.array_equal(
                        predictor_dict[this_key], predictor_dicts[i][this_key]
                ):
                    continue
            elif this_key == NORMALIZATION_FILE_KEY:
                if predictor_dict[this_key] == predictor_dicts[i][this_key]:
                    continue
            else:
                if numpy.allclose(
                        predictor_dict[this_key], predictor_dicts[i][this_key],
                        atol=TOLERANCE
                ):
                    continue

            error_string = (
                '1st and {0:d}th dictionaries have different values for '
                '"{1:s}".  1st dictionary:\n{2:s}\n\n'
                '{0:d}th dictionary:\n{3:s}'
            ).format(
                i + 1, this_key,
                str(predictor_dict[this_key]),
                str(predictor_dicts[i][this_key])
            )

            raise ValueError(error_string)

    for i in range(1, len(predictor_dicts)):
        for this_key in ONE_PER_PREDICTOR_TIME_KEYS:
            if predictor_dict[this_key] is None:
                continue

            predictor_dict[this_key] = numpy.concatenate((
                predictor_dict[this_key], predictor_dicts[i][this_key]
            ), axis=0)

    return predictor_dict


def concat_target_data(target_dicts):
    """Concatenates many dictionaries with target data into one.

    :param target_dicts: List of dictionaries, each in the format returned by
        `read_target_file`.
    :return: target_dict: Single dictionary, also in the format returned by
        `read_target_file`.
    :raises: ValueError: if any two dictionaries have different masks,
        latitudes, or longitudes.
    """

    target_dict = copy.deepcopy(target_dicts[0])
    keys_to_match = [
        LATITUDES_KEY, LONGITUDES_KEY, MASK_MATRIX_KEY,
        FULL_LATITUDES_KEY, FULL_LONGITUDES_KEY, FULL_MASK_MATRIX_KEY
    ]

    for i in range(1, len(target_dicts)):
        for this_key in keys_to_match:
            if this_key in [MASK_MATRIX_KEY, FULL_MASK_MATRIX_KEY]:
                if numpy.array_equal(
                        target_dict[this_key], target_dicts[i][this_key]
                ):
                    continue
            else:
                if numpy.allclose(
                        target_dict[this_key], target_dicts[i][this_key],
                        atol=TOLERANCE
                ):
                    continue

            error_string = (
                '1st and {0:d}th dictionaries have different values for '
                '"{1:s}".  1st dictionary:\n{2:s}\n\n'
                '{0:d}th dictionary:\n{3:s}'
            ).format(
                i + 1, this_key,
                str(target_dict[this_key]),
                str(target_dicts[i][this_key])
            )

            raise ValueError(error_string)

    for i in range(1, len(target_dicts)):
        for this_key in ONE_PER_TARGET_TIME_KEYS:
            if target_dict[this_key] is None:
                continue

            target_dict[this_key] = numpy.concatenate((
                target_dict[this_key], target_dicts[i][this_key]
            ), axis=0)

    return target_dict
