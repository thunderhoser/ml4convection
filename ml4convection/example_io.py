"""Input/output methods for learning examples."""

import os
import sys
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
import normalization
import standalone_utils

DATE_FORMAT = '%Y%m%d'

PREDICTOR_MATRIX_UNNORM_KEY = 'predictor_matrix_unnorm'
PREDICTOR_MATRIX_NORM_KEY = 'predictor_matrix_norm'
PREDICTOR_MATRIX_UNIF_NORM_KEY = 'predictor_matrix_unif_norm'
VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
NORMALIZATION_FILE_KEY = 'normalization_file_name'

TARGET_MATRIX_KEY = 'target_matrix'
COMPOSITE_REFL_MATRIX_KEY = 'composite_refl_matrix_dbz'
COMPOSITE_REFL_THRESHOLD_KEY = 'composite_refl_threshold_dbz'

TIME_DIMENSION_KEY = 'time'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
BAND_DIMENSION_KEY = 'band'

BAND_NUMBERS_KEY = 'band_numbers'


def _process_predictors_one_day(
        input_file_name, spatial_downsampling_factor, norm_dict_for_count,
        normalization_file_name):
    """Processes predictors (satellite data) for one day.

    E = number of examples per batch
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    :param input_file_name: Path to input file (will be read by
        `satellite_io.read_file`).
    :param spatial_downsampling_factor: See doc for `process_predictors`.
    :param norm_dict_for_count: Dictionary returned by
        `normalization.read_file`.  Will use this to normalize data.
    :param normalization_file_name: See doc for `process_predictors`.

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
        netcdf_file_name=input_file_name, read_temperatures=False,
        read_counts=True
    )

    if spatial_downsampling_factor > 1:
        satellite_dict = downsample_data_in_space(
            satellite_dict=satellite_dict,
            downsampling_factor=spatial_downsampling_factor,
            change_coordinates=True
        )[0]

    predictor_matrix_unnorm = (
        satellite_dict[satellite_io.BRIGHTNESS_COUNT_KEY] + 0.
    )

    satellite_dict = normalization.normalize_data(
        satellite_dict=satellite_dict, uniformize=False,
        norm_dict_for_count=norm_dict_for_count
    )
    predictor_matrix_norm = (
        satellite_dict[satellite_io.BRIGHTNESS_COUNT_KEY] + 0.
    )
    satellite_dict[satellite_io.BRIGHTNESS_COUNT_KEY] = (
        predictor_matrix_unnorm + 0.
    )

    satellite_dict = normalization.normalize_data(
        satellite_dict=satellite_dict, uniformize=True,
        norm_dict_for_count=norm_dict_for_count
    )
    predictor_matrix_unif_norm = (
        satellite_dict[satellite_io.BRIGHTNESS_COUNT_KEY] + 0.
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


def _process_targets_one_day(input_file_name, spatial_downsampling_factor,
                             composite_refl_threshold_dbz):
    """Processes targets (radar data) for one day.

    E = number of examples per batch
    M = number of rows in grid
    N = number of columns in grid

    :param input_file_name: Path to input file (will be read by
        `radar_io.read_2d_file`).
    :param spatial_downsampling_factor: See doc for `process_targets`.
    :param composite_refl_threshold_dbz: Same.

    :return: target_dict: Dictionary with the following keys.
    target_dict['composite_refl_matrix_dbz']: E-by-M-by-N numpy array of
        composite reflectivities.
    target_dict['target_matrix']: E-by-M-by-N numpy array of target values
        (0 or 1), indicating when and where convection occurs.
    target_dict['valid_times_unix_sec']: length-E numpy array of valid times.
    target_dict['latitudes_deg_n']: length-M numpy array of latitudes
        (deg N).
    target_dict['longitudes_deg_e']: length-N numpy array of longitudes
        (deg E).
    target_dict['composite_refl_threshold_dbz']: Same as input (metadata).
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    radar_dict = radar_io.read_2d_file(input_file_name)

    if spatial_downsampling_factor > 1:
        radar_dict = downsample_data_in_space(
            radar_dict=radar_dict,
            downsampling_factor=spatial_downsampling_factor,
            change_coordinates=True
        )[1]

    composite_refl_matrix_dbz = radar_dict[radar_io.COMPOSITE_REFL_KEY]
    print('Mean reflectivity = {0:.2f} dBZ'.format(
        numpy.mean(composite_refl_matrix_dbz)
    ))

    target_matrix = (
        composite_refl_matrix_dbz >= composite_refl_threshold_dbz
    ).astype(int)

    print((
        'Number of target values = {0:d} ... event frequency = {1:.2g}'
    ).format(
        target_matrix.size, numpy.mean(target_matrix)
    ))

    return {
        COMPOSITE_REFL_MATRIX_KEY: composite_refl_matrix_dbz,
        TARGET_MATRIX_KEY: target_matrix,
        VALID_TIMES_KEY: radar_dict[satellite_io.VALID_TIMES_KEY],
        LATITUDES_KEY: radar_dict[satellite_io.LATITUDES_KEY],
        LONGITUDES_KEY: radar_dict[satellite_io.LONGITUDES_KEY],
        COMPOSITE_REFL_THRESHOLD_KEY: composite_refl_threshold_dbz
    }


def _write_predictor_file(predictor_dict, netcdf_file_name):
    """Writes predictors to NetCDF file.

    :param predictor_dict: Dictionary created by `_process_predictors_one_day`.
    :param netcdf_file_name: Path to output file.
    """

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


def _write_target_file(target_dict, netcdf_file_name):
    """Writes targets to NetCDF file.

    :param target_dict: Dictionary created by `_process_targets_one_day`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    target_matrix = target_dict[TARGET_MATRIX_KEY]
    num_times = target_matrix.shape[0]
    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]

    dataset_object.setncattr(
        COMPOSITE_REFL_THRESHOLD_KEY, target_dict[COMPOSITE_REFL_THRESHOLD_KEY]
    )

    dataset_object.createDimension(TIME_DIMENSION_KEY, num_times)
    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)

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

    these_dim = (TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    dataset_object.createVariable(
        COMPOSITE_REFL_MATRIX_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[COMPOSITE_REFL_MATRIX_KEY][:] = (
        target_dict[COMPOSITE_REFL_MATRIX_KEY]
    )

    dataset_object.close()


def downsample_data_in_space(downsampling_factor, change_coordinates=False,
                             satellite_dict=None, radar_dict=None):
    """Downsamples satellite and/or radar data in space.

    At least one of `satellite_dict` and `radar_dict` must be specified
    (not None).

    :param downsampling_factor: Downsampling factor (integer).
    :param change_coordinates: Boolean flag.  If True (False), will (not) change
        coordinates in dictionaries to reflect downsampling.
    :param satellite_dict: Dictionary in format returned by
        `satellite_io.read_file`.
    :param radar_dict: Dictionary in format returned by `radar_io.read_2d_file`.
    :return: satellite_dict: Same as input but maybe with coarser spatial
        resolution.
    :return: radar_dict: Same as input but maybe with coarser spatial
        resolution.
    :raises: ValueError: if `satellite_dict is None and radar_dict is None`.
    """

    error_checking.assert_is_integer(downsampling_factor)
    error_checking.assert_is_greater(downsampling_factor, 1)
    error_checking.assert_is_boolean(change_coordinates)

    if satellite_dict is None and radar_dict is None:
        raise ValueError(
            'At least one of satellite_dict and radar_dict must be specified '
            '(not None).'
        )

    if satellite_dict is not None:
        this_key = (
            satellite_io.BRIGHTNESS_COUNT_KEY
            if satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] is None
            else satellite_io.BRIGHTNESS_TEMP_KEY
        )
        satellite_dict[this_key] = standalone_utils.do_2d_pooling(
            feature_matrix=satellite_dict[this_key],
            window_size_px=downsampling_factor, do_max_pooling=False
        )

    if radar_dict is not None:
        composite_refl_matrix_dbz = numpy.expand_dims(
            radar_dict[radar_io.COMPOSITE_REFL_KEY], axis=-1
        )
        composite_refl_matrix_dbz = standalone_utils.do_2d_pooling(
            feature_matrix=composite_refl_matrix_dbz,
            window_size_px=downsampling_factor, do_max_pooling=True
        )
        radar_dict[radar_io.COMPOSITE_REFL_KEY] = (
            composite_refl_matrix_dbz[..., 0]
        )

    if not change_coordinates:
        return satellite_dict, radar_dict

    if satellite_dict is None:
        latitude_matrix_deg_n = numpy.expand_dims(
            radar_dict[radar_io.LATITUDES_KEY], axis=0
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
    if radar_dict is not None:
        radar_dict[radar_io.LATITUDES_KEY] = latitudes_deg_n + 0.

    if satellite_dict is None:
        longitude_matrix_deg_e = numpy.expand_dims(
            radar_dict[radar_io.LONGITUDES_KEY], axis=0
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
    if radar_dict is not None:
        radar_dict[radar_io.LONGITUDES_KEY] = longitudes_deg_e + 0.

    return satellite_dict, radar_dict


def process_predictors(
        top_input_dir_name, spatial_downsampling_factor,
        first_date_string, last_date_string, normalization_file_name,
        top_output_dir_name, raise_error_if_all_missing=True):
    """Processes predictors (satellite data).

    :param top_input_dir_name: Name of top-level directory with satellite
        data.  Files therein will be found by `satellite_io.find_file` and read
        by `satellite_io.read_file`.
    :param spatial_downsampling_factor: Downsampling factor (integer), used to
        coarsen spatial resolution.  If you do not want to coarsen spatial
        resolution, make this 1.
    :param first_date_string: First day (format "yyyymmdd").  Will process
        predictors for all days in `first_date_string`...`last_date_string`.
    :param last_date_string: See above.
    :param normalization_file_name: Path to file with normalization parameters
        (will be read by `normalization.read_file`).
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
    norm_dict_for_count = normalization.read_file(normalization_file_name)[1]

    for this_input_file_name in input_file_names:
        print('\n')

        this_predictor_dict = _process_predictors_one_day(
            input_file_name=this_input_file_name,
            spatial_downsampling_factor=spatial_downsampling_factor,
            norm_dict_for_count=norm_dict_for_count,
            normalization_file_name=normalization_file_name
        )

        this_output_file_name = find_predictor_file(
            top_directory_name=top_output_dir_name,
            date_string=satellite_io.file_name_to_date(this_input_file_name),
            raise_error_if_missing=False
        )

        print('Writing processed predictors to: "{0:s}"...'.format(
            this_output_file_name
        ))
        _write_predictor_file(
            predictor_dict=this_predictor_dict,
            netcdf_file_name=this_output_file_name
        )


def process_targets(
        top_input_dir_name, spatial_downsampling_factor,
        first_date_string, last_date_string, composite_refl_threshold_dbz,
        top_output_dir_name, raise_error_if_all_missing=True):
    """Processes targets (radar data).

    :param top_input_dir_name: Name of top-level directory with radar data.
        Files therein will be found by `radar_io.find_file` and read by
        `radar_io.read_2d_file`.
    :param spatial_downsampling_factor: See doc for `process_predictors`.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param composite_refl_threshold_dbz: Composite-reflectivity threshold.  Grid
        cells with composite (column-max) reflectivity >= threshold will be
        considered convective.
    :param raise_error_if_all_missing: Boolean flag.  If all input files are
        missing and `raise_error_if_all_missing == True`, will throw error.
    """

    error_checking.assert_is_integer(spatial_downsampling_factor)
    error_checking.assert_is_geq(spatial_downsampling_factor, 1)

    input_file_names = radar_io.find_many_files(
        top_directory_name=top_input_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string, with_3d=False,
        raise_error_if_all_missing=raise_error_if_all_missing,
        raise_error_if_any_missing=False
    )

    for i in range(len(input_file_names)):
        this_target_dict = _process_targets_one_day(
            input_file_name=input_file_names[i],
            spatial_downsampling_factor=spatial_downsampling_factor,
            composite_refl_threshold_dbz=composite_refl_threshold_dbz
        )

        this_output_file_name = find_target_file(
            top_directory_name=top_output_dir_name,
            date_string=radar_io.file_name_to_date(input_file_names[i]),
            raise_error_if_missing=False
        )

        print('Writing processed targets to: "{0:s}"...'.format(
            this_output_file_name
        ))
        _write_target_file(
            target_dict=this_target_dict, netcdf_file_name=this_output_file_name
        )

        if i != len(input_file_names) - 1:
            print('\n')


def find_predictor_file(top_directory_name, date_string,
                        raise_error_if_missing=True):
    """Finds NetCDF file with predictors.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param date_string: Date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: predictor_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)

    predictor_file_name = '{0:s}/{1:s}/predictors_{2:s}.nc'.format(
        top_directory_name, date_string[:4], date_string
    )

    if os.path.isfile(predictor_file_name) or not raise_error_if_missing:
        return predictor_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        predictor_file_name
    )
    raise ValueError(error_string)


def read_predictor_file(netcdf_file_name):
    """Reads predictors from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: predictor_dict: Dictionary in format returned by
        `_process_predictors_one_day`.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    predictor_dict = {
        PREDICTOR_MATRIX_UNNORM_KEY:
            dataset_object.variables[PREDICTOR_MATRIX_UNNORM_KEY][:],
        PREDICTOR_MATRIX_NORM_KEY:
            dataset_object.variables[PREDICTOR_MATRIX_NORM_KEY][:],
        PREDICTOR_MATRIX_UNIF_NORM_KEY:
            dataset_object.variables[PREDICTOR_MATRIX_UNIF_NORM_KEY][:],
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
        BAND_NUMBERS_KEY: dataset_object.variables[BAND_NUMBERS_KEY][:],
        NORMALIZATION_FILE_KEY:
            str(getattr(dataset_object, NORMALIZATION_FILE_KEY))
    }

    dataset_object.close()
    return predictor_dict


def find_target_file(top_directory_name, date_string,
                     raise_error_if_missing=True):
    """Finds NetCDF file with targets.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param date_string: Date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: target_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)

    target_file_name = '{0:s}/{1:s}/targets_{2:s}.nc'.format(
        top_directory_name, date_string[:4], date_string
    )

    if os.path.isfile(target_file_name) or not raise_error_if_missing:
        return target_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        target_file_name
    )
    raise ValueError(error_string)


def read_target_file(netcdf_file_name):
    """Reads targets from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: target_dict: Dictionary in format returned by
        `_process_targets_one_day`.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    predictor_dict = {
        COMPOSITE_REFL_MATRIX_KEY:
            dataset_object.variables[COMPOSITE_REFL_MATRIX_KEY][:],
        TARGET_MATRIX_KEY: dataset_object.variables[TARGET_MATRIX_KEY][:],
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
        COMPOSITE_REFL_THRESHOLD_KEY:
            str(getattr(dataset_object, COMPOSITE_REFL_THRESHOLD_KEY))
    }

    dataset_object.close()
    return predictor_dict
