"""IO methods for raw satellite data from CIRA."""

import os
import glob
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Maybe add equidistant coords as well?

TIME_FORMAT_IN_FILES = '%Y-%m-%dT%H:%M:%S'
TIME_FORMAT_IN_FILE_NAMES = '%Y%j%H%MM'

YEAR_REGEX = '[0-9][0-9][0-9][0-9]'
JULIAN_DAY_REGEX = '[0-3][0-9][0-9]'
HOUR_REGEX = '[0-2][0-9]'
MINUTE_REGEX = '[0-5][0-9]'

TOLERANCE = 1e-6
KM_TO_METRES = 1000.

NORTH_ATLANTIC_ID_STRING = 'AL'
SOUTH_ATLANTIC_ID_STRING = 'SL'
NORTHEAST_PACIFIC_ID_STRING = 'EP'
NORTH_CENTRAL_PACIFIC_ID_STRING = 'CP'
NORTHWEST_PACIFIC_ID_STRING = 'WP'
NORTH_INDIAN_ID_STRING = 'IO'
SOUTHERN_HEMISPHERE_ID_STRING = 'SH'

VALID_BASIN_ID_STRINGS = [
    NORTH_ATLANTIC_ID_STRING, SOUTH_ATLANTIC_ID_STRING,
    NORTHEAST_PACIFIC_ID_STRING, NORTH_CENTRAL_PACIFIC_ID_STRING,
    NORTHWEST_PACIFIC_ID_STRING, NORTH_INDIAN_ID_STRING,
    SOUTHERN_HEMISPHERE_ID_STRING
]

LATITUDE_DIM_ORIG = 'latitude'
LONGITUDE_DIM_ORIG = 'longitude'
TIME_DIM_ORIG = 'time'

SATELLITE_NUMBER_KEY_ORIG = 'satellite_number'
BAND_NUMBER_KEY_ORIG = 'band_id'
BAND_WAVELENGTH_KEY_ORIG = 'band_wavelength'
SATELLITE_LONGITUDE_KEY_ORIG = 'satellite_subpoint_longitude'
STORM_ID_KEY_ORIG = 'storm_atcfid'
STORM_TYPE_KEY_ORIG = 'storm_development_level'
STORM_NAME_KEY_ORIG = 'storm_name'
STORM_LATITUDE_KEY_ORIG = 'storm_latitude'
STORM_LONGITUDE_KEY_ORIG = 'storm_longitude'
STORM_INTENSITY_KEY_ORIG = 'storm_intensity'
STORM_INTENSITY_NUM_KEY_ORIG = 'storm_current_intensity_number'
STORM_SPEED_KEY_ORIG = 'storm_speed'
STORM_HEADING_KEY_ORIG = 'storm_heading'
STORM_DISTANCE_TO_LAND_KEY_ORIG = 'storm_distance_to_land'
STORM_RADIUS_KEY_ORIG = 'R5'
STORM_RADIUS_FRACTIONAL_KEY_ORIG = 'fR5'
SATELLITE_AZIMUTH_ANGLE_KEY_ORIG = 'satellite_azimuth_angle'
SATELLITE_ZENITH_ANGLE_KEY_ORIG = 'satellite_zenith_angle'
SOLAR_AZIMUTH_ANGLE_KEY_ORIG = 'solar_azimuth_angle'
SOLAR_ZENITH_ANGLE_KEY_ORIG = 'solar_zenith_angle'
SOLAR_ELEVATION_ANGLE_KEY_ORIG = 'solar_elevation_angle'
SOLAR_HOUR_ANGLE_KEY_ORIG = 'solar_hour_angle'
BRIGHTNESS_TEMPERATURE_KEY_ORIG = 'brightness_temperature'

GRID_ROW_DIM = 'grid_row'
GRID_COLUMN_DIM = 'grid_column'
TIME_DIM = 'valid_time_unix_sec'

SATELLITE_NUMBER_KEY = 'satellite_number'
BAND_NUMBER_KEY = 'band_number'
BAND_WAVELENGTH_KEY = 'band_wavelength_micrometres'
SATELLITE_LONGITUDE_KEY = 'satellite_longitude_deg_e'
STORM_ID_KEY = 'storm_id_string'
STORM_TYPE_KEY = 'storm_type_string'
STORM_NAME_KEY = 'storm_name'
STORM_LATITUDE_KEY = 'storm_latitude_deg_n'
STORM_LONGITUDE_KEY = 'storm_longitude_deg_e'
STORM_INTENSITY_KEY = 'storm_intensity_kt'
STORM_INTENSITY_NUM_KEY = 'storm_intensity_number'
STORM_SPEED_KEY = 'storm_speed_m_s01'
STORM_HEADING_KEY = 'storm_heading_deg'
STORM_DISTANCE_TO_LAND_KEY = 'storm_distance_to_land_metres'
STORM_RADIUS_VERSION1_KEY = 'storm_radius_version1_metres'
STORM_RADIUS_VERSION2_KEY = 'storm_radius_version2_metres'
STORM_RADIUS_FRACTIONAL_KEY = 'storm_radius_fractional'
SATELLITE_AZIMUTH_ANGLE_KEY = 'satellite_azimuth_angle_deg'
SATELLITE_ZENITH_ANGLE_KEY = 'satellite_zenith_angle_deg'
SOLAR_AZIMUTH_ANGLE_KEY = 'solar_azimuth_angle_deg'
SOLAR_ZENITH_ANGLE_KEY = 'solar_zenith_angle_deg'
SOLAR_ELEVATION_ANGLE_KEY = 'solar_elevation_angle_deg'
SOLAR_HOUR_ANGLE_KEY = 'solar_hour_angle_deg'
BRIGHTNESS_TEMPERATURE_KEY = 'brightness_temp_kelvins'
GRID_LATITUDE_KEY = 'grid_latitude_deg_n'
GRID_LONGITUDE_KEY = 'grid_longitude_deg_e'


def _singleton_to_array(input_var):
    """Converts singleton (unsized numpy array) to sized numpy array.

    :param input_var: Input variable (unsized or sized numpy array).
    :return: output_var: Output variable (sized numpy array).
    """

    try:
        _ = len(input_var)
        return input_var
    except TypeError:
        return numpy.array([input_var])


def _check_basin_id(basin_id_string):
    """Ensures that basin ID is valid.

    :param basin_id_string: Basin ID.
    :raises: ValueError: if `basin_id_strings not in VALID_BASIN_ID_STRINGS`.
    """

    error_checking.assert_is_string(basin_id_string)
    if basin_id_string in VALID_BASIN_ID_STRINGS:
        return

    error_string = (
        'Basin ID ("{0:s}") must be in the following list:\n{1:s}'
    ).format(basin_id_string, str(VALID_BASIN_ID_STRINGS))

    raise ValueError(error_string)


def get_cyclone_id(year, basin_id_string, cyclone_number):
    """Creates cyclone ID from metadata.

    :param year: Year (integer).
    :param basin_id_string: Basin ID (must be accepted by `_check_basin_id`).
    :param cyclone_number: Cyclone number (integer).
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_integer(year)
    error_checking.assert_is_geq(year, 0)
    _check_basin_id(basin_id_string)
    error_checking.assert_is_integer(cyclone_number)
    error_checking.assert_is_greater(cyclone_number, 0)

    return '{0:04d}{1:s}{2:02d}'.format(year, basin_id_string, cyclone_number)


def parse_cyclone_id(cyclone_id_string):
    """Parses metadata from cyclone ID.

    :param cyclone_id_string: Cyclone ID, formatted like "yyyybbcc", where yyyy
        is the year; bb is the basin ID; and cc is the cyclone number ([cc]th
        cyclone of the season in the given basin).
    :return: year: Year (integer).
    :return: basin_id_string: Basin ID.
    :return: cyclone_number: Cyclone number (integer).
    """

    error_checking.assert_is_string(cyclone_id_string)
    assert len(cyclone_id_string) == 8

    year = int(cyclone_id_string[:4])
    error_checking.assert_is_geq(year, 0)

    basin_id_string = cyclone_id_string[4:6]
    _check_basin_id(basin_id_string)

    cyclone_number = int(cyclone_id_string[6:])
    error_checking.assert_is_greater(cyclone_number, 0)

    return year, basin_id_string, cyclone_number


def find_file(
        top_directory_name, cyclone_id_string, valid_time_unix_sec,
        raise_error_if_missing=True):
    """Finds NetCDF file with satellite data.

    :param top_directory_name: Name of top-level directory with satellite data.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `parse_cyclone_id`).
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: satellite_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    satellite_file_name = '{0:s}/{1:s}/{2:s}/{3:s}{4:s}_{5:s}.nc'.format(
        top_directory_name, cyclone_id_string[:4], cyclone_id_string,
        cyclone_id_string[4:], cyclone_id_string[2:4],
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
        )
    )

    if os.path.isfile(satellite_file_name) or not raise_error_if_missing:
        return satellite_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        satellite_file_name
    )
    raise ValueError(error_string)


def find_files_one_cyclone(top_directory_name, cyclone_id_string,
                           raise_error_if_all_missing=True):
    """Finds all NetCDF files with satellite data for one cyclone.

    :param top_directory_name: Name of top-level directory with satellite data.
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `parse_cyclone_id`).
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing == True`, will throw error.  If no files are
        found and `raise_error_if_all_missing == False`, will return empty list.
    :return: satellite_file_names: List of file paths.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    parse_cyclone_id(cyclone_id_string)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    file_pattern = (
        '{0:s}/{1:s}/{2:s}/{3:s}{4:s}_{5:s}{6:s}{7:s}{8:s}M.nc'
    ).format(
        top_directory_name, cyclone_id_string[:4], cyclone_id_string,
        cyclone_id_string[4:], cyclone_id_string[2:4],
        YEAR_REGEX, JULIAN_DAY_REGEX, HOUR_REGEX, MINUTE_REGEX
    )

    satellite_file_names = glob.glob(file_pattern)
    satellite_file_names.sort()

    if raise_error_if_all_missing and len(satellite_file_names) == 0:
        error_string = 'Could not find any files with pattern: "{0:s}"'.format(
            file_pattern
        )
        raise ValueError(error_string)

    return satellite_file_names


def find_cyclones_one_year(top_directory_name, year,
                           raise_error_if_all_missing=True):
    """Finds all cyclones in one year.

    :param top_directory_name: Name of top-level directory with satellite data.
    :param year: Year (integer).
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    directory_pattern = '{0:s}/{1:04d}/{1:04d}[A-Z][A-Z][0-9][0-9]'.format(
        top_directory_name, year
    )
    directory_names = glob.glob(directory_pattern)
    cyclone_id_strings = []

    for this_directory_name in directory_names:
        this_cyclone_id_string = this_directory_name.split('/')[-1]

        try:
            parse_cyclone_id(this_cyclone_id_string)
            cyclone_id_strings.append(this_cyclone_id_string)
        except:
            pass

    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs from directories with pattern: '
            '"{0:s}"'
        ).format(directory_pattern)

        raise ValueError(error_string)

    return cyclone_id_strings


def file_name_to_cyclone_id(satellite_file_name):
    """Parses cyclone ID from name of file with satellite data.

    :param satellite_file_name: File path.
    :return: cyclone_id_string: Cyclone ID.
    """

    error_checking.assert_is_string(satellite_file_name)
    cyclone_id_string = satellite_file_name.split('/')[-2]
    parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def read_file(netcdf_file_name):
    """Reads satellite data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: orig_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    orig_table_xarray = xarray.open_dataset(netcdf_file_name)

    valid_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(
            str(t).split('.')[0], TIME_FORMAT_IN_FILES
        )
        for t in orig_table_xarray.coords[TIME_DIM_ORIG].values
    ], dtype=int)

    grid_latitude_matrix_deg_n = numpy.expand_dims(
        orig_table_xarray.coords[LATITUDE_DIM_ORIG].values, axis=0
    )
    grid_longitude_matrix_deg_e = numpy.expand_dims(
        orig_table_xarray.coords[LONGITUDE_DIM_ORIG].values, axis=0
    )
    grid_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitude_matrix_deg_e, allow_nan=False
    )

    num_times = len(valid_times_unix_sec)
    grid_latitude_matrix_deg_n = numpy.repeat(
        grid_latitude_matrix_deg_n, axis=0, repeats=num_times
    )
    grid_longitude_matrix_deg_e = numpy.repeat(
        grid_longitude_matrix_deg_e, axis=0, repeats=num_times
    )

    num_grid_rows = grid_latitude_matrix_deg_n.shape[1]
    num_grid_columns = grid_longitude_matrix_deg_e.shape[1]
    grid_row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=int
    )
    grid_column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=int
    )

    metadata_dict = {
        GRID_ROW_DIM: grid_row_indices,
        GRID_COLUMN_DIM: grid_column_indices,
        TIME_DIM: valid_times_unix_sec
    }

    satellite_numbers_float = _singleton_to_array(
        orig_table_xarray[SATELLITE_NUMBER_KEY_ORIG].values
    )
    satellite_numbers = numpy.round(satellite_numbers_float).astype(int)
    assert numpy.allclose(
        satellite_numbers, satellite_numbers_float, atol=TOLERANCE
    )

    band_numbers_float = _singleton_to_array(
        orig_table_xarray[BAND_NUMBER_KEY_ORIG].values
    )
    band_numbers = numpy.round(band_numbers_float).astype(int)
    assert numpy.allclose(band_numbers, band_numbers_float, atol=TOLERANCE)

    satellite_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=_singleton_to_array(
            orig_table_xarray[SATELLITE_LONGITUDE_KEY_ORIG].values
        ),
        allow_nan=False
    )

    try:
        storm_id_strings = numpy.array([
            s for s in orig_table_xarray[STORM_ID_KEY_ORIG].values
        ], dtype='S10')

        storm_type_strings = numpy.array([
            s for s in orig_table_xarray[STORM_TYPE_KEY_ORIG].values
        ], dtype='S10')

        storm_names = numpy.array([
            s for s in orig_table_xarray[STORM_NAME_KEY_ORIG].values
        ], dtype='S10')
    except TypeError:
        storm_id_strings = numpy.array(
            [orig_table_xarray[STORM_ID_KEY_ORIG].values], dtype='S10'
        )
        storm_type_strings = numpy.array(
            [orig_table_xarray[STORM_TYPE_KEY_ORIG].values], dtype='S10'
        )
        storm_names = numpy.array(
            [orig_table_xarray[STORM_NAME_KEY_ORIG].values], dtype='S10'
        )

    storm_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=_singleton_to_array(
            orig_table_xarray[STORM_LONGITUDE_KEY_ORIG].values
        ),
        allow_nan=False
    )
    storm_distances_to_land_metres = _singleton_to_array(
        orig_table_xarray[STORM_DISTANCE_TO_LAND_KEY_ORIG].values * KM_TO_METRES
    )
    storm_radii_metres = (
        orig_table_xarray[STORM_RADIUS_KEY_ORIG].values[0] * KM_TO_METRES
    )

    these_dim = (TIME_DIM,)

    main_data_dict = {
        SATELLITE_NUMBER_KEY: (these_dim, satellite_numbers),
        BAND_NUMBER_KEY: (these_dim, band_numbers),
        BAND_WAVELENGTH_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[BAND_WAVELENGTH_KEY_ORIG].values
            )
        ),
        SATELLITE_LONGITUDE_KEY: (these_dim, satellite_longitudes_deg_e),
        STORM_ID_KEY: (these_dim, storm_id_strings),
        STORM_TYPE_KEY: (these_dim, storm_type_strings),
        STORM_NAME_KEY: (these_dim, storm_names),
        STORM_LATITUDE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_LATITUDE_KEY_ORIG].values
            )
        ),
        STORM_LONGITUDE_KEY: (these_dim, storm_longitudes_deg_e),
        STORM_INTENSITY_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_INTENSITY_KEY_ORIG].values
            )
        ),
        STORM_INTENSITY_NUM_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_INTENSITY_NUM_KEY_ORIG].values
            )
        ),
        STORM_SPEED_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_SPEED_KEY_ORIG].values
            )
        ),
        STORM_HEADING_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_HEADING_KEY_ORIG].values
            )
        ),
        STORM_DISTANCE_TO_LAND_KEY: (these_dim, storm_distances_to_land_metres),
        STORM_RADIUS_VERSION1_KEY: (these_dim, storm_radii_metres[[0]]),
        STORM_RADIUS_VERSION2_KEY: (these_dim, storm_radii_metres[[1]]),
        STORM_RADIUS_FRACTIONAL_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[STORM_RADIUS_FRACTIONAL_KEY_ORIG].values
            )
        ),
        SATELLITE_AZIMUTH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SATELLITE_AZIMUTH_ANGLE_KEY_ORIG].values
            )
        ),
        SATELLITE_ZENITH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SATELLITE_ZENITH_ANGLE_KEY_ORIG].values
            )
        ),
        SOLAR_AZIMUTH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_AZIMUTH_ANGLE_KEY_ORIG].values
            )
        ),
        SOLAR_ZENITH_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_ZENITH_ANGLE_KEY_ORIG].values
            )
        ),
        SOLAR_ELEVATION_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_ELEVATION_ANGLE_KEY_ORIG].values
            )
        ),
        SOLAR_HOUR_ANGLE_KEY: (
            these_dim,
            _singleton_to_array(
                orig_table_xarray[SOLAR_HOUR_ANGLE_KEY_ORIG].values
            )
        ),
        BRIGHTNESS_TEMPERATURE_KEY: (
            (TIME_DIM, GRID_ROW_DIM, GRID_COLUMN_DIM),
            orig_table_xarray[BRIGHTNESS_TEMPERATURE_KEY_ORIG].values
        ),
        GRID_LATITUDE_KEY: (
            (TIME_DIM, GRID_ROW_DIM),
            grid_latitude_matrix_deg_n
        ),
        GRID_LONGITUDE_KEY: (
            (TIME_DIM, GRID_COLUMN_DIM),
            grid_longitude_matrix_deg_e
        )
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=metadata_dict)


def concat_tables_over_time(satellite_tables_xarray):
    """Concatenates tables with satellite data over the time dimension.

    :param satellite_tables_xarray: 1-D list of xarray tables in format returned
        by `read_file`.
    :return: satellite_table_xarray: One xarray table, in format returned by
        `read_file`, created by concatenating inputs.
    """

    for this_table_xarray in satellite_tables_xarray[1:]:
        assert numpy.array_equal(
            satellite_tables_xarray[0].coords[GRID_ROW_DIM].values,
            this_table_xarray.coords[GRID_ROW_DIM].values
        )
        assert numpy.array_equal(
            satellite_tables_xarray[0].coords[GRID_COLUMN_DIM].values,
            this_table_xarray.coords[GRID_COLUMN_DIM].values
        )

    return xarray.concat(objs=satellite_tables_xarray, dim=TIME_DIM)
