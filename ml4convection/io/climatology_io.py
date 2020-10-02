"""Input/output methods for climatology files."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_HOURS_PER_DAY = 24
NUM_MONTHS_PER_YEAR = 12

EVENT_FREQ_OVERALL_KEY = 'event_frequency_overall'
EVENT_FREQ_BY_HOUR_KEY = 'event_frequency_by_hour'
EVENT_FREQ_BY_MONTH_KEY = 'event_frequency_by_month'
EVENT_FREQ_BY_PIXEL_KEY = 'event_frequency_by_pixel'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'

ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
HOUR_DIMENSION_KEY = 'hour'
MONTH_DIMENSION_KEY = 'month'


def write_file(
        event_frequency_overall, event_frequency_by_hour,
        event_frequency_by_month, event_frequency_by_pixel,
        latitudes_deg_n, longitudes_deg_e, netcdf_file_name):
    """Writes climatology (event frequencies in training data) to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param event_frequency_overall: Overall event frequency (fraction of
        convective pixels).
    :param event_frequency_by_hour: length-24 numpy array of hourly frequencies.
    :param event_frequency_by_month: length-12 numpy array of monthly
        frequencies.
    :param event_frequency_by_pixel: M-by-N numpy array of event frequencies by
        pixel.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param netcdf_file_name: Path to output file.
    """

    # Check input args.
    error_checking.assert_is_greater(event_frequency_overall, 0.)
    error_checking.assert_is_less_than(event_frequency_overall, 1.)

    error_checking.assert_is_numpy_array(
        event_frequency_by_hour,
        exact_dimensions=numpy.array([NUM_HOURS_PER_DAY], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(
        event_frequency_by_hour, 0.
    )
    error_checking.assert_is_less_than_numpy_array(
        event_frequency_by_hour, 1.
    )

    error_checking.assert_is_numpy_array(
        event_frequency_by_month,
        exact_dimensions=numpy.array([NUM_MONTHS_PER_YEAR], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(
        event_frequency_by_month, 0.
    )
    error_checking.assert_is_less_than_numpy_array(
        event_frequency_by_month, 1.
    )

    error_checking.assert_is_numpy_array(
        event_frequency_by_pixel, num_dimensions=2
    )
    error_checking.assert_is_greater_numpy_array(
        event_frequency_by_pixel, 0., allow_nan=True
    )
    error_checking.assert_is_less_than_numpy_array(
        event_frequency_by_pixel, 1., allow_nan=True
    )

    num_grid_rows = event_frequency_by_pixel.shape[0]
    num_grid_columns = event_frequency_by_pixel.shape[1]

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)
    error_checking.assert_is_numpy_array(
        latitudes_deg_n,
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )

    longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg_e, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        longitudes_deg_e,
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )

    # Write file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(EVENT_FREQ_OVERALL_KEY, event_frequency_overall)

    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)
    dataset_object.createDimension(HOUR_DIMENSION_KEY, NUM_HOURS_PER_DAY)
    dataset_object.createDimension(MONTH_DIMENSION_KEY, NUM_MONTHS_PER_YEAR)

    dataset_object.createVariable(
        EVENT_FREQ_BY_HOUR_KEY, datatype=numpy.float32,
        dimensions=HOUR_DIMENSION_KEY
    )
    dataset_object.variables[EVENT_FREQ_BY_HOUR_KEY][:] = (
        event_frequency_by_hour
    )

    dataset_object.createVariable(
        EVENT_FREQ_BY_MONTH_KEY, datatype=numpy.float32,
        dimensions=MONTH_DIMENSION_KEY
    )
    dataset_object.variables[EVENT_FREQ_BY_MONTH_KEY][:] = (
        event_frequency_by_month
    )

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32, dimensions=COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = longitudes_deg_e

    dataset_object.createVariable(
        EVENT_FREQ_BY_PIXEL_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[EVENT_FREQ_BY_PIXEL_KEY][:] = (
        event_frequency_by_pixel
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads climatology (event frequencies in training data) from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: climo_dict: Dictionary with the following keys.
    climo_dict['event_frequency_overall']: See doc for `write_file`.
    climo_dict['event_frequency_by_hour']: Same.
    climo_dict['event_frequency_by_month']: Same.
    climo_dict['event_frequency_by_pixel']: Same.
    climo_dict['latitudes_deg_n']: Same.
    climo_dict['longitudes_deg_e']: Same.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    climo_dict = {
        EVENT_FREQ_OVERALL_KEY: getattr(dataset_object, EVENT_FREQ_OVERALL_KEY),
        EVENT_FREQ_BY_HOUR_KEY:
            dataset_object.variables[EVENT_FREQ_BY_HOUR_KEY][:],
        EVENT_FREQ_BY_MONTH_KEY:
            dataset_object.variables[EVENT_FREQ_BY_MONTH_KEY][:],
        EVENT_FREQ_BY_PIXEL_KEY:
            dataset_object.variables[EVENT_FREQ_BY_PIXEL_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:]
    }

    dataset_object.close()
    return climo_dict
