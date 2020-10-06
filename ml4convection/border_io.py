"""Input/output methods for political borders."""

import os
import sys
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils

VERTEX_DIMENSION_KEY = 'vertex'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'


def _write_file(latitudes_deg_n, longitudes_deg_e, netcdf_file_name):
    """Writes borders to NetCDF file.

    P = number of points

    :param latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_points = len(latitudes_deg_n)
    dataset_object.createDimension(VERTEX_DIMENSION_KEY, num_points)

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32,
        dimensions=VERTEX_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32,
        dimensions=VERTEX_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = longitudes_deg_e

    dataset_object.close()


def read_file(netcdf_file_name=None):
    """Reads borders from NetCDF file.

    :param netcdf_file_name: Path to input file.  If None, will look for file in
        repository.
    :return: latitudes_deg_n: See doc for `write_file`.
    :return: longitudes_deg_e: Same.
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/borders.nc'.format(THIS_DIRECTORY_NAME)

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    latitudes_deg_n = numpy.array(
        dataset_object.variables[LATITUDES_KEY][:]
    )
    longitudes_deg_e = numpy.array(
        dataset_object.variables[LONGITUDES_KEY][:]
    )
    dataset_object.close()

    return latitudes_deg_n, longitudes_deg_e
