"""USE ONCE AND DESTROY."""

import os
import sys
import glob

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import radar_io

TOP_SATELLITE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4convection_project/'
    'satellite_data'
)


def _run():
    """USE ONCE AND DESTROY."""

    satellite_file_names = glob.glob(
        '{0:s}/201*/satellite*.nc'.format(TOP_SATELLITE_DIR_NAME)
    )

    for this_file_name in satellite_file_names:
        print(this_file_name)
        radar_io.compress_file(this_file_name)
        # os.remove(this_file_name)


if __name__ == '__main__':
    _run()
