"""USE ONCE AND DESTROY."""

import os
import sys
import glob

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation

CLIMO_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4convection_project/targets'
)


def _run():
    """USE ONCE AND DESTROY."""

    climo_file_names = glob.glob('{0:s}/climo*.p'.format(CLIMO_DIR_NAME))

    for this_file_name in climo_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))

        (
            event_frequency_overall,
            event_frequency_by_hour,
            event_frequency_by_month
        ) = evaluation.read_climo_from_file(this_file_name)

        print(event_frequency_overall)
        print(event_frequency_by_hour)
        print(event_frequency_by_month)
        print('\n\n')


if __name__ == '__main__':
    _run()
