"""Creates spread-skill plot and saves plot to image file."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import uq_evaluation
import uq_evaluation_plotting as uq_eval_plotting

FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `uq_evaluation.read_spread_vs_skill`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Figure will be saved as an image here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, output_file_name):
    """Creates spread-skill plot and saves plot to image file.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    spread_skill_dict = uq_evaluation.read_spread_vs_skill(input_file_name)

    figure_object, axes_object = uq_eval_plotting.plot_spread_vs_skill(
        mean_prediction_stdevs=
        spread_skill_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY],
        rmse_values=spread_skill_dict[uq_evaluation.RMSE_VALUES_KEY]
    )

    if spread_skill_dict[uq_evaluation.USE_MEDIAN_KEY]:
        axes_object.set_ylabel('Skill (RMSE of median prediction)')
    else:
        axes_object.set_ylabel('Skill (RMSE of mean prediction)')

    half_window_size_px = int(numpy.round(
        spread_skill_dict[uq_evaluation.HALF_WINDOW_SIZE_KEY]
    ))
    axes_object.set_title(
        'Spread-skill plot for {0:d}-by-{0:d} eval neigh'.format(
            2 * half_window_size_px + 1
        )
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
