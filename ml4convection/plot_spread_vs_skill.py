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
    result_dict = uq_evaluation.read_spread_vs_skill(input_file_name)

    figure_object, axes_object = uq_eval_plotting.plot_spread_vs_skill(
        result_dict=result_dict
    )

    if result_dict[uq_evaluation.USE_MEDIAN_KEY]:
        axes_object.set_ylabel('Skill (RMSE of median prediction)')
    else:
        axes_object.set_ylabel('Skill (RMSE of mean prediction)')

    axes_object.set_xlabel('Spread (stdev of predictive distribution)')

    title_string = (
        'Spread-skill plot for {0:s}\n(SSREL = {1:.2g}; SSRAT = {2:.2g}; '
        'CRPS = {3:s})'
    ).format(
        'QR' if 'qfss' in input_file_name else 'MC dropout',
        result_dict[uq_evaluation.SPREAD_SKILL_RELIABILITY_KEY],
        result_dict[uq_evaluation.OVERALL_SPREAD_SKILL_RATIO_KEY],
        '0.034' if 'qfss' in input_file_name else '0.020'
    )

    print(title_string)
    axes_object.set_title(title_string)

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
