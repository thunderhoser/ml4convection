"""Plots results of model evaluation."""

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

import gg_model_evaluation as gg_model_eval
import file_system_utils
import error_checking
import evaluation
import evaluation_plotting as eval_plotting

TOLERANCE = 1e-6

BOUNDING_BOX_DICT = {
    'facecolor': 'white',
    'alpha': 0.5,
    'edgecolor': 'black',
    'linewidth': 2,
    'boxstyle': 'round'
}

MARKER_TYPE = '*'
MARKER_SIZE = 50
MARKER_EDGE_WIDTH = 0

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

ADVANCED_SCORE_FILE_ARG_NAME = 'input_advanced_score_file_name'
BEST_THRESHOLD_ARG_NAME = 'best_prob_threshold'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ADVANCED_SCORE_FILE_HELP_STRING = (
    'Path to file with advanced evaluation scores.  Will be read by '
    '`evaluation.read_advanced_score_file`.'
)
BEST_THRESHOLD_HELP_STRING = (
    'Best probability threshold (ideally chosen based on validation data).  '
    'Will be marked with a star in the performance diagram.  If you want to let'
    ' this script choose the best probability threshold (e.g., if you are '
    'plotting validation data), leave this argument empty.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ADVANCED_SCORE_FILE_ARG_NAME, type=str, required=True,
    help=ADVANCED_SCORE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BEST_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=BEST_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(advanced_score_file_name, best_prob_threshold, output_dir_name):
    """Plots results of model evaluation.

    This is effectively the main method.

    :param advanced_score_file_name: See documentation at top of file.
    :param best_prob_threshold: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if input file contains gridded scores.
    """

    if best_prob_threshold < 0:
        best_prob_threshold = None
    if best_prob_threshold is not None:
        error_checking.assert_is_leq(best_prob_threshold, 1.)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(advanced_score_file_name))
    advanced_score_table_xarray = evaluation.read_advanced_score_file(
        advanced_score_file_name
    )

    gridded = evaluation.LATITUDE_DIM in advanced_score_table_xarray.coords

    if gridded:
        raise ValueError(
            'File should contain ungridded scores (aggregated over the full '
            'domain).'
        )

    # Plot performance diagram.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    a = advanced_score_table_xarray

    eval_plotting.plot_performance_diagram(
        axes_object=axes_object,
        pod_by_threshold=a[evaluation.POD_KEY].values,
        success_ratio_by_threshold=a[evaluation.SUCCESS_RATIO_KEY].values
    )

    area_under_curve = gg_model_eval.get_area_under_perf_diagram(
        pod_by_threshold=a[evaluation.POD_KEY].values,
        success_ratio_by_threshold=a[evaluation.SUCCESS_RATIO_KEY].values
    )

    if best_prob_threshold is None:
        best_threshold_index = numpy.nanargmin(
            numpy.absolute(a[evaluation.FREQUENCY_BIAS_KEY].values - 1.)
        )
        max_csi = numpy.nanmax(a[evaluation.CSI_KEY].values)
        csi_at_best_threshold = (
            a[evaluation.CSI_KEY].values[best_threshold_index]
        )

        if csi_at_best_threshold < 0.9 * max_csi:
            best_threshold_index = numpy.nanargmax(a[evaluation.CSI_KEY].values)
    else:
        threshold_diffs = numpy.absolute(
            a.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values -
            best_prob_threshold
        )
        best_threshold_index = numpy.where(threshold_diffs <= TOLERANCE)[0][0]

    best_prob_threshold = a.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values[
        best_threshold_index
    ]
    csi_at_best_threshold = a[evaluation.CSI_KEY].values[best_threshold_index]
    bias_at_best_threshold = (
        a[evaluation.FREQUENCY_BIAS_KEY].values[best_threshold_index]
    )

    annotation_string = (
        'Area under curve = {0:.3g}\n'
        'Best prob threshold = {1:.2g}\n'
        'CSI = {2:.3g}\n'
        'Frequency bias = {3:.3g}'
    ).format(
        area_under_curve, best_prob_threshold,
        csi_at_best_threshold, bias_at_best_threshold
    )

    print(annotation_string)

    axes_object.text(
        0.98, 0.98, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
        horizontalalignment='right', verticalalignment='top',
        transform=axes_object.transAxes
    )

    axes_object.plot(
        a[evaluation.SUCCESS_RATIO_KEY].values[best_threshold_index],
        a[evaluation.POD_KEY].values[best_threshold_index],
        linestyle='None', marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        markerfacecolor=eval_plotting.PERF_DIAGRAM_COLOUR,
        markeredgecolor=eval_plotting.PERF_DIAGRAM_COLOUR
    )

    axes_object.set_title('Performance diagram')

    figure_file_name = '{0:s}/performance_diagram.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot attributes diagram.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=a[evaluation.BINNED_MEAN_PROBS_KEY].values,
        mean_observations=a[evaluation.BINNED_EVENT_FREQS_KEY].values,
        example_counts=a[evaluation.BINNED_NUM_EXAMPLES_KEY].values,
        mean_value_in_training=a[evaluation.TRAINING_EVENT_FREQ_KEY].values[0],
        min_value_to_plot=0., max_value_to_plot=1.
    )

    axes_object.set_title('Attributes diagram')

    # annotation_string = (
    #     'Brier score = {0:.2g}\n'
    #     'Brier skill score = {1:.2g}\n'
    #     'Reliability = {2:.2g}\n'
    #     'Resolution = {3:.2g}'
    # ).format(
    #     a[evaluation.BRIER_SCORE_KEY].values[0],
    #     a[evaluation.BRIER_SKILL_SCORE_KEY].values[0],
    #     a[evaluation.RELIABILITY_KEY].values[0],
    #     a[evaluation.RESOLUTION_KEY].values[0]
    # )

    annotation_string = (
        'Brier score = {0:.2g}\n'
        'Brier skill score = {1:.2g}'
    ).format(
        a[evaluation.BRIER_SCORE_KEY].values[0],
        a[evaluation.BRIER_SKILL_SCORE_KEY].values[0]
    )

    axes_object.text(
        0.98, 0.02, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
        horizontalalignment='right', verticalalignment='bottom',
        transform=axes_object.transAxes
    )

    figure_file_name = '{0:s}/attributes_diagram.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        advanced_score_file_name=getattr(
            INPUT_ARG_OBJECT, ADVANCED_SCORE_FILE_ARG_NAME
        ),
        best_prob_threshold=getattr(INPUT_ARG_OBJECT, BEST_THRESHOLD_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
