"""Plots results of model evaluation."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from gewittergefahr.gg_utils import file_system_utils
from ml4convection.utils import evaluation
from ml4convection.plotting import evaluation_plotting as eval_plotting

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

ADVANCED_SCORE_FILE_ARG_NAME = 'input_advanced_score_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ADVANCED_SCORE_FILE_HELP_STRING = (
    'Path to file with advanced evaluation scores.  Will be read by '
    '`evaluation.read_advanced_score_file`.'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(advanced_score_file_name, output_dir_name):
    """Plots results of model evaluation.

    This is effectively the main method.

    :param advanced_score_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    :raises: ValueError: if input file contains gridded scores.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(advanced_score_file_name))
    advanced_score_table_xarray = evaluation.read_advanced_score_file(
        advanced_score_file_name
    )

    num_grid_rows = len(
        advanced_score_table_xarray.coords[evaluation.LATITUDE_DIM].values
    )
    num_grid_columns = len(
        advanced_score_table_xarray.coords[evaluation.LONGITUDE_DIM].values
    )

    if num_grid_rows * num_grid_columns > 1:
        error_string = (
            'File should contain ungridded scores (aggregated over the full '
            'domain).  Instead, contains scores at {0:d} grid points.'
        ).format(num_grid_rows * num_grid_columns)

        raise ValueError(error_string)

    # Plot performance diagram.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    a = advanced_score_table_xarray

    eval_plotting.plot_performance_diagram(
        axes_object=axes_object,
        pod_by_threshold=a[evaluation.POD_KEY].values[0, 0, :],
        success_ratio_by_threshold=
        a[evaluation.SUCCESS_RATIO_KEY].values[0, 0, :]
    )

    area_under_curve = gg_model_eval.get_area_under_perf_diagram(
        pod_by_threshold=a[evaluation.POD_KEY].values[0, 0, :],
        success_ratio_by_threshold=
        a[evaluation.SUCCESS_RATIO_KEY].values[0, 0, :]
    )
    max_csi = numpy.nanmax(a[evaluation.CSI_KEY].values[0, 0, :])
    best_threshold_index = numpy.nanargmax(
        a[evaluation.CSI_KEY].values[0, 0, :]
    )
    best_prob_threshold = a.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values[
        best_threshold_index
    ]

    title_string = 'Area under curve = {0:.3f} ... max CSI = {1:.3f}'.format(
        area_under_curve, max_csi
    )
    axes_object.set_title(title_string)

    print(title_string)
    print('Corresponding prob threshold = {0:.4f}'.format(best_prob_threshold))

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
        mean_predictions=a[evaluation.MEAN_FORECAST_PROB_KEY].values[0, 0, :],
        mean_observations=a[evaluation.EVENT_FREQUENCY_KEY].values[0, 0, :],
        example_counts=a[evaluation.EXAMPLE_COUNT_KEY].values[0, 0, :],
        mean_value_in_training=
        a[evaluation.TRAINING_EVENT_FREQ_KEY].values[0, 0],
        min_value_to_plot=0., max_value_to_plot=1.
    )

    title_string = (
        'BS = {0:.3f} ... BSS = {1:.3f} ... REL = {2:.3f} ... RES = {3:.3f}'
    ).format(
        a[evaluation.BRIER_SCORE_KEY].values[0, 0],
        a[evaluation.BRIER_SKILL_SCORE_KEY].values[0, 0],
        a[evaluation.RELIABILITY_KEY].values[0, 0],
        a[evaluation.RESOLUTION_KEY].values[0, 0]
    )
    axes_object.set_title(title_string)

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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
