"""Plots predictions with uncertainty."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.utils import radar_utils
from ml4convection.machine_learning import neural_net
from ml4convection.plotting import plotting_utils
from ml4convection.plotting import prediction_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)
DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

FONT_SIZE = 30
TITLE_FONT_SIZE = 40
MASK_OUTLINE_COLOUR = numpy.full(3, 152. / 255)
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
DAILY_TIMES_ARG_NAME = 'daily_times_seconds'
PERCENTILE_LEVELS_ARG_NAME = 'percentile_levels'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will plot predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

DAILY_TIMES_HELP_STRING = (
    'List of times to plot for each day.  All values should be in the range '
    '0...86399.'
)
PERCENTILE_LEVELS_HELP_STRING = (
    'List of percentile levels (from 0...100).  For each time step, will plot '
    'convection probability at each of these levels.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DAILY_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=DAILY_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PERCENTILE_LEVELS_ARG_NAME, type=float, nargs='+', required=True,
    help=PERCENTILE_LEVELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_predictions_one_time(
        prediction_dict, example_index, border_latitudes_deg_n,
        border_longitudes_deg_e, mask_matrix, percentile_levels,
        output_dir_name):
    """Plots predictions (with uncertainty) for one time step.

    M = number of rows in grid
    N = number of columns in grid

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param example_index: Will plot [i]th example, where i = `example_index`.
    :param border_latitudes_deg_n: See doc for `_plot_predictions_one_day`.
    :param border_longitudes_deg_e: Same.
    :param mask_matrix: M-by-N numpy array of integers (0 or 1), where 1 means
        the grid point is unmasked.
    :param percentile_levels: See documentation at top of file.
    :param output_dir_name: Same.
    :return: output_file_name: Path to output file.
    """

    colour_map_object, colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=1., make_lowest_prob_grey=True
        )
    )
    latitudes_deg_n = prediction_dict[prediction_io.LATITUDES_KEY]
    longitudes_deg_e = prediction_dict[prediction_io.LONGITUDES_KEY]

    i = example_index
    valid_time_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY][i]
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...]

    # Plot mean.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_width=4
    )
    pyplot.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=4, linestyles='solid',
        axes=axes_object
    )

    prediction_plotting.plot_probabilistic(
        target_matrix=target_matrix,
        probability_matrix=
        prediction_io.get_mean_predictions(prediction_dict)[i, ...],
        figure_object=figure_object, axes_object=axes_object,
        min_latitude_deg_n=latitudes_deg_n[0],
        min_longitude_deg_e=longitudes_deg_e[0],
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    axes_object.set_title(
        'Mean convection probability', fontsize=TITLE_FONT_SIZE
    )
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
        meridian_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
        font_size=FONT_SIZE
    )

    this_file_name = '{0:s}/{1:s}_mean.jpg'.format(
        output_dir_name, valid_time_string
    )
    panel_file_names = [this_file_name]

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot standard deviation.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_width=4
    )
    pyplot.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=4, linestyles='solid',
        axes=axes_object
    )

    prediction_plotting.plot_probabilistic(
        target_matrix=target_matrix,
        probability_matrix=
        prediction_io.get_predictive_stdevs(prediction_dict)[i, ...],
        figure_object=figure_object, axes_object=axes_object,
        min_latitude_deg_n=latitudes_deg_n[0],
        min_longitude_deg_e=longitudes_deg_e[0],
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    axes_object.set_title(
        'Stdev of convection probability', fontsize=TITLE_FONT_SIZE
    )
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
        meridian_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
        font_size=FONT_SIZE
    )

    this_file_name = '{0:s}/{1:s}_stdev.jpg'.format(
        output_dir_name, valid_time_string
    )
    panel_file_names.append(this_file_name)

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    letter_label = 'b'

    for k in range(len(percentile_levels)):
        letter_label = chr(ord(letter_label) + 1)

        if prediction_dict[prediction_io.QUANTILE_LEVELS_KEY] is None:
            this_prob_matrix = numpy.percentile(
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...],
                q=percentile_levels[k], axis=-1
            )
        else:
            interp_object = interp1d(
                x=prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...],
                y=prediction_dict[prediction_io.QUANTILE_LEVELS_KEY],
                kind='linear', axis=-1, bounds_error=False, assume_sorted=True,
                fill_value='extrapolate'
            )

            this_prob_matrix = interp_object(0.01 * percentile_levels[k])
            this_prob_matrix = numpy.maximum(this_prob_matrix, 0.)
            this_prob_matrix = numpy.minimum(this_prob_matrix, 1.)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object, line_width=4
        )
        pyplot.contour(
            longitudes_deg_e, latitudes_deg_n, mask_matrix,
            numpy.array([0.999]), colors=(MASK_OUTLINE_COLOUR,),
            linewidths=4, linestyles='solid', axes=axes_object
        )

        prediction_plotting.plot_probabilistic(
            target_matrix=target_matrix, probability_matrix=this_prob_matrix,
            figure_object=figure_object, axes_object=axes_object,
            min_latitude_deg_n=latitudes_deg_n[0],
            min_longitude_deg_e=longitudes_deg_e[0],
            latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
            longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object
        )

        title_string = '{0:.1f}th percentile of convection prob'.format(
            percentile_levels[k]
        )
        axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=latitudes_deg_n,
            plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
            parallel_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
            meridian_spacing_deg=1. if len(latitudes_deg_n) > 300 else 0.5,
            font_size=FONT_SIZE
        )

        if k == len(percentile_levels) - 1:
            gg_plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object, data_matrix=this_prob_matrix,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical',
                extend_min=True, extend_max=False, font_size=FONT_SIZE
            )

        this_file_name = '{0:s}/{1:s}_percentile={2:014.10f}.jpg'.format(
            output_dir_name, valid_time_string, percentile_levels[k]
        )
        panel_file_names.append(this_file_name)

        print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
        figure_object.savefig(
            panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, valid_time_string
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    num_panels = len(panel_file_names)
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    for this_file_name in panel_file_names:
        os.remove(this_file_name)

    return concat_figure_file_name


def _plot_predictions_one_day(
        prediction_file_name, border_latitudes_deg_n, border_longitudes_deg_e,
        daily_times_seconds, percentile_levels, output_dir_name):
    """Plots predictions (with uncertainty) for one day.

    P = number of points in border set

    :param prediction_file_name: Path to prediction file.  Will be read by
        `prediction_io.read_file`.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param daily_times_seconds: See documentation at top of file.
    :param percentile_levels: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    mask_matrix = model_metadata_dict[neural_net.MASK_MATRIX_KEY]

    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    num_times = target_matrix.shape[0]
    for i in range(num_times):
        target_matrix[i, ...][mask_matrix == False] = 0

    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = target_matrix

    # TODO(thunderhoser): Put this code somewhere reusable.
    valid_times_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY]
    base_time_unix_sec = number_rounding.floor_to_nearest(
        valid_times_unix_sec[0], DAYS_TO_SECONDS
    )
    desired_times_unix_sec = numpy.round(
        base_time_unix_sec + daily_times_seconds
    ).astype(int)

    good_flags = numpy.array([
        t in valid_times_unix_sec for t in desired_times_unix_sec
    ], dtype=bool)

    if not numpy.any(good_flags):
        return

    desired_times_unix_sec = desired_times_unix_sec[good_flags]
    prediction_dict = prediction_io.subset_by_time(
        prediction_dict=prediction_dict,
        desired_times_unix_sec=desired_times_unix_sec
    )[0]

    num_examples = len(prediction_dict[prediction_io.VALID_TIMES_KEY])

    for i in range(num_examples):
        _plot_predictions_one_time(
            prediction_dict=prediction_dict, example_index=i,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix.astype(int),
            percentile_levels=percentile_levels,
            output_dir_name=output_dir_name
        )


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         daily_times_seconds, percentile_levels, output_dir_name):
    """Plots predictions with uncertainty.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_times_seconds: Same.
    :param percentile_levels: Same.
    :param output_dir_name: Same.
    """

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_geq_numpy_array(daily_times_seconds, 0)
    error_checking.assert_is_less_than_numpy_array(
        daily_times_seconds, DAYS_TO_SECONDS
    )

    date_strings = []

    for k in range(NUM_RADARS):
        if len(date_strings) == 0:
            prediction_file_names = prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                radar_number=k, prefer_zipped=True, allow_other_format=True,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=k > 0
            )

            date_strings = [
                prediction_io.file_name_to_date(f)
                for f in prediction_file_names
            ]
        else:
            prediction_file_names = [
                prediction_io.find_file(
                    top_directory_name=top_prediction_dir_name,
                    valid_date_string=d, radar_number=k,
                    prefer_zipped=True, allow_other_format=True,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

        this_output_dir_name = '{0:s}/radar{1:d}'.format(output_dir_name, k)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name
        )

        num_days = len(date_strings)

        for i in range(num_days):
            _plot_predictions_one_day(
                prediction_file_name=prediction_file_names[i],
                daily_times_seconds=daily_times_seconds,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                percentile_levels=percentile_levels,
                output_dir_name=this_output_dir_name
            )

            if not (i == num_days - 1 and k == NUM_RADARS - 1):
                print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        daily_times_seconds=numpy.array(
            getattr(INPUT_ARG_OBJECT, DAILY_TIMES_ARG_NAME), dtype=int
        ),
        percentile_levels=numpy.array(
            getattr(INPUT_ARG_OBJECT, PERCENTILE_LEVELS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
