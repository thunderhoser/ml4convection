"""Makes figure with ungridded model evaluation."""

import os
import argparse
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 200
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 3
LARGE_BORDER_WIDTH_PX = 225
SMALL_BORDER_WIDTH_PX = 10
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

TEMPORAL_DIR_ARG_NAME = 'input_temporal_eval_dir_name'
NON_TEMPORAL_DIR_ARG_NAME = 'input_non_temporal_eval_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TEMPORAL_DIR_HELP_STRING = (
    'Name of directory with temporal-evaluation figures, created by '
    'plot_evaluation_by_time.py.'
)
NON_TEMPORAL_DIR_HELP_STRING = (
    'Name of directory with non-temporal-evaluation figures, created by '
    'plot_evaluation.py.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  New figures (most importantly, the 8-panel '
    'figure with all ungridded evaluation) will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORAL_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NON_TEMPORAL_DIR_ARG_NAME, type=str, required=True,
    help=NON_TEMPORAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Overlays text on image.

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _run(temporal_eval_dir_name, non_temporal_eval_dir_name, output_dir_name):
    """Makes figure with ungridded model evaluation.

    This is effectively the main method.

    :param temporal_eval_dir_name: See documentation at top of file.
    :param non_temporal_eval_dir_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    panel_file_names = [
        '{0:s}/attributes_diagram.jpg'.format(non_temporal_eval_dir_name),
        '{0:s}/performance_diagram.jpg'.format(non_temporal_eval_dir_name),
        '{0:s}/monthly_performance_diagrams.jpg'.format(temporal_eval_dir_name),
        '{0:s}/hourly_performance_diagrams.jpg'.format(temporal_eval_dir_name),
        '{0:s}/monthly_reliability_curves.jpg'.format(temporal_eval_dir_name),
        '{0:s}/hourly_reliability_curves.jpg'.format(temporal_eval_dir_name),
        '{0:s}/monthly_scores.jpg'.format(temporal_eval_dir_name),
        '{0:s}/hourly_scores.jpg'.format(temporal_eval_dir_name)
    ]

    pathless_panel_file_names = [os.path.split(f)[-1] for f in panel_file_names]

    resized_panel_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, f)
        for f in pathless_panel_file_names
    ]

    letter_label = None

    for i in range(len(panel_file_names)):
        print('Resizing panel and saving to: "{0:s}"...'.format(
            resized_panel_file_names[i]
        ))

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        if 'scores.jpg' in pathless_panel_file_names[i]:
            imagemagick_utils.trim_whitespace(
                input_file_name=panel_file_names[i],
                output_file_name=resized_panel_file_names[i],
                border_width_pixels=SMALL_BORDER_WIDTH_PX
            )
            _overlay_text(
                image_file_name=resized_panel_file_names[i],
                x_offset_from_left_px=0,
                y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
                text_string='({0:s})'.format(letter_label)
            )
        else:
            imagemagick_utils.trim_whitespace(
                input_file_name=panel_file_names[i],
                output_file_name=resized_panel_file_names[i],
                border_width_pixels=LARGE_BORDER_WIDTH_PX
            )
            _overlay_text(
                image_file_name=resized_panel_file_names[i],
                x_offset_from_left_px=LARGE_BORDER_WIDTH_PX,
                y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
                text_string='({0:s})'.format(letter_label)
            )

        imagemagick_utils.trim_whitespace(
            input_file_name=resized_panel_file_names[i],
            output_file_name=resized_panel_file_names[i],
            border_width_pixels=SMALL_BORDER_WIDTH_PX
        )
        imagemagick_utils.resize_image(
            input_file_name=resized_panel_file_names[i],
            output_file_name=resized_panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/evaluation_ungridded.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=resized_panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        temporal_eval_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORAL_DIR_ARG_NAME),
        non_temporal_eval_dir_name=getattr(
            INPUT_ARG_OBJECT, NON_TEMPORAL_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
