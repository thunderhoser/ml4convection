"""Makes satellite-QC figure."""

import shutil
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.io import satellite_io
from ml4convection.io import border_io
from ml4convection.figures import make_data_overview_figure as make_overview_fig

TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = '%Y%m%d'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

RAW_DIRECTORY_ARG_NAME = 'input_raw_directory_name'
QC_DIRECTORY_ARG_NAME = 'input_qc_directory_name'
VALID_TIME_ARG_NAME = 'valid_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

RAW_DIRECTORY_HELP_STRING = (
    'Name of directory with raw satellite data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
QC_DIRECTORY_HELP_STRING = (
    'Same as `{0:s}` but for quality-controlled data.'
).format(RAW_DIRECTORY_ARG_NAME)

VALID_TIME_HELP_STRING = (
    'Will plot data for this valid time (format "yyyy-mm-dd-HHMM").'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RAW_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + QC_DIRECTORY_ARG_NAME, type=str, required=True,
    help=QC_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_raw_directory_name, top_qc_directory_name, valid_time_string,
         output_dir_name):
    """Makes satellite-QC figure.

    This is effectively the main method.

    :param top_raw_directory_name: See documentation at top of file.
    :param top_qc_directory_name: Same.
    :param valid_time_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    valid_date_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, DATE_FORMAT
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    raw_satellite_file_name = satellite_io.find_file(
        top_directory_name=top_raw_directory_name,
        valid_date_string=valid_date_string, prefer_zipped=False,
        allow_other_format=True
    )
    qc_satellite_file_name = satellite_io.find_file(
        top_directory_name=top_qc_directory_name,
        valid_date_string=valid_date_string, prefer_zipped=False,
        allow_other_format=True
    )

    print('Reading data from: "{0:s}"...'.format(raw_satellite_file_name))
    raw_satellite_dict = satellite_io.read_file(
        netcdf_file_name=raw_satellite_file_name, fill_nans=False
    )
    raw_satellite_dict = satellite_io.subset_by_time(
        satellite_dict=raw_satellite_dict,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )[0]
    raw_satellite_dict = satellite_io.subset_by_band(
        satellite_dict=raw_satellite_dict,
        band_numbers=numpy.array([8], dtype=int)
    )

    print('Reading data from: "{0:s}"...'.format(qc_satellite_file_name))
    qc_satellite_dict = satellite_io.read_file(
        netcdf_file_name=qc_satellite_file_name, fill_nans=False
    )
    qc_satellite_dict = satellite_io.subset_by_time(
        satellite_dict=qc_satellite_dict,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )[0]
    qc_satellite_dict = satellite_io.subset_by_band(
        satellite_dict=qc_satellite_dict,
        band_numbers=numpy.array([8], dtype=int)
    )

    this_file_name = make_overview_fig._plot_one_satellite_image(
        satellite_dict=raw_satellite_dict, time_index=0, band_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        letter_label='a', output_dir_name=output_dir_name,
        cbar_orientation_string=None
    )

    raw_figure_file_name = '{0:s}_raw.jpg'.format(
        os.path.splitext(this_file_name)[0]
    )
    shutil.move(this_file_name, raw_figure_file_name)

    this_file_name = make_overview_fig._plot_one_satellite_image(
        satellite_dict=qc_satellite_dict, time_index=0, band_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        letter_label='b', output_dir_name=output_dir_name,
        cbar_orientation_string='vertical'
    )

    qc_figure_file_name = '{0:s}_qc.jpg'.format(
        os.path.splitext(this_file_name)[0]
    )
    shutil.move(this_file_name, qc_figure_file_name)

    concat_figure_file_name = '{0:s}/data_overview_{1:s}.jpg'.format(
        output_dir_name, valid_time_string
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[raw_figure_file_name, qc_figure_file_name],
        output_file_name=concat_figure_file_name,
        num_panel_rows=1, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_raw_directory_name=getattr(
            INPUT_ARG_OBJECT, RAW_DIRECTORY_ARG_NAME
        ),
        top_qc_directory_name=getattr(INPUT_ARG_OBJECT, QC_DIRECTORY_ARG_NAME),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
