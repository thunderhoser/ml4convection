"""USE ONCE AND DESTROY."""

import glob
import shutil
import os.path

RADAR_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4convection_project/'
    'radar_data'
)

reflectivity_file_names = glob.glob(
    '{0:s}/201*/radar*.nc.gz'.format(RADAR_DIR_NAME)
)

for this_orig_file_name in reflectivity_file_names:
    this_directory_name, this_pathless_file_name = os.path.split(
        this_orig_file_name
    )
    this_new_file_name = '{0:s}/{1:s}'.format(
        this_directory_name,
        this_pathless_file_name.replace('radar_', 'reflectivity_')
    )

    print('Moving "{0:s}" to "{1:s}"...'.format(
        this_orig_file_name, this_new_file_name
    ))
    shutil.move(this_orig_file_name, this_new_file_name)
