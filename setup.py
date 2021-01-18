"""Setup file for ml4convection."""

from setuptools import setup

PACKAGE_NAMES = [
    'ml4convection', 'ml4convection.io', 'ml4convection.machine_learning',
    'ml4convection.plotting', 'ml4convection.utils', 'ml4convection.scripts',
    'ml4convection.figures'
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data science', 'weather', 'meteorology', 'thunderstorm', 'convection',
    'satellite', 'radar'
]
SHORT_DESCRIPTION = (
    'Uses machine learning to predict convective initiation and decay from '
    'satellite data.'
)
LONG_DESCRIPTION = SHORT_DESCRIPTION
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

# You also need to install the following packages, which are not available in
# pip.  They can both be installed by "git clone" and "python setup.py install",
# the normal way one installs a GitHub package.
#
# https://github.com/matplotlib/basemap
# https://github.com/tkrajina/srtm.py

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'geopy',
    'metpy'
]

if __name__ == '__main__':
    setup(
        name='ml4convection',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ryan.lagerquist@noaa.gov',
        url='https://github.com/thunderhoser/ml4convection',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
