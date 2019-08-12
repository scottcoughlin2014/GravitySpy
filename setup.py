#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2018)
#
# This file is part of the gravityspy python package.
#
# gravityspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gravityspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gravityspy.  If not, see <http://www.gnu.org/licenses/>.

"""Setup the GravitySpy package
"""

from __future__ import print_function

import sys
if sys.version < '2.6':
    raise ImportError("Python versions older than 2.6 are not supported.")

import glob
import os.path

from setuptools import (setup, find_packages)

# set basic metadata
PACKAGENAME = 'gravityspy'
DISTNAME = 'gravityspy'
AUTHOR = 'Scott Coughlin'
AUTHOR_EMAIL = 'scott.coughlin@ligo.org'
LICENSE = 'GPLv3'

cmdclass = {}

# -- versioning ---------------------------------------------------------------

import versioneer
__version__ = versioneer.get_version()
cmdclass.update(versioneer.get_cmdclass())

# -- documentation ------------------------------------------------------------

# import sphinx commands
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    pass
else:
    cmdclass['build_sphinx'] = BuildDoc

# -- dependencies -------------------------------------------------------------

setup_requires = [
    'setuptools',
    'pytest-runner',
]

install_requires = [
    'gwpy >= 0.12',
    'numpy >= 1.7.1, <=1.15.4',
    'scipy >= 0.12.1',
    'matplotlib >= 1.2.0, != 2.1.0, != 2.1.1',
    'astropy >= 1.1.1, < 3.0.0 ; python_version < \'3\'',
    'astropy >= 1.1.1 ; python_version >= \'3\'',
    'configparser',
    'pandas >= 0.22 ; python_version >= \'3.5\'',
    'pandas < 0.21 ; python_version == \'3.4\'',
    'pandas >= 0.22 ; python_version == \'2.7\'',
    'tables > 3.0.0',
    'h5py >= 1.3',
    'panoptes-client >= 1.0.3',
    'psycopg2-binary >= 2.7.5',
    'sqlalchemy >= 1.2.12',
    'scikit_image >= 0.14.0',
    'keras == 2.1.6',
    'Theano == 1.0.1',
    'gwtrigfind >= 0.7',
    'lscsoft-glue >= 1.59.3 ',
    'lalsuite >= 6.49',
    'scikit-learn >= 0.20.0',
    'dqsegdb >= 1.5.0',
    'mysqlclient >= 1.4.0'
]

tests_require = [
    'pytest'
]

extras_require = {
    'doc': [
        'ipython',
        'sphinx',
        'numpydoc',
        'sphinx_rtd_theme',
        'sphinxcontrib_programoutput',
    ],
    "tf": ["tensorflow>=1.13.0"],
    "tf_gpu": ["tensorflow-gpu>=1.13.0"],
}

# enum34 required for python < 3.4
try:
    import enum  # pylint: disable=unused-import
except ImportError:
    install_requires.append('enum34')

# -- run setup ----------------------------------------------------------------

packagenames = find_packages()
data_extensions = ('.h5', '.pklz')
scripts = [fn for fn in glob.glob(os.path.join('bin', '*')) if
           not fn.endswith(data_extensions)]

setup(name=DISTNAME,
      provides=[PACKAGENAME],
      version=__version__,
      description=None,
      long_description=None,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=packagenames,
      include_package_data=True,
      cmdclass=cmdclass,
      scripts=scripts,
      url='https://github.com/Gravity-Spy/GravitySpy.git',
      setup_requires=setup_requires,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      test_suite='gravityspy.tests',
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
      use_2to3=True,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      ],
)
