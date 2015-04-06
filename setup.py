import sys
sys.path.insert(0, '.')
import os
import pkg_resources

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import version


pyx_files = ['camip/CAMIP.pyx', 'camip/device/CAMIP.pyx']
ext_modules = [Extension(f[:-4].replace('/', '.'), [f],
                         extra_compile_args=['-O3', '-msse3', '-std=c++0x'],
                         include_dirs=['camip',
                                       os.path.expanduser('~/local/include'),
                                       '/usr/local/cuda-6.5/include',
                                       pkg_resources
                                       .resource_filename('cythrust', ''),
                                       np.get_include()],
                         define_macros=[('THRUST_DEVICE_SYSTEM',
                                         'THRUST_DEVICE_SYSTEM_CPP')])
               for f in pyx_files]


setup(name='camip',
      version=version.getVersion(),
      description='Concurrent Associated-Moves Iterative Placement',
      keywords='fpga iterative placement',
      author='Christian Fobel',
      author_email='christian@fobel.net',
      url='http://github.com/cfobel/camip.git',
      license='GPL',
      packages=['camip'],
      install_requires=['pandas>=0.16.0', 'numpy>=1.9.2', 'path-helpers>=0.2',
                        'scipy>=0.15.1', 'vpr-netfile-parser>=0.3',
                        'numexpr>=2.0.0', 'tables>=3.1.1', 'si-prefix>=0.1',
                        'cythrust>=0.9.3', 'fpga-netlist>=0.3'],
      ext_modules=cythonize(ext_modules))
