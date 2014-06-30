from distutils.core import setup

import version

setup(name='camip',
      version=version.getVersion(),
      description='Concurrent Associated-Moves Iterative Placement',
      keywords='fpga iterative placement',
      author='Christian Fobel',
      author_email='christian@fobel.net',
      #url='http://github.com/wheeler-microfluidics/microdrop_utility.git',
      license='GPL',
      packages=['camip'],
      install_requires=['pandas', 'numpy', 'scipy'])
