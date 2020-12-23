from setuptools import setup
from Cython.Build import cythonize
import numpy

name = ''
target = 'protof'
include_path = [numpy.get_include()]
setup(
    name=f'{name}',
    ext_modules=cythonize(
        f'{target}.pyx',
        include_path=include_path
    ),
    include_dirs=include_path,
    zip_safe=False,
)
