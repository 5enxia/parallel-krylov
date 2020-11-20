from setuptools import setup
from Cython.Build import cythonize
import numpy

name = ''
target = 'protof'
setup(
    name=f'{name}',
    ext_modules=cythonize(f'{target}.pyx'),
    # include_path=[numpy.get_include()],
    zip_safe=False,
)
