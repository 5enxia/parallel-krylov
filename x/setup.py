from setuptools import setup
from Cython.Build import cythonize

name = ''
target = 'mynumpy'
setup(
    name=f'{name}',
    ext_modules=cythonize(f'{target}.pyx'),
    zip_safe=False,
)
