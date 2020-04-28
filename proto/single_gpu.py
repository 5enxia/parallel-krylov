import sys

import cupy as cp
import numpy as np

sys.path.append('../../')
from krylov.util import toepliz_matrix_generator

A, b = toepliz_matrix_generator.generate(N=10,diag=2.5)
# A, b = cp.asarray(A), cp.asarray(b)
print(cp.dot(A,b))
