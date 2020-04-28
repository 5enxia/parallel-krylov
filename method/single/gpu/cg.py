import sys

import cupy as cp
from cupy import dot
from cupy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.single.common import start, end
from krylov.method.single.gpu.common import init

def cg(A, b, epsilon, callback = None, T = cp.float64):
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)

    start_time = start(method_name = sys._getframe().f_code.co_name)

    r = b - dot(A,x)
    residual[0] = norm(r) / b_norm
    p = r.copy()

    for i in range(0, max_iter):
        alpha = dot(r,p) / dot(dot(p,A),p)
        x += alpha * p
        old_r = r.copy()
        r -= alpha * dot(A,p)

        residual[i+1] = norm(r) / b_norm
        solution_updates[i] = i + 1
        if residual[i+1] < epsilon:
            isConverged = True
            break

        beta = dot(r,r) / dot(old_r, old_r)
        p = r + beta * p

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = num_of_iter
    
    end(start_time, isConverged, num_of_iter, residual, residual_index)
    
    return isConverged


if __name__ == "__main__":
    import unittest
    from krylov.util import loader, toepliz_matrix_generator

    class TestCgMethod(unittest.TestCase):
        T = cp.float64
        epsilon = 1e-8

        def test_single_cg_method(self):
            A ,b = toepliz_matrix_generator.generate(N=10000,diag=2.5)
            A, b= cp.asarray(A), cp.asarray(b)
            self.assertTrue(cg(A, b, TestCgMethod.epsilon, TestCgMethod.T))

    unittest.main()
