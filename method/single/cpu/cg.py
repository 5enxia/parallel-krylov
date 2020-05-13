import sys

import numpy as np
from numpy import dot
from numpy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.single.common import start, end 
from krylov.method.single.cpu.common import init 

def cg(A, b, epsilon, callback = None, T = np.float64):
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
        def test_single_cg_method(self):
            import json

            with open('../../../../krylov/data/condition.json') as f:
                params = json.load(f)
            f.close()

            T = np.float64
            epsilon = params['epsilon']
            N = params['N'] 
            diag = params['diag']

            A ,b = toepliz_matrix_generator.generate(N=N, diag=diag, T=T)
            print(f'N:\t{N}')
            self.assertTrue(cg(A, b, epsilon, T))

    unittest.main()
