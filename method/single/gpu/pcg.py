import sys

import cupy as cp
from cupy import dot
from cupy.linalg import norm, multi_dot

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.single.common import start, end 
from krylov.method.single.gpu.common import init

def pcg(A, b, M, epsilon, callback = None, T = cp.float64):
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)
    
    start_time = start(method_name='Preconditioned CG')

    r = b - dot(A,x)
    residual[0] = norm(r) / b_norm
    z = dot(M,r)
    p = z.copy() 

    for i in range(max_iter):
        alpha = dot(r,z) / multi_dot([p,A,p]) 
        x += alpha * p
        old_r = r.copy()
        old_z = z.copy()
        r -= alpha * dot(A,p)
        z = dot(M,r)

        residual[i+1] = norm(r) / b_norm
        if residual[i+1] < epsilon:
            isConverged = True
            break

        beta = dot(r,z) / dot(old_r,old_z)
        p = z + beta * p
        
        solution_updates[i] = i
    
    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = num_of_iter

    end(start_time, isConverged, num_of_iter, residual, residual_index)
    
    return isConverged


if __name__ == "__main__":
    import unittest
    from krylov.util import toepliz_matrix_generator,precondition

    class TestMethod(unittest.TestCase):
        epsilon = 1e-8
        T = cp.float64
        N = 40000

        def test_single_pcg_method(self):
            N = TestMethod.N
            A, b = toepliz_matrix_generator.generate(N = N, diag=2.005)
            M = precondition.ilu(A)
            print(f'N:\t{N}')
            A, b = cp.asarray(A), cp.asarray(b)

            self.assertTrue(pcg(A, b, M ,TestMethod.epsilon, TestMethod.T))

    unittest.main()