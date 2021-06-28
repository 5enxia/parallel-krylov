import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm, multi_dot

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.single.common import start, end 
from krylov.method.single.cpu.common import init 

def pcg(A, b, M, epsilon, callback = None, T = np.float64):
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
    from krylov.util import loader,toepliz_matrix_generator,precondition

    class TestCgMethod(unittest.TestCase):
        epsilon = 1e-4
        T = np.float64

        def test_single_pcg_method(self):
            A, b = toepliz_matrix_generator.generate(N = 2, diag=2.5)
            M = precondition.ilu(A)
            self.assertTrue(pcg(A, b, M ,TestCgMethod.epsilon, TestCgMethod.T))

    unittest.main()