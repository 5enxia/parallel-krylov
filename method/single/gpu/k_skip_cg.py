import sys

import cupy as cp
from cupy import dot
from cupy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.single.common import start, end
from krylov.method.single.gpu.common import init 

def k_skip_cg(A, b, k, epsilon, callback = None, T = cp.float64):
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)

    start_time = start(method_name = 'k-skip CG', k = k)
    
    Ar = cp.zeros((k + 2, N), T)
    Ar[0] = b - dot(A,x)
    Ap = cp.zeros((k + 3, N), T)
    Ap[0] = Ar[0]

    a = cp.zeros(2 * k + 2, T)
    f = cp.zeros(2 * k + 4, T)
    c = cp.zeros(2 * k + 2, T)

    for i in range(0, max_iter):
        
        residual[i] = norm(Ar[0]) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        for j in range(1, k + 1):
            Ar[j] = dot(A,Ar[j-1])

        for j in range(1, k + 2):
            Ap[j] = dot(A,Ap[j-1])

        for j in range(0, 2 * k + 1, 2):
            jj = j // 2
            a[j] = dot(Ar[jj],Ar[jj])
            a[j+1] = dot(Ar[jj],Ar[jj+1])

        for j in range(0, 2 * k + 3, 2):
            jj = j // 2
            f[j] = dot(Ap[jj],Ap[jj])
            f[j+1] = dot(Ap[jj],Ap[jj+1])

        for j in range(0, 2 * k + 1, 2):
            jj = j // 2
            c[j] = dot(Ar[jj],Ap[jj])
            c[j+1] = dot(Ar[jj],Ap[jj+1])

        alpha = a[0] / f[1]
        beta = alpha ** 2 * f[2] / a[0] - 1
        x += alpha * Ap[0]
        Ar[0] -= alpha * Ap[1]
        Ap[0] = Ar[0] + beta * Ap[0]
        Ap[1] = dot(A,Ap[0])
    
        for j in range(0, k):
            for l in range(0, 2*(k-j)+1):
                a[l] += alpha*(alpha*f[l+2] - 2*c[l+1])
                d = c[l] - alpha*f[l+1]
                c[l] = a[l] + d*beta
                f[l] = c[l] + beta*(d + beta*f[l])
    
            alpha = a[0] / f[1]
            beta = alpha ** 2 * f[2] / a[0] - 1
            x += alpha * Ap[0]
            Ar[0] -= alpha * Ap[1]
            Ap[0] = Ar[0] + beta * Ap[0]
            Ap[1] = dot(A,Ap[0])
        
        solution_updates[i+1] = solution_updates[i] + k + 1

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = i 

    end(start_time, isConverged, num_of_iter, residual, residual_index)
    
    return isConverged


if __name__ == "__main__":
    import unittest
    from krylov.util import toepliz_matrix_generator

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    class TestMethod(unittest.TestCase):
        T = cp.float64
        epsilon = 1e-8
        N = 40000

        def test_single_k_skip_cg_method(self):
            N = TestMethod.N
            k = 10
            A ,b = toepliz_matrix_generator.generate(N=N,diag=2.005)
            print(f'N:\t{N}')
            A, b= cp.asarray(A), cp.asarray(b)

            self.assertTrue(k_skip_cg(A, b, k, TestMethod.epsilon, TestMethod.T))

    unittest.main()