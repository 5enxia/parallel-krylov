import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.single.common import start, end 
from krylov.method.single.cpu.common import init 

def k_skip_mrr(A, b, k, epsilon, callback = None, T = np.float64):
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)
    start_time = start(method_name = 'k-skip MrR', k = k)
    
    Ar = np.empty((k + 3, N), T)
    Ar[0] = b - dot(A, x)
    residual[0] = norm(Ar[0]) / b_norm
    Ay = np.empty((k + 2, N), T)

    # ======= first iter ====== #
    Ar[1] = dot(A,Ar[0])
    zeta = dot(Ar[0],Ar[1]) / dot(Ar[1],Ar[1])
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    # ========================= #

    alpha = np.empty(2 * k + 3, T)
    beta = np.empty(2 * k + 2, T)
    delta = np.empty(2 * k + 1, T)
    beta[0] = 0

    solution_updates[1] = 1

    for i in range(1, max_iter):

        residual[i] = norm(Ar[0]) / b_norm
        if residual[i] < epsilon:
            # self._converged(i,i)
            isConverged = True
            break

        for j in range(1, k + 2):
            Ar[j] = dot(A,Ar[j-1])

        for j in range(1, k + 1):
            Ay[j] = dot(A,Ay[j-1])

        for j in range(2 * k + 3):
            jj = j // 2
            alpha[j] = dot(Ar[jj],Ar[jj+j%2])

        for j in range(1, 2 * k + 2):
            jj = j//2
            beta[j] = dot(Ay[jj],Ar[jj+j%2])

        for j in range(2 * k + 1):
            jj = j // 2
            delta[j] = dot(Ay[jj],Ay[jj+j%2])
            
        sigma = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / sigma
        eta = -alpha[1] * beta[1] / sigma

        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta*z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar[1] = dot(A,Ar[0])
        x -= z

        for j in range(k):

            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]

            for l in range(2, 2 * (k - j) + 1):

                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l+1] + zeta ** 2 * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]

            sigma = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / sigma
            eta = -alpha[1] * beta[1] / sigma

            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            Ar[1] = dot(A,Ar[0])
            x -= z

        solution_updates[i + 1] = solution_updates[i] + k + 1

    else:
        isConverged = False
        
    num_of_iter = i + 1
    residual_index = i

    end(start_time, isConverged, num_of_iter, residual, residual_index)

    return isConverged



if __name__ == "__main__":
    import unittest
    from krylov.util import loader, toepliz_matrix_generator

    class TestCgMethod(unittest.TestCase):
        def test_single_k_skip_MrR_method(self):
            import json

            with open('condition.json') as f:
                params = json.load(f)
            f.close()

            T = np.float64
            epsilon = params['epsilon']
            N = params['N'] 
            diag = params['diag']
            k = params['k']

            A, b = toepliz_matrix_generator.generate(N=N,diag=diag,T=T)
            self.assertTrue(k_skip_mrr(A, b, k, epsilon, T))

    unittest.main()