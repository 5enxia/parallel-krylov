import sys
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm, multi_dot

if __name__ == "__main__":
    homedir = os.path.expanduser("~")
    directory = os.path.join(homedir, 'krylov')
    sys.path.append(directory)

from krylov.method.threads.common import start, end, init


def pcg(A, b, M, epsilon, callback=None, T=np.float64):
    x, b_norm, N, max_iter, residual, solution_updates = init(
        A, b, T, pu='cpu')

    start_time = start(method_name='Preconditioned CG')

    r = b - dot(A, x)
    residual[0] = norm(r) / b_norm
    z = dot(M, r)
    p = z.copy()

    for i in range(max_iter):
        alpha = dot(r, z) / multi_dot([p, A, p])
        x += alpha * p
        old_r = r.copy()
        old_z = z.copy()
        r -= alpha * dot(A, p)
        z = dot(M, r)

        residual[i+1] = norm(r) / b_norm
        if residual[i+1] < epsilon:
            isConverged = True
            break

        beta = dot(r, z) / dot(old_r, old_z)
        p = z + beta * p

        solution_updates[i] = i

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = num_of_iter

    # end(start_time, isConverged, num_of_iter, residual, residual_index)
    end(start_time, isConverged, num_of_iter, residual[residual_index])

    return isConverged


if __name__ == "__main__":
    import unittest
    from krylov.util import toepliz_matrix_generator, precondition, load_condition
    params = load_condition.load_condition_params('./condition.json')

    N = params['N']
    diag = params['diag']
    sub_diag = params['sub_diag']
    eps = params['epsilon']

    mn = params['method_name']
    mt = params['matrix_type']
    length = params['length']
    k = params['k']
    pt = params['processor_type']
    nop = params['number_of_process']

    toepliz_matrix_generator.generate(N, diag, sub_diag, T)
    M = precondition.ilu(A)

    pcg(A, b, M, epsilon, T)
