import numpy as np
from numpy import dot
from numpy.linalg import norm

from .common import start, end, init


def pcg(A, b, ilu, epsilon: float, T=np.float64, pt: str='cpu'):
    isConverged = False
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T, pt)

    start_time = start(method_name='Preconditioned CG')

    r = b - dot(A, x)
    residual[0] = norm(r)/b_norm

    u = ilu.solve(r)
    p = u.copy()

    for i in range(1, max_iter):
        s = dot(A, p)

        alpha = dot(r, u)/dot(s, p)
        x += alpha*p
        r -= alpha*s
        residual[i] = norm(r)/b_norm
        if residual[i] < epsilon:
            isConverged = True
            break
        old_r = r.copy()
        old_u = u.copy()
        u = ilu.solve(r)
        beta = dot(r, u)/dot(old_r, old_u)
        p = u + beta*p

    end(start_time, isConverged, i, residual[i])
