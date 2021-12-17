import numpy as np
from numpy import dot
from numpy.linalg import norm
from .common import start, end as finish, init


def gropp(A, b, ilu, epsilon, T=np.float64, pt='cpu'):
    isConverged = False

    if pt == 'cpu':
        import numpy as xp
        from numpy import dot
        from numpy.linalg import norm
        x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T, pt)
    else:
        import cupy as xp
        from cupy import dot
        from cupy.linalg import norm
        A, b, x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T, pt)

    start_time = start(method_name='gropp')

    r = b - dot(A, x)
    residual[0] = norm(r) / b_norm
    u = ilu.solve(r)
    p = u.copy()
    s = dot(A, p)
    gamma = dot(r, u)

    i = 0
    for i in range(1, max_iter):
        delta = dot(p, s)
        q = ilu.solve(s)
        alpha = gamma/delta
        x += alpha*p
        r -= alpha*s
        residual[i] = norm(r)/b_norm
        if residual[i] < epsilon:
            isConverged = True
            break
        u -= alpha*q
        gamma = dot(r, u)
        old_gamma = gamma
        w = dot(A, u)
        beta = gamma/old_gamma
        p = u + beta*p
        s = w + beta*s

    elapsed_time = finish(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
