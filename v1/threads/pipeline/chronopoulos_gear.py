import numpy as np
from numpy import dot
from numpy.linalg import norm
from .common import start, end as finish, init


def chronopoulos_gear(A, b, ilu, epsilon, T=np.float64, pt: str='cpu'):
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

    start_time = start(method_name='chronopoulos gear')

    r = b - dot(A, x)
    residual[0] = norm(r)/b_norm
    num_of_solution_updates[1] = 1
    u = ilu.solve(r)
    w = dot(A, u)

    alpha = dot(r, u)/dot(w, u)
    beta = 0
    gamma = dot(r, u)
    old_gamma = gamma

    p = np.zeros(N, T)
    s = np.zeros(N, T)

    for i in range(1, max_iter):
        p = u + beta*p
        s = w + beta*s
        x += alpha*p
        r -= alpha*s
        residual[i] = norm(r)/b_norm
        if residual[i] < epsilon:
            isConverged = True
            break
        u = ilu.solve(r)
        w = dot(A, u)
        gamma = dot(r, u)
        delta = dot(w, u)
        beta = gamma/old_gamma
        alpha = gamma/(delta - beta*gamma/alpha)

        num_of_solution_updates[i] = i

    elapsed_time = finish(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
