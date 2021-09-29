import numpy as np
from .common import start, end, init

def pcg(A, b, ilu, epsilon: float, T=np.float64, pt: str='cpu'):
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

    start_time = start(method_name='Preconditioned CG')

    r = b - dot(A, x)
    residual[0] = norm(r)/b_norm
    num_of_solution_updates[1] = 1

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

        num_of_solution_updates[i] = i

    elapsed_time = end(start_time, isConverged, i, residual[i])
    return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
