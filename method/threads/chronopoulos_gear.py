import numpy as np
from numpy import dot
from numpy.linalg import norm
from .common import start, end, init


def chronopoulos_gear(A, b, ilu, epsilon, callback=None, T=np.float64):
    isConverged = False
    x, b_norm, N, max_iter, residual, solution_updates = init(
        A, b, T, pu='cpu')

    start_time = start(method_name='chronopoulos gear')

    r = b - dot(A, x)
    residual[0] = norm(r)/b_norm
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
        print(residual[i])
        if residual[i] < epsilon:
            isConverged = True
            break
        u = ilu.solve(r)
        w = dot(A, u)
        gamma = dot(r, u)
        delta = dot(w, u)
        beta = gamma/old_gamma
        alpha = gamma/(delta - beta*gamma/alpha)

    end(start_time, isConverged, i, residual[i])
