import numpy as np
from numpy import dot
from numpy.linalg import norm
from .common import start, end, init


def pipeline(A, b, ilu, epsilon, T=np.float64, pt='cpu'):
    isConverged = False
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T, pt)

    start_time = start(method_name='pipeline')

    r = b - dot(A, x)
    residual[0] = norm(r) / b_norm

    u = ilu.solve(r)
    w = dot(A, u)

    z = np.zeros(N)
    q = np.zeros(N)
    s = np.zeros(N)
    p = np.zeros(N)

    for i in range(1, max_iter):
        gamma = dot(r, u)
        old_gamma = gamma
        delta = dot(w, u)

        m = ilu.solve(r)

        n = dot(A, m)
        if i > 1:
            beta = gamma/old_gamma
            alpha = gamma/(delta - beta*gamma/alpha)
        else:
            beta = 0
            alpha = gamma/delta
        z = n + beta*z
        q = m + beta*q
        s = w + beta*s
        p = u + beta*p
        x += alpha*p
        r -= alpha*s
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break
        u -= alpha*q
        w -= alpha*z

    end(start_time, isConverged, i, residual[i])
