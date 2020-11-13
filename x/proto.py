from time import perf_counter
import numpy as np
import cupy as cp

from protof import f
from protog import g


def t(callback):
    T = np.float64
    n = 10000
    k = 5

    A = cp.ones((n, n))
    x = cp.ones(n)
    z = cp.ones(n)
    Ar = cp.ones(n)
    Ay = cp.ones(n)
    alpha = cp.ones(2*k + 3, T)
    beta = cp.ones(2*k + 2, T)
    delta = cp.ones(2*k + 1, T)

    Ar_cpu = np.ones(n)

    s = perf_counter()
    callback(k, A, x, z, Ar, Ay, alpha, beta, delta, Ar_cpu)
    print(perf_counter() - s)


t(f)
t(g)
