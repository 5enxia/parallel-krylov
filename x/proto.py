from time import perf_counter
import numpy as np
from mpi4py import MPI

from protof import f
from protog import g


def t(callback):
    n = 16000
    T = np.float64
    k = 7

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    ns = (n//size)
    begin = ns * rank
    end = ns * (rank+1)

    A = np.ones((n, n))
    x = np.ones(n)
    z = np.ones(n)
    Ar = np.ones((k + 2, n), T)
    Ay = np.ones((k + 1, n), T)
    alpha = np.ones(2*k + 3, T)
    beta = np.ones(2*k + 2, T)
    delta = np.ones(2*k + 1, T)

    Ar_cpu = np.ones(n)

    s = perf_counter()
    callback(comm, k, begin, end, A, x, z, Ar, Ay, alpha, beta, delta, Ar_cpu)
    print(perf_counter() - s)


t(f)
t(g)
