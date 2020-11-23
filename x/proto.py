from time import perf_counter
import numpy as np

import cupy as cp

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

    # GPU
    ############################
    num_of_gpu = cp.cuda.runtime.getDeviceCount()
    cp.cuda.Device(rank % num_of_gpu).use()
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    A = cp.ones((ns, n))
    # A = cp.ones((n, n))
    x = cp.ones(n)
    z = cp.ones(n)
    Ar = cp.ones((k + 2, n), T)
    Ay = cp.ones((k + 1, n), T)
    alpha = cp.ones(2*k + 3, T)
    beta = cp.ones(2*k + 2, T)
    delta = cp.ones(2*k + 1, T)

    ############################

    # CPU
    ############################
    # A = np.ones((ns, n))
    # # A = np.ones((n, n))
    # x = np.ones(n)
    # z = np.ones(n)
    # Ar = np.ones((k + 2, n), T)
    # Ay = np.ones((k + 1, n), T)
    # alpha = np.ones(2*k + 3, T)
    # beta = np.ones(2*k + 2, T)
    # delta = np.ones(2*k + 1, T)
    ############################

    Ar_cpu = np.ones((k + 2, n), T)

    s = perf_counter()
    callback(comm, k, begin, end, A, x, z, Ar, Ay, alpha, beta, delta, Ar_cpu)
    print('rank', rank, 'All:', perf_counter() - s)


# t(f)
t(g)
