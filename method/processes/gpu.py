import numpy as np
import cupy as cp
from cupy import dot


def init_matvec(N, local_N, T=cp.float64):
    local_A = cp.empty((local_N, N), cp.float64)
    Ax = np.empty(N, cp.float64)
    local_Ax = cp.empty(local_N, cp.float64)
    return local_A, Ax, local_Ax


def init_vecvec(local_N, T=cp.float64):
    local_a = cp.empty(local_N, cp.float64)
    local_b = cp.empty(local_N, cp.float64)
    return local_a, local_b


def mpi_matvec(local_A, x, Ax, local_Ax, comm):
    x_cpu = x.get()
    comm.Bcast(x_cpu, root=0)
    x = cp.asarray(x_cpu)

    local_Ax = dot(local_A, x)

    comm.Gather(local_Ax.get(), Ax, root=0)
    return cp.asarray(Ax)


def mpi_vecvec2(a, b, local_a, local_b, comm):
    ab = np.empty(1, cp.float64)
    local_a_cpu = local_a.get()
    local_b_cpu = local_b.get()
    comm.Scatter(a.get(), local_a_cpu, root=0)
    comm.Scatter(b.get(), local_b_cpu, root=0)
    local_a = cp.asarray(local_a_cpu)
    local_b = cp.asarray(local_b_cpu)

    local_ab = dot(local_a, local_b)

    comm.Reduce(local_ab.get(), ab, root=0)
    return cp.asarray(ab)
