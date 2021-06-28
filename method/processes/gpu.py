import numpy as np
import cupy as cp
from cupy import dot


def init_matvec(N, local_N, T):
    local_A = cp.empty((local_N, N), T)
    Ax = np.empty(N, T)
    local_Ax = cp.empty(local_N, T)
    return local_A, Ax, local_Ax


def init_vecvec(local_N, T):
    local_a = cp.empty(local_N, T)
    local_b = cp.empty(local_N, T)
    return local_a, local_b


def mpi_matvec(local_A, x, Ax, local_Ax, comm):
    Ax = x.get()
    comm.Bcast(Ax)
    x = cp.asarray(Ax)
    local_Ax = dot(local_A, x)
    comm.Gather(local_Ax.get(), Ax)
    return cp.asarray(Ax)


def mpi_vecvec1(a, local_a, comm):
    ab = np.empty(1, cp.float64)
    local_a_cpu = local_a.get()
    comm.Scatter(a.get(), local_a_cpu)
    local_a = cp.asarray(local_a_cpu)
    local_ab = dot(local_a, local_a)
    comm.Reduce(local_ab.get(), ab)
    return cp.asarray(ab)


def mpi_vecvec2(a, b, local_a, local_b, comm):
    ab = np.empty(1, cp.float64)
    local_a_cpu = local_a.get()
    local_b_cpu = local_b.get()
    comm.Scatter(a.get(), local_a_cpu)
    comm.Scatter(b.get(), local_b_cpu)
    local_a = cp.asarray(local_a_cpu)
    local_b = cp.asarray(local_b_cpu)
    local_ab = dot(local_a, local_b)
    comm.Reduce(local_ab.get(), ab)
    return cp.asarray(ab)
