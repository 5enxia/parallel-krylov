import numpy as np
from numpy import dot


def init_matvec(N, local_N, T):
    local_A = np.empty((local_N, N), T)
    Ax = np.empty(N, T)
    local_Ax = np.empty(local_N, T)
    return local_A, Ax, local_Ax


def init_vecvec(local_N, T):
    local_a = np.empty(local_N, T)
    local_b = np.empty(local_N, T)
    return local_a, local_b


def mpi_matvec(local_A, x, Ax, local_Ax, comm):
    comm.Bcast(x)
    local_Ax = dot(local_A, x)
    comm.Gather(local_Ax, Ax)
    return Ax


def mpi_vecvec1(a, local_a, comm):
    ab = np.empty(1, np.float64)
    comm.Scatter(a, local_a)
    local_ab = dot(local_a, local_a)
    comm.Reduce(local_ab, ab)
    return ab


def mpi_vecvec2(a, b, local_a, local_b, comm):
    ab = np.empty(1, np.float64)
    comm.Scatter(a, local_a)
    comm.Scatter(b, local_b)
    local_ab = dot(local_a, local_b)
    comm.Reduce(local_ab, ab)
    return ab
