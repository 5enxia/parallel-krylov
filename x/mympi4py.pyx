from mpi4py import MPI
from mpi4py.MPI cimport MPI as CMPI

cdef CMPI.Comm comm = MPI.COMM_WORLD
print(comm.Get_size)
