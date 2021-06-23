from __future__ import print_function

from mpi4py import MPI
from numba import cuda
import numpy as np

mpi_comm = MPI.COMM_WORLD

# Input data size
total_n = 10


# Process 0 creates input data
if mpi_comm.rank == 0:
    input_data = np.arange(total_n, dtype=np.int32)
    print("Input:", input_data)
else:
    input_data = None


# Compute partitioning of the input array
proc_n = [total_n // mpi_comm.size + (total_n % mpi_comm.size > n) for n in range(mpi_comm.size)]
pos = 0
pos_n = []
for n in range(mpi_comm.size):
    pos_n.append(pos)
    pos += proc_n[n]

my_n = proc_n[mpi_comm.rank]
my_offset = pos_n[mpi_comm.rank]
print('Process %d, my_n = %d' % (mpi_comm.rank, my_n))
print('Process %d, my_offset = %d' % (mpi_comm.rank, my_offset))


# Distribute input data across processes
my_input_data = np.zeros(my_n, dtype=np.int32)
mpi_comm.Scatterv([input_data, proc_n, pos_n, MPI.INT], my_input_data)
print('Process %d, my_input_data = %s' % (mpi_comm.rank, my_input_data))


# Perform computation on local data

@cuda.jit
def sqplus2(input_data, output_data):
    for i in range(len(input_data)):
        d = input_data[i]
        output_data[i] = d * d + 2


my_output_data = np.empty_like(my_input_data)
sqplus2(my_input_data, my_output_data)
print('Process %d, my_output_data = %s' % (mpi_comm.rank, my_output_data))


# Bring result back to root process
if mpi_comm.rank == 0:
    output_data = np.empty_like(input_data)
else:
    output_data = None

mpi_comm.Gatherv(my_output_data, [output_data, proc_n, pos_n, MPI.INT])

if mpi_comm.rank == 0:
    print("Output:", output_data)


MPI.Finalize()
