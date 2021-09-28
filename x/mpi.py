import numpy as np
import cupy as cp
from cupy.cuda import Device
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
from common import load_data, padding

# load data
A, b = load_data()

# mpi init
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # 0~15 
num_of_process = comm.Get_size() # 16

# cuda init
num_of_gpu = getDeviceCount() # 4
Device(rank % num_of_gpu).use()
# print(cp.cuda.get_device_id())
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

# padding data
A, b, local_N = padding(A, b, num_of_process)
begin = rank*local_N
end = (rank+1)*local_N

###############################################################################
res = np.zeros(b.size)
local_res = np.zeros(b.size // num_of_process)
local_A = cp.asarray(A[begin:end])
b = cp.asarray(b)
if rank == 0:
	print("only mpi")
	start = MPI.Wtime()
###############################################################################
for j in range(10000):
	comm.Allgather(local_A.dot(b).get(), res)
###############################################################################
if rank == 0:
	print(MPI.Wtime() - start)
	print(res)
###############################################################################
