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
rank = comm.Get_rank() # 0~3
num_of_process = comm.Get_size() # 4

# cuda init
num_of_gpu = getDeviceCount() # 4
for i in range(num_of_gpu):
	Device(i).use()
	pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
	cp.cuda.set_allocator(pool.malloc)

# padding data
A, b, local_N = padding(A, b, num_of_process)
local_local_N = local_N // num_of_gpu
begins = [(num_of_gpu*rank + i)*local_local_N for i in range(num_of_gpu)]
ends = [(num_of_gpu*rank + i + 1)*local_local_N for i in range(num_of_gpu)]
local_begins =[i*local_local_N for i in range(num_of_gpu)] 
local_ends =[(i + 1)*local_local_N for i in range(num_of_gpu)] 

###############################################################################
res = np.zeros(b.size)
local_res = np.zeros(b.size//num_of_process)

for i in range(num_of_gpu):
	Device(i).use()
	local_A = cp.asarray(A[begins[i]:ends[i]])
	b = cp.asarray(b)
if rank == 0:
	print("hybrid mpi")
	start = MPI.Wtime()
###############################################################################
for j in range(1000):
	for i in range(num_of_gpu):
		Device(i).use()
		local_res[local_begins[i]:local_ends[i]] = local_A.dot(b).get()
	for i in range(num_of_gpu):
		Device(i).synchronize()
	comm.Allgather(local_res, res)
if rank == 0:
	print(MPI.Wtime() - start)
	print(res)
###############################################################################
