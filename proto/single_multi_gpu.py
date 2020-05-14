import sys
import cupy as cp
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with cp.cuda.Device(rank):
    print(f'Device ID: {rank}')