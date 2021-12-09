import argparse
from typing import Optional

import numpy as np
from scipy.linalg import toeplitz

T = np.float64
N = 1081
k = 1
elements = np.zeros(N, T)
elements[0] = 2
elements[1] = 1e-4

def normal(method):
	if method == 'cg':
		from refactor.cpu.cg import cg
		A, b = toeplitz(elements), np.ones(N, T)
		cg(A, b, 1e-10, T)

	if method == 'mrr':
		from refactor.cpu.mrr import mrr
		A, b = toeplitz(elements), np.ones(N, T)
		mrr(A, b, 1e-10, T)

	if method == 'kskipcg':
		from refactor.cpu.kskipcg import kskipcg
		A, b = toeplitz(elements), np.ones(N, T)
		kskipcg(A, b, 1e-10, k, T)

	if method == 'kskipmrr':
		from refactor.cpu.kskipmrr import kskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		kskipmrr(A, b, 1e-10, k, T)

	if method == 'adaptivekskipmrr':
		from refactor.cpu.adaptivekskipmrr import adaptivekskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		adaptivekskipmrr(A, b, 1e-10, k, T)

def mpi(method):
	if method == 'cg':
		from refactor.cpu.mpi.cg import cg
		A, b = toeplitz(elements), np.ones(N, T)
		cg(A, b, 1e-10, T)

	if method == 'mrr':
		from refactor.cpu.mpi.mrr import mrr
		A, b = toeplitz(elements), np.ones(N, T)
		mrr(A, b, 1e-10, T)

	if method == 'kskipcg':
		from refactor.cpu.mpi.kskipcg import kskipcg
		A, b = toeplitz(elements), np.ones(N, T)
		kskipcg(A, b, 1e-10, k, T)

	if method == 'kskipmrr':
		from refactor.cpu.mpi.kskipmrr import kskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		kskipmrr(A, b, 1e-10, k, T)

	if method == 'adaptivekskipmrr':
		from refactor.cpu.mpi.adaptivekskipmrr import adaptivekskipmrr
		A, b = toeplitz(elements), np.ones(N, T)
		adaptivekskipmrr(A, b, 1e-10, k, T)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='test exce cuter')
	parser.add_argument("-m", "--mpi", action="store_true", help="is mpi")
	parser.add_argument("method", help="")

	# parse argv
	args = parser.parse_args()
	if args.mpi:
		mpi(args.method)
	else:
		normal(args.method)