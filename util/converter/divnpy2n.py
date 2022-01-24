import argparse
from numpy import load, save

parser = argparse.ArgumentParser(description='divide .npy to n files')
parser.add_argument("path", help="data file path")
parser.add_argument("n", help="number of output files")

args = parser.parse_args()
n = int(args.n)

npy = load(args.path)
local_M = npy.shape[0] // n
print(f'Dived: {npy.shape} -> {n}*({local_M}, {npy.shape[1]})')
for i in range(n):
	print(f'Converting... {i+1} / {n}')
	local_npy = npy[i*local_M:(i+1)*local_M]
	path = args.path.replace('.npy', f'.{i:02d}.npy')
	save(path, local_npy)
	print(f'Done: {path}')