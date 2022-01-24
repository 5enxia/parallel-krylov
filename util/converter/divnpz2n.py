import argparse
from scipy.sparse import load_npz, save_npz

parser = argparse.ArgumentParser(description='divide .npz to n files')
parser.add_argument("path", help="data file path")
parser.add_argument("n", help="number of output files")

args = parser.parse_args()
n = int(args.n)

npz = load_npz(args.path)
local_M = npz.shape[0] // n
print(f'Dived: {npz.shape} -> {n}*({local_M}, {npz.shape[1]})')
for i in range(n):
	print(f'Converting... {i+1} / {n}')
	local_npz = npz[i*local_M:(i+1)*local_M]
	path = args.path.replace('.npz', f'.{i:02d}.npz')
	save_npz(path, local_npz)
	print(f'Done: {path}')