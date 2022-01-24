import argparse
from scipy.io import mmwrite
from scipy.sparse import bsr_matrix, csc_matrix, csr_matrix, coo_matrix, dia_matrix, dok_matrix, lil_matrix
from scipy.sparse import load_npz, save_npz
from scipy.sparse import vstack, hstack

methods = {
    "bsr": bsr_matrix,
    "csc": csc_matrix,
    "csr": csr_matrix,
    "coo": coo_matrix,
    "dia": dia_matrix,
    "dok": dok_matrix,
    "lil": lil_matrix,
}

parser = argparse.ArgumentParser(description='divide .npz to n files')
parser.add_argument("path", help="data file path")
parser.add_argument("n", help="number of processes")
parser.add_argument("format", help="Matrix Storage Format")

args = parser.parse_args()
np = int(args.n)
npz = load_npz(args.path)
method = methods[args.format]

old_dim = npz.shape[0]
num_of_appends = np - (old_dim % np)
num_of_appends = 0 if num_of_appends == np else num_of_appends
new_n = old_dim + num_of_appends

print('Expanding...')
npz = hstack([npz, method((old_dim, num_of_appends))], args.format)
npz = vstack([npz, method((num_of_appends, new_n))], args.format)
print('Done')

print('Saving...')
path = args.path.replace('.npz', f'.{np:02d}.npz')
save_npz(path, npz)
print(f'Saved {path}')
