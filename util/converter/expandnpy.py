import argparse
from numpy import load, save, hstack, vstack
from numpy import zeros

parser = argparse.ArgumentParser(description='divide .npz to n files')
parser.add_argument("path", help="data file path")
parser.add_argument("n", help="number of processes")

args = parser.parse_args()
np = int(args.n)
npy = load(args.path)

old_dim = npy.shape[0]
num_of_appends = np - (old_dim % np)
num_of_appends = 0 if num_of_appends == np else num_of_appends
new_n = old_dim + num_of_appends

print('Expanding...')
npy = hstack([npz, zeros((old_dim, num_of_appends))])
npy = vstack([npz, zeros((num_of_appends, new_n))])
print('Done')

print('Saving...')
path = args.path.replace('.npy', f'.{np:02d}.npy')
save(path, npy)
print(f'Saved {path}')
