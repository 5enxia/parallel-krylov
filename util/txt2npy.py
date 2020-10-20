import loader
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='.txt to .npy converter')
parser.add_argument("path", help="data file path")
parser.add_argument("-t", "--type", help="vector file")
parser.add_argument("-m", "--matrix", action="store_true",
                    help="matrix file")
parser.add_argument("-v", "--vector", action="store_true",
                    help="vector file")

args = parser.parse_args()
if args.matrix:
    matrix = loader.load_matrix(args.path, args.type)
    np.save('matrix.npy', matrix)
elif args.vector:
    vector = loader.load_vector(args.path, args.type)
    np.save('vector.npy', vector)

print(args)
