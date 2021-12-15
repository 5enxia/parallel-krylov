# normal
python3 test.py cg
python3 test.py mrr
python3 test.py kskipcg
python3 test.py kskipmrr
python3 test.py adaptivekskipmrr

# mpi
mpiexec python3 test.py --mpi cg
mpiexec python3 test.py --mpi mrr
mpiexec python3 test.py --mpi kskipcg
mpiexec python3 test.py --mpi kskipmrr
mpiexec python3 test.py --mpi adaptivekskipmrr

# nomal gpu
python3 test.py --gpu cg