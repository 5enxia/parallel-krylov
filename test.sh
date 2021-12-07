# normal
python3 test.py cg
python3 test.py mrr
python3 test.py kskipcg

# mpi
mpiexec python3 test.py -m cg
mpiexec python3 test.py -m mrr
mpiexec python3 test.py -m kskipcg
