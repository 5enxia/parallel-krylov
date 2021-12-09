# normal
python3 test.py cg
python3 test.py mrr
python3 test.py kskipcg
python3 test.py kskipmrr
python3 test.py adaptivekskipmrr

# mpi
mpiexec python3 test.py -m cg
mpiexec python3 test.py -m mrr
mpiexec python3 test.py -m kskipcg
mpiexec python3 test.py -m kskipmrr
