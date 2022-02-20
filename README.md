# krylov

Krylov subspace methods module.

## Methods
- CG
- MrR
- *k*-skip CG
- *k*-skip MrR
- Adaptive *k*-skip MrR

## directories
- v3
  - cpu
    - mpi
      - cg
      - mrr
      - kskipcg
      - kskipmrr
      - adaptivekskipmrr
    - cg
    - mrr
    - kskipcg
    - kskipmrr
    - adaptivekskipmrr
  - gpu
    - mpi
      - cg
      - mrr
      - kskipcg
      - kskipmrr
      - adaptivekskipmrr
    - cg
    - mrr
    - kskipcg
    - kskipmrr
    - adaptivekskipmrr

## requirements

### C libs

#### only CPU
- C Compiler
  - GCC
  - Intel C Compiler
  - etc...
- BLAS library
  - [OpenBLAS](https://www.openblas.net/)
  - AppleBLAS
  - etc..

#### with GPU
- CUDA(10.1)

#### with MPI
- MPI library
  - OpenMPI(https://www.open-mpi.org/)
  - MVAPICH2(http://mvapich.cse.ohio-state.edu/)
  - Intel MPI
  - etc...

### Pyhton3 modules

#### only CPU

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)

#### with GPU

- [cupy9.6.0](https://github.com/cupy/cupy)

#### with MPI

- [mpi4py](https://github.com/mpi4py/mpi4py)

#### only exec with cuda and mpiexec.hydra(Intel MPI)

- fastrlock

## settings

### on macOS(10.14.6)

#### export below param

``` sh
export PMIX_MCA_gds=hash
```

### with cupy

#### expand mermory allocator limit

```py
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)
```
