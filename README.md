# krylov

Krylov subspace methods module.
Do not publish code before publishing paper.

## dir

- archive
  - origin
  - thread
- converter
- cpp
  - archive
  - docs
  - npy
  - util
- doc
- method
  - build
  - cpp
  - origin
  - process
    - pyx
    - rewite
  - threads
- plot
- test
- util
- x

## requirements

### libs

- openmpi
- cuda(10.1)

### pip

- numpy
- scipy
- mpi4py
- cupy(10.1)

### only exec with cuda and mpiexec.hydra

- fastrlock

## settings

### on macOS(10.14.6)

#### export below param

``` sh
export PMIX_MCA_gds=hash
```

### with cupy (wrapper for cuda with python)

#### expand mermory allocator limit

```py
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)
```
