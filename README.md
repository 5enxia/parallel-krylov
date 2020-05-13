# krylov
Krylov subspace methods module.
Do not publish code before publishing paper.

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
