#### before exec mpiexec

``` sh
export PMIX_MCA_gds=hash
```

#### expand mermory allocator limit
```py
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)
```
