# k段飛ばしアルゴリズム

## 実装前

```py {.line-numbers}
for j in range(1, k + 1):
    Ar[j] = dot(A, Ar[j-1])
for j in range(1, k + 2):
    Ap[j] = dot(A, Ap[j-1])
for j in range(2 * k + 1):
    jj = j // 2
    a[j] = dot(Ar[jj], Ar[jj + j % 2])
for j in range(2 * k + 4):
    jj = j // 2
    f[j] = dot(Ap[jj], Ap[jj + j % 2])
for j in range(2 * k + 2):
    jj = j // 2
    c[j] = dot(Ar[jj], Ap[jj + j % 2])
```

## 処理手順

## Ar, Ap共通

係数行列AをScatterして，local_Aに渡す.（Aを生成した後に，一回だけ実行するだけでよい）

#### Ar

1. Ar[i-1]をBroadCast
2. 各プロセスでlocal_Ar = loca_A.dot(Ar[i-1])を実行
3. local_ArをAr[i]にgather
4. 1-3を繰り返す

#### Ap

1. Ap[i-1]をBroadCast
2. 各プロセスでlocal_Ap = loca_A.dot(Ap[i-1])を実行
3. local_ApをAp[i]にgather
4. 1-3を繰り返す

### a(alpha), f(beta), c(zeta)共通

1. ArをScatter(+1行を渡す必要がある)　or Bcast
2. ApをScatter(+1行を渡す必要がある) or Bcast

#### a(alpha)

1. 各プロセスで

```py
local_a[j] = dot(
    Ar[jj][ランク＊ローカルカラム数：ランク*ローカルカラム数], 
    Ar[jj + j % 2][ランク＊ローカルカラム数：ランク*ローカルカラム数]
)
```

を実行

2. a[j]をReduce(sum)

#### f(beta)

1. 各プロセスで

```py
local_f[j] = dot(
    Ap[jj][ランク＊ローカルカラム数：ランク*ローカルカラム数], 
    Ap[jj + j % 2][ランク＊ローカルカラム数：ランク*ローカルカラム数]
)
```

を実行

2. f[j]をReduce(sum)

#### c(zeta)

1. 各プロセスで

```py
local_c[j] = dot(
    Ar[jj][ランク＊ローカルカラム数：ランク*ローカルカラム数], 
    Ap[jj + j % 2][ランク＊ローカルカラム数：ランク*ローカルカラム数]
)
```

を実行

2. c[j]をReduce(sum)

## 実装後

```py
comm.Scatter(A, local_A, root=0)
for j in range(1, k + 1):
    comm.Bcast(Ar[j-1], root=0)
    local_Ar = dot(loca_A, Ar[j-1])
    comm.Gather(local_Ar, Ar[j], root=0)
for j in range(1, k + 2):
    comm.Bcast(Ap[j-1], root=0)
    local_Ap = dot(local_A, Ap[j-1])
    comm.Gather(local_Ap, Ap[j], root=0)
comm.Bcast(Ar)
comm.Bcast(Ap)
for j in range(2 * k + 1):
    jj = j // 2
    local_a[j] = dot(
        Ar[jj][rank * local_N: (rank+1) * local_N],
        Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
    )
comm.Reduce(local_a, a, root=0)
for j in range(2 * k + 4):
    jj = j // 2
    local_f[j] = dot(
        Ap[jj][rank * local_N: (rank+1) * local_N],
        Ap[jj + j % 2][rank * local_N: (rank+1) * local_N]
    )
comm.Reduce(local_f, f, root=0)
for j in range(2 * k + 2):
    jj = j // 2
    local_c[j] = dot(
        Ar[jj],
        Ap[jj + j % 2][rank * num_of_local_col: (rank+1) * num_of_local_col]
    )
comm.Reduce(local_c, c, root=0)
```
