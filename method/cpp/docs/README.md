# OpenBLAS

## cblas_dgemm
```cpp
// alpha*A*B + beta*C
cblas_dgemm( const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const int M,                        // 行列Aの行数
    const int N,                        // 行列Bの列数
    const int K,                        // 行列Aの列数、行列Bの行数
    const double alpha,                 // 行列の積に掛けるスカラ値(なければ1を設定)
    const double *A,                    // 行列A
    const int ldA,                      // Aのleading dimension (通常は行数を指定すれば良い）
    const double *B,                    // 行列B
    const int ldB,                      // Bのleading dimension
    const double beta,                  // 行列Cに掛けるスカラ値(なければ0を設定)
    double *C,                          // 行列C（ＡとＢの積） !破壊され結果が代入される
    const int ldc );                    // Cのleading dimension

// Orderには行列の形式を指定
enum CBLAS_ORDER {
	CblasRowMajor=101,		// 行形式
	CblasColMajor=102		// 列形式
};

// TransAおよびTransBには積を求める前に行列を転置するかどうかを指定
enum CBLAS_TRANSPOSE {
	CblasNoTrans=111,		// 転置なし
	CblasTrans=112,			// 転置
	CblasConjTrans=113		// 共役転置
};
```