# The `linalg` package: a user guide

`wotan/linalg` is the dense linear-algebra layer of Wotan. It provides a heap-allocated matrix type and these operations on it:
- matrix products, including the multi-threaded SIMD kernels that the neural-network packages are built on;
- covariance and correlation;
- LU, Cholesky and QR factorisations and linear solves;
- least squares;
- symmetric and general eigenvalues, the singular value decomposition and condition numbers;
- a set of element-wise SIMD vector kernels.

Everything works on `f64`. The `analytics`, `ML` and `nn` packages build on it, so you will meet the `Matrix` type there too.

This guide walks through the package with small examples. It assumes you know basic Odin. Section 3 also uses a few DataFrame calls from the [core user guide](core_guide.md).

Every example was compiled and run against the current code, and the output shown is the real output. The numbers agree with numpy 1.23.5. The examples free everything they allocate: a debug build with a tracking allocator reports no leaks.

## Contents

1. [Setup](#1-setup)
2. [The Matrix type](#2-the-matrix-type)
3. [Matrices from DataFrames](#3-matrices-from-dataframes)
4. [Products, transposes and Kronecker products](#4-products-transposes-and-kronecker-products)
5. [Covariance and correlation](#5-covariance-and-correlation)
6. [LU: general square systems, inverses and determinants](#6-lu-general-square-systems-inverses-and-determinants)
7. [Cholesky: symmetric positive definite systems](#7-cholesky-symmetric-positive-definite-systems)
8. [QR and triangular solves](#8-qr-and-triangular-solves)
9. [Least squares](#9-least-squares)
10. [Eigenvalues, SVD and condition numbers](#10-eigenvalues-svd-and-condition-numbers)
11. [Vector kernels](#11-vector-kernels)
12. [Memory management](#12-memory-management)
13. [A complete example](#13-a-complete-example)
14. [Quick reference](#14-quick-reference)

## 1. Setup

Import the package under a short name. This guide uses `l`, as the tests and the other packages do:

```odin
import l "wotan/linalg"
import "core:fmt"
```

Build with `-debug` while you learn the library and install a tracking allocator, as described in the [core guide](core_guide.md#1-setup):

```odin
import "core:mem"

main :: proc() {
    track: mem.Tracking_Allocator
    mem.tracking_allocator_init(&track, context.allocator)
    context.allocator = mem.tracking_allocator(&track)
    defer {
        for _, e in track.allocation_map {fmt.printf("leak: %v bytes at %v\n", e.size, e.location)}
        mem.tracking_allocator_destroy(&track)
    }
    // ... your code ...
}
```

The examples print matrices with this small helper. It is not part of the library:

```odin
print_matrix :: proc(name: string, M: ^l.Matrix(f64)) {
    fmt.printf("%s (%d x %d)\n", name, M.rows, M.cols)
    for i in 0 ..< M.rows {
        fmt.print("  ")
        for j in 0 ..< M.cols {fmt.printf("%9.4f", M.data[i * M.cols + j])}
        fmt.println()
    }
}
```

## 2. The Matrix type

A `Matrix(T)` is a heap-allocated, row-major matrix:

```odin
Matrix :: struct($T: typeid) {
    rows:      int,
    cols:      int,
    data:      []T,           // rows * cols values, row by row
    allocator: mem.Allocator, // the allocator data came from
    stride:    int,           // informational only, see below
}
```

The element in row `i` and column `j` is `data[i * cols + j]`. The storage is always contiguous, so `stride` is never used for indexing. `matrix_from_flat` sets it to `cols`, while `matrix_new` leaves it at 0. Every procedure in the package works on `Matrix(f64)`.

`matrix_new` allocates a zero-filled matrix. `matrix_from_flat` copies a row-major slice into a new matrix. `matrix_at` returns a pointer to one element. `matrix_free` releases the data with the allocator recorded in the matrix, so you never pass the allocator again:

```odin
A := l.matrix_new(f64, 2, 3) // zero-filled
defer l.matrix_free(&A)
A.data[0 * A.cols + 2] = 7   // row 0, column 2
l.matrix_at(&A, 1, 0)^ = 5   // row 1, column 0
print_matrix("A", &A)

B := l.matrix_from_flat([]f64{1, 2, 3, 4, 5, 6}, 2, 3) // copies the slice
defer l.matrix_free(&B)
fmt.println(B.rows, B.cols, B.data)
```

```
A (2 x 3)
     0.0000   0.0000   7.0000
     5.0000   0.0000   0.0000
2 3 [1, 2, 3, 4, 5, 6]
```

- `matrix_new` panics if the allocation fails, rather than returning an empty matrix.
- `matrix_from_flat` panics if the slice is shorter than `rows * cols`.
- `matrix_at` does no bounds check beyond the slice's own.

Procedures take matrices by pointer (`^Matrix(f64)`) and do not modify them, unless this guide says so. The in-place ones are `cholesky*`, `jacobi_eigen_symmetric` and the accumulating `matmul_*_acc_simd` kernels, which modify their output argument.

### Fixed-size matrices

For small matrices whose size is known at compile time, use Odin's built-in `matrix[R, C]f64` type instead. It lives on the stack and needs no freeing. The package has three thin wrappers for it:

```odin
M := matrix[2, 2]f64{1, 2, 3, 4}
v := [2]f64{1, 1}
fmt.println(l.matmul(M, M), l.matvec(M, v), l.transpose(M))
```

```
matrix[7, 10; 15, 22] [3, 7] matrix[1, 3; 2, 4]
```

These are the same as writing `M * M`, `M * v` and `transpose(M)` directly. Nothing else in the package accepts fixed-size matrices.

## 3. Matrices from DataFrames

`matrix_from_df` copies the named columns of a `core` DataFrame into a new matrix, one column per name in the order given. `vector_from_df` copies one column into a `[]f64`. Int columns are converted to `f64`, and views are read correctly:

```odin
df := w.df_from(
    w.column_from_floats("ret",    []f64{0.01, -0.02, 0.015, 0.03}),
    w.column_from_ints("volume",   []int{100, 250, 175, 300}),
    w.column_from_strings("ticker", []string{"A", "B", "C", "D"}),
)
defer w.destroy_dataframe(&df)

X := l.matrix_from_df(&df, []string{"ret", "volume"})
defer l.matrix_free(&X)
print_matrix("X", &X)

y := l.vector_from_df(&df, "volume")
defer delete(y)
fmt.println(y)
```

```
X (4 x 2)
     0.0100 100.0000
    -0.0200 250.0000
     0.0150 175.0000
     0.0300 300.0000
[100, 250, 175, 300]
```

Both procedures panic on an unknown column name. They also panic on a column that is neither Float nor Int, such as `ticker` above.

**Nulls are not checked.** A null cell is read as the value stored underneath it, which is 0:

```odin
col := w.column_new("x", .Float, 3)
w.append_float(&col, 1.5)
w.append_null(&col)
w.append_float(&col, 2.5)
df2 := w.df_from(col)
defer w.destroy_dataframe(&df2)
v2 := l.vector_from_df(&df2, "x")
defer delete(v2)
fmt.println("with null:", v2)
```

```
with null: [1.5, 0, 2.5]
```

Drop or fill missing values in the DataFrame first, as the core guide describes.

## 4. Products, transposes and Kronecker products

```odin
A := l.matrix_from_flat([]f64{1, 2, 3, 4, 5, 6}, 2, 3)
defer l.matrix_free(&A)
B := l.matrix_from_flat([]f64{1, 0, 0, 1, 1, 1}, 3, 2)
defer l.matrix_free(&B)

C := l.matmul_dyn_simd(&A, &B) // A·B, a new 2 x 2 matrix
defer l.matrix_free(&C)
print_matrix("A·B", &C)

y := l.matvec_dyn_simd(&A, []f64{1, 1, 1}) // A·x, a new slice
defer delete(y)
fmt.println("A·1 =", y, " dot =", l.dot_simd([]f64{1, 2, 3}, []f64{4, 5, 6}))
```

```
A·B (2 x 2)
     4.0000   5.0000
    10.0000  11.0000
A·1 = [6, 15]  dot = 32
```

`matmul_dyn`, `matvec_dyn` and `dot` are plain scalar versions with the same signatures, and they give the same results. All of these panic on a dimension mismatch.

### Accumulating kernels

The three `_acc_` kernels add a product into an existing output matrix instead of allocating one. The autograd backward pass needs exactly this (`grad += …`):

| Procedure | Computes | Shapes |
|---|---|---|
| `matmul_nn_acc_simd(a, b, out)` | out += A·B | A: M×K, B: K×N, out: M×N |
| `matmul_nt_acc_simd(a, b, out)` | out += A·Bᵀ | A: M×K, B: N×K, out: M×N |
| `matmul_tn_acc_simd(a, b, out)` | out += Aᵀ·B | A: M×K, B: M×N, out: K×N |

For a plain product, start from a zero matrix:

```odin
out := l.matrix_new(f64, 2, 2)
defer l.matrix_free(&out)
l.matmul_nt_acc_simd(&A, &A, &out) // out += A·Aᵀ
print_matrix("A·Aᵀ", &out)
```

```
A·Aᵀ (2 x 2)
    14.0000  32.0000
    32.0000  77.0000
```

These kernels are register-blocked. Products of at least 2¹⁶ multiply-adds are split across threads by output row.

Each output element is computed by one thread, in the same order as in a single-threaded run, so results are bit-identical whatever the thread count. The package variable `l.matmul_max_threads` caps the threads: 0, the default, means one per logical processor, and 1 means always single-threaded.

### XᵀX, Xᵀy, transposes and Kronecker products

```odin
AtA := l.xtx_simd(&A)                // AᵀA, 3 x 3
defer l.matrix_free(&AtA)
Aty := l.xty(&A, []f64{1, -1})       // Aᵀy
defer delete(Aty)
fmt.println("AᵀA diag:", AtA.data[0], AtA.data[4], AtA.data[8], " Aᵀy =", Aty)

T := l.matrix_transpose(&A)
defer l.matrix_free(&T)
print_matrix("Aᵀ", &T)

K1 := l.matrix_from_flat([]f64{1, 2, 3, 4}, 2, 2)
defer l.matrix_free(&K1)
I2 := l.matrix_from_flat([]f64{1, 0, 0, 1}, 2, 2)
defer l.matrix_free(&I2)
K := l.kron(&K1, &I2)                // K1 ⊗ I
defer l.matrix_free(&K)
print_matrix("K1 ⊗ I", &K)
```

```
AᵀA diag: 17 29 45  Aᵀy = [-3, -3, -3]
Aᵀ (3 x 2)
     1.0000   4.0000
     2.0000   5.0000
     3.0000   6.0000
K1 ⊗ I (4 x 4)
     1.0000   0.0000   2.0000   0.0000
     0.0000   1.0000   0.0000   2.0000
     3.0000   0.0000   4.0000   0.0000
     0.0000   3.0000   0.0000   4.0000
```

Some of these procedures come in versions that give the same results:
- `xtx`, `xtx_simd` and `xtx_blocked_simd` all compute XᵀX.
- `kron` and `kron_simd` both compute the Kronecker product.
- `matrix_transpose` switches to `matrix_transpose_blocked` for matrices with more than 4096 elements.

## 5. Covariance and correlation

`covariance` treats each row as an observation and each column as a variable. It returns the sample covariance matrix, divided by n − 1, as `numpy.cov(X, rowvar=False)` does. `correlation` returns the Pearson correlation matrix:

```odin
D := l.matrix_from_flat([]f64{1, 2, 3, 4, 5, 7, 2, 1}, 4, 2) // 4 observations of 2 variables
defer l.matrix_free(&D)
S := l.covariance(&D)
defer l.matrix_free(&S)
R := l.correlation(&D)
defer l.matrix_free(&R)
print_matrix("cov", &S)
print_matrix("corr", &R)
```

```
cov (2 x 2)
     2.9167   4.1667
     4.1667   7.0000
corr (2 x 2)
     1.0000   0.9221
     0.9221   1.0000
```

`covariance` panics with fewer than two rows. `correlation` panics if a column has zero variance.

For shrinkage estimators, exponentially weighted covariance and PCA, see section 5 of the [analytics guide](analytics_guide.md#5-covariance-matrices-and-pca).

## 6. LU: general square systems, inverses and determinants

`lu_decompose` factors a square matrix as P·A = L·U with partial pivoting. It returns four values:
- `LU`: L and U combined in one matrix. L has a unit diagonal, which is not stored.
- `piv`: the row permutation.
- `sign`: the sign of the permutation, used for the determinant.
- `ok`: false if a pivot was exactly zero, that is, the matrix is singular.

The input matrix is not modified. Solve with `lu_solve`, and get the determinant from `lu_det`:

```odin
A := l.matrix_from_flat([]f64{4, 1, 2, 1, 3, 0, 2, 0, 5}, 3, 3)
defer l.matrix_free(&A)
b := []f64{1, 2, 3}

LU, piv, sign, ok := l.lu_decompose(&A)
defer l.matrix_free(&LU)
defer delete(piv)
if !ok {fmt.println("singular"); return}

x := l.lu_solve(&LU, piv, b) // reuse LU for as many right-hand sides as you like
defer delete(x)
fmt.println("x =", x, " det =", l.lu_det(&LU, sign))
```

```
x = [-0.30232558139534893, 0.7674418604651163, 0.7209302325581396]  det = 43
```

`lu_extract_LU` splits the combined factor into separate L and U matrices, which is useful for checking or printing:

```odin
L, U := l.lu_extract_LU(&LU)
defer l.matrix_free(&L)
defer l.matrix_free(&U)
print_matrix("L", &L)
print_matrix("U", &U)
```

```
L (3 x 3)
     1.0000   0.0000   0.0000
     0.2500   1.0000   0.0000
     0.5000  -0.1818   1.0000
U (3 x 3)
     4.0000   1.0000   2.0000
     0.0000   2.7500  -0.5000
     0.0000   0.0000   3.9091
```

`lu_solve_simd` is a variant of `lu_solve` that checks the dimensions and gives the same results.

### Inverses

There are two LU-based inverses. They differ in how they handle a singular matrix:

```odin
Ainv, inv_ok := l.mat_inverse_lu(&A)   // ok = false for a singular matrix
defer l.matrix_free(&Ainv)
print_matrix("A⁻¹", &Ainv)

S := l.matrix_from_flat([]f64{1, 2, 2, 4}, 2, 2) // singular
defer l.matrix_free(&S)
Sinv, sok := l.mat_inverse_lu(&S)
defer l.matrix_free(&Sinv)                       // safe: an empty matrix
fmt.println("singular:", sok, Sinv.rows, Sinv.cols)
```

```
A⁻¹ (3 x 3)
     0.3488  -0.1163  -0.1395
    -0.1163   0.3721   0.0465
    -0.1395   0.0465   0.2558
singular: false 0 0
```

`matrix_inverse(&A)` returns the inverse directly but **panics** on a singular matrix. Use it only when you know the matrix is invertible.

To solve a system, call `lu_solve` or `solve_spd_cholesky`. Computing an inverse and multiplying by it is slower and less accurate.

Singularity means an exactly zero pivot. A nearly singular matrix passes and gives large, inaccurate results. Check it first with the condition number from section 10.

## 7. Cholesky: symmetric positive definite systems

For a symmetric positive definite (SPD) matrix, such as a covariance matrix or XᵀX, the Cholesky factorisation A = L·Lᵀ is about twice as fast as LU, and it fails exactly when A is not positive definite.

`solve_spd_cholesky` and `spd_inverse` do the whole job and leave A unchanged:

```odin
x := l.solve_spd_cholesky(&A, []f64{1, 2, 3}) // A from section 6 is SPD
defer delete(x)
fmt.println("x =", x)

Ainv := l.spd_inverse(&A)
defer l.matrix_free(&Ainv)
fmt.printf("A⁻¹[0,0] = %.6f\n", Ainv.data[0])
```

```
x = [-0.30232558139534893, 0.7674418604651163, 0.7209302325581396]
A⁻¹[0,0] = 0.348837
```

Both panic if A is not positive definite. Both also accept a `mode` argument: `.Blocked` is the default and `.Unblocked` is the other choice. The two modes give the same factor to rounding; the blocked one is faster for large matrices.

The factorisation procedures work **in place**:
- `cholesky(&M, mode)` overwrites M with L. It writes L in the lower triangle and zeroes the upper one.
- `cholesky_try(&M)` does the same unblocked factorisation, but returns false instead of panicking when M is not positive definite. M is then partly overwritten.

Factor a copy if you still need the original. `forward_substitute_simd` and `backward_substitute_simd` then solve L·z = b and Lᵀ·x = z:

```odin
L := l.matrix_new(f64, 3, 3)
defer l.matrix_free(&L)
copy(L.data, A.data)                 // factor a copy: cholesky_try overwrites its argument
if !l.cholesky_try(&L) {fmt.println("not positive definite")}
print_matrix("L", &L)

z := l.forward_substitute_simd(&L, []f64{1, 2, 3})  // L z = b
defer delete(z)
x2 := l.backward_substitute_simd(&L, z)             // Lᵀ x = z
defer delete(x2)
fmt.println("x2 =", x2)

N := l.matrix_from_flat([]f64{1, 2, 2, 1}, 2, 2)    // eigenvalues 3 and -1
defer l.matrix_free(&N)
fmt.println("indefinite:", l.cholesky_try(&N))
```

```
L (3 x 3)
     2.0000   0.0000   0.0000
     0.5000   1.6583   0.0000
     1.0000  -0.3015   1.9771
x2 = [-0.30232558139534893, 0.7674418604651163, 0.7209302325581396]
indefinite: false
```

`cholesky_try` is the right test for positive definiteness, for example before using a sample covariance matrix. None of these procedures check symmetry: only the lower triangle of A is read.

## 8. QR and triangular solves

`qr_decompose` computes the full Householder QR factorisation A = Q·R of an m×n matrix. Q is m×m orthogonal, and R is m×n upper triangular. `qr_decompose_thin` returns only R:

```odin
A := l.matrix_from_flat([]f64{1, 1, 1, 2, 1, 3, 1, 4}, 4, 2)
defer l.matrix_free(&A)
Q, R := l.qr_decompose(&A)
defer l.matrix_free(&Q)
defer l.matrix_free(&R)
fmt.println(Q.rows, Q.cols, R.rows, R.cols)
print_matrix("R", &R)

Rt := l.qr_decompose_thin(&A) // R only
defer l.matrix_free(&Rt)
fmt.println(Rt.rows, Rt.cols)
```

```
4 4 4 2
R (4 x 2)
    -2.0000  -5.0000
    -0.0000   2.2361
    -0.0000   0.0000
    -0.0000  -0.0000
4 2
```

The signs follow the Householder convention: the diagonal of R may be negative, as in LAPACK.

Both procedures accept a `QR_Mode`, but it currently makes no difference. The former blocked variant was wrong for matrices with 32 or more columns, so both modes now use the same Householder code.

Q is m×m, so it needs m² memory. For least squares on a tall matrix, use `lstsq` from section 9, which never builds Q.

`upper_tri_solve(&U, b)` solves U·x = b by back substitution, for the leading `len(b)`×`len(b)` block of an upper triangular U:

```odin
U := l.matrix_from_flat([]f64{2, 1, 0, 3}, 2, 2)
defer l.matrix_free(&U)
xs := l.upper_tri_solve(&U, []f64{4, 6})
defer delete(xs)
fmt.println("upper_tri_solve:", xs)
```

```
upper_tri_solve: [1, 2]
```

## 9. Least squares

`lstsq(&X, y, method)` minimises ‖y − X·β‖². It returns an `LstsqResult`:

| Field | Meaning |
|---|---|
| `beta` | the coefficients, one per column of X |
| `residuals` | y − X·β, one per row |
| `rank` | numerical rank of X |
| `condition_estimate` | an estimate of the condition number; see below |
| `method_used` | the method that produced the result |
| `success` | false if the result is unreliable |

Free it with `lstsq_result_free(&res, allocator)`, passing the allocator you gave to `lstsq`, which is `context.allocator` by default.

```odin
// y = b0 + b1·x for x = 1..4
X := l.matrix_from_flat([]f64{1, 1, 1, 2, 1, 3, 1, 4}, 4, 2)
defer l.matrix_free(&X)
y := []f64{1, 2, 2, 4}

res := l.lstsq(&X, y)
defer l.lstsq_result_free(&res, context.allocator)
fmt.println(res.method_used, res.success, res.rank)
fmt.printf("beta = [%.4f, %.4f]  cond = %.2f\n", res.beta[0], res.beta[1], res.condition_estimate)
fmt.println("residuals =", res.residuals)
```

```
QR true 2
beta = [-0.0000, 0.9000]  cond = 1.12
residuals = [0.10000000000000009, 0.20000000000000018, -0.6999999999999997, 0.40000000000000036]
```

### Methods

| `LstsqMethod` | How | Use it when |
|---|---|---|
| `.QR` | Householder QR applied to X and y; Q is never formed | the default choice; accurate and O(m·n²) |
| `.Cholesky` | normal equations XᵀX·β = Xᵀy | X is well conditioned and you want speed; it squares the condition number |
| `.SVD` | singular value decomposition | X may be rank-deficient; gives the minimum-norm solution |
| `.Auto` (default) | picks one of the above | see below |

`.Auto` chooses as follows:
- QR for small problems (fewer than 50 rows and 20 columns) and for tall ones (at least 5 rows per column).
- Otherwise SVD if the column norms differ by more than a factor of 10⁶, and Cholesky if they don't.

Whichever method you ask for, `lstsq` falls back to the SVD in two cases, and then reports `method_used = .SVD`:

- **Rank-deficient or numerically singular X.** This is when QR or Cholesky finds a condition estimate of 10¹² or more. The SVD then gives the minimum-norm solution, which is what `numpy.linalg.lstsq` returns:

  ```odin
  Z := l.matrix_from_flat([]f64{1, 2, 2, 4, 3, 6}, 3, 2) // column 2 = 2 · column 1
  defer l.matrix_free(&Z)
  r2 := l.lstsq(&Z, []f64{1, 2, 3.1})
  defer l.lstsq_result_free(&r2, context.allocator)
  fmt.println(r2.method_used, r2.rank, r2.beta)
  ```

  ```
  SVD 1 [0.2042857142857143, 0.4085714285714286]
  ```

- **Fewer rows than columns.** An underdetermined system has infinitely many exact solutions, and the SVD returns the one with the smallest norm:

  ```odin
  W := l.matrix_from_flat([]f64{1, 2, 3, 4, 5, 6}, 2, 3)
  defer l.matrix_free(&W)
  r3 := l.lstsq(&W, []f64{1, 2})
  defer l.lstsq_result_free(&r3, context.allocator)
  fmt.println(r3.method_used, r3.rank, r3.beta)
  ```

  ```
  SVD 2 [-0.055555555555562186, 0.11111111111111022, 0.27777777777778256]
  ```

`condition_estimate` is a quick diagnostic and its meaning depends on the method:
- **QR:** the ratio of the largest to the smallest |Rⱼⱼ|. This is a rough estimate of cond(X).
- **Cholesky:** the square of that ratio, taken from the diagonal of L. This estimates cond(XᵀX).
- **SVD:** σ_max / σ_min over the singular values above the rank tolerance. This is the exact cond(X) restricted to the column space.

For the exact 2-norm condition number of X, use `cond2_svd` (section 10).

`lstsq` gives coefficients only. For standard errors, t-statistics, robust covariance and the rest of a regression table, use `ml.ols_fit` from the `ML` package, described in section 4 of the [ML guide](ml_guide.md#4-linear-regression).

## 10. Eigenvalues, SVD and condition numbers

### Symmetric matrices

`eigh` computes the eigenvalues and eigenvectors of a symmetric matrix by cyclic Jacobi rotations, which are accurate to machine precision. It returns the eigenvalues in the requested order, `.Ascending` (the default) or `.Descending`. The matching eigenvectors are the columns of V:

```odin
A := l.matrix_from_flat([]f64{4, 1, 2, 1, 3, 0, 2, 0, 5}, 3, 3)
defer l.matrix_free(&A)

vals, vecs := l.eigh(&A, .Descending)
defer delete(vals)
defer l.matrix_free(&vecs)
fmt.println("eigenvalues:", vals)
fmt.printf("first eigenvector: [%.4f, %.4f, %.4f]\n", vecs.data[0], vecs.data[3], vecs.data[6])
```

```
eigenvalues: [6.669079088282286, 3.4760236029181333, 1.8548973087995764]
first eigenvector: [0.6312, 0.1720, 0.7563]
```

`eigh` works on a copy and leaves A unchanged.

It checks that A is square and symmetric (to 10⁻¹²) only in `-debug` builds. In a release build, a non-symmetric A gives meaningless results without any warning.

The sign of each eigenvector is arbitrary, as in every eigen solver.

`jacobi_eigen_symmetric(&A)` is the underlying routine. It **overwrites A** and returns the eigenvalues unsorted.

### General matrices

`eigvals_general` returns the eigenvalues of any real square matrix as two slices, real and imaginary parts. Complex eigenvalues come in conjugate pairs. It uses Hessenberg reduction followed by the Francis double-shift QR iteration (EISPACK `elmhes`/`hqr`).

It does not compute eigenvectors. `eigvals_modulus_max` gives the spectral radius, which is the usual stability check for a VAR or a linear recursion:

```odin
B := l.matrix_from_flat([]f64{2, 1, 0, -1, 2, 1, 0, -1, 1}, 3, 3)
defer l.matrix_free(&B)
re, im, ok := l.eigvals_general(&B)
defer delete(re)
defer delete(im)
for i in 0 ..< len(re) {fmt.printf("  %.6f %+.6fi\n", re[i], im[i])}
fmt.printf("ok = %v  spectral radius = %.6f\n", ok, l.eigvals_modulus_max(&B))
```

```
  1.784920 -1.307141i
  1.784920 +1.307141i
  1.430160 +0.000000i
ok = true  spectral radius = 2.212365
```

The order is the order in which the iteration finds the eigenvalues. If an eigenvalue fails to converge within 30 iterations, `ok` is false and every entry is NaN, and `eigvals_modulus_max` returns NaN.

### Singular value decomposition

`svd` computes A = U·diag(S)·Vᵀ with the one-sided Jacobi method. The singular values come in descending order.

For an m×n matrix:
- U is m×n.
- S has n entries.
- V is n×n.

When m < n, the last n − m singular values are 0.

`svd_thin` keeps only the r singular values above a rank tolerance of max(m, n)·10⁻¹²·σ_max. Its U is m×r and its V is n×r:

```odin
U, S, V := l.svd(&A)
defer l.matrix_free(&U)
defer delete(S)
defer l.matrix_free(&V)
fmt.println("singular values:", S)

W := l.matrix_from_flat([]f64{1, 2, 3, 4, 5, 6}, 2, 3)
defer l.matrix_free(&W)
Ut, St, Vt := l.svd_thin(&W)
defer l.matrix_free(&Ut)
defer delete(St)
defer l.matrix_free(&Vt)
fmt.println("thin:", Ut.rows, Ut.cols, St, Vt.rows, Vt.cols)
```

```
singular values: [6.669079088282286, 3.4760236029181337, 1.8548973087995773]
thin: 2 2 [9.508032000695724, 0.772869635673484] 3 2
```

The `SVDMethod` argument accepts `.Jacobi` and `.GolubReinsch`, but both run the same one-sided Jacobi code. The name `svd_golub_reinsch` is kept for compatibility. It used to compute the SVD from the eigenvalues of AᵀA, which loses the accuracy of small singular values.

### Condition numbers

| Procedure | Returns |
|---|---|
| `cond2_svd(&A)` | σ_max / σ_min, the 2-norm condition number; +Inf if A is rank-deficient |
| `cond2_svd_thin(&A)` | the same ratio over the singular values above the rank tolerance |
| `cond2_sym(&A)` | λ_max / λ_min for a symmetric matrix; +Inf unless A is positive definite |
| `rcond2_svd(&A)`, `rcond2_sym(&A)` | the reciprocals; 0 for a singular matrix |

```odin
fmt.printf("cond2 = %.6f  rcond = %.6f  cond2_sym = %.6f\n", l.cond2_svd(&A), l.rcond2_svd(&A), l.cond2_sym(&A))
```

```
cond2 = 3.595390  rcond = 0.278134  cond2_sym = 3.595390
```

As a rule of thumb, a linear solve loses about log₁₀(cond) of its 16 significant digits.

## 11. Vector kernels

The package also has element-wise kernels on `[]f64` slices. They use SIMD instructions when the target supports them and fall back to scalar loops otherwise. They write into an output slice you provide, so they allocate nothing:

```odin
a := []f64{1, -2, 3, -4}
b := []f64{0.5, 0.5, 0.5, 0.5}
out := make([]f64, 4)
defer delete(out)

l.vec_add_simd(a, b, out);   fmt.println("add  ", out)
l.vec_mul_simd(a, b, out);   fmt.println("mul  ", out)
l.vec_scale_simd(a, 2, out); fmt.println("scale", out)
l.vec_relu_simd(a, out);     fmt.println("relu ", out)
fmt.println("sum  ", l.sum_simd(a))

y := []f64{1, 1, 1, 1}
l.axpy_simd(10, a, y)            // y += 10·a
fmt.println("axpy ", y)
c := []f64{0, 0, 0, 0}
l.vec_fma_inplace_simd(a, b, c)  // c += a·b
fmt.println("fma  ", c)
```

```
add   [1.5, -1.5, 3.5, -3.5]
mul   [0.5, -1, 1.5, -2]
scale [2, -4, 6, -8]
relu  [1, 0, 3, 0]
sum   -2
axpy  [11, -19, 31, -39]
fma   [0.5, -1, 1.5, -2]
```

Every kernel panics if the slice lengths differ.

| Procedure | Operation |
|---|---|
| `vec_add_simd(a, b, out)`, `vec_sub_simd`, `vec_mul_simd` | out = a + b, a − b, a · b |
| `vec_sub_inplace_simd(a, b)` | a −= b |
| `vec_fma_inplace_simd(a, b, c)` | c += a · b |
| `vec_scale_simd(a, s, out)`, `vec_neg_simd(a, out)` | out = s · a, −a |
| `vec_broadcast_add_simd(s, out)` | out += s |
| `axpy_simd(alpha, x, y)` | y += alpha · x |
| `ger_simd(alpha, x, y, &A)` | A += alpha · x·yᵀ (rank-1 update) |
| `rotate_pair_simd(c, s, p, q, n)` | Givens rotation of the first n entries of p and q |
| `dot_simd(a, b)`, `sum_simd(a)` | a·b, Σa |
| `vec_relu_simd`, `vec_leaky_relu_simd(a, alpha, out)`, `vec_sigmoid_simd`, `vec_tanh_simd` | activations, out = f(a) |
| `vec_relu_backward_simd(grad_out, a_in, grad_in)` | grad_in += grad_out where a_in > 0 |
| `vec_sigmoid_backward_simd(grad, a, out)`, `vec_tanh_backward_simd` | out += grad · f′(a), where `a` is the **input** of the activation |

The backward kernels **add** into their output, like the matmul kernels. Zero the output first if you want the plain gradient.

## 12. Memory management

The rules are the same as in the rest of Wotan:

- **Every procedure that returns a `Matrix` or a slice allocates it.** Free a matrix with `matrix_free` and a slice with `delete`. `matrix_free` uses the allocator stored in the matrix, so it is always correct.
- **Delete a slice with the allocator it was allocated with.** Most procedures take an optional `allocator` argument, which defaults to `context.allocator`. If you pass another allocator, delete the returned slices with that same allocator: `delete(x, my_alloc)`.
- **Free an `LstsqResult` with `lstsq_result_free(&res, allocator)`.** It frees `beta` and `residuals`.
- **Internal scratch is freed inside each procedure.** Only the documented outputs are left for you to free, and a tracking allocator reports no leaks for any procedure in this guide. Some procedures use `context.temp_allocator` for short-lived buffers, and they free those too.
- **One exception returns temporary memory.** `apply_pivots_vec` returns a slice on `context.temp_allocator`. Do not `delete` it with the default allocator; it is released by `free_all(context.temp_allocator)`.
- **A matrix may live on the temp allocator.** Passing `context.temp_allocator` and never freeing is fine for short computations, as long as you call `free_all(context.temp_allocator)` regularly, for example once per loop iteration.
- **A zero-sized matrix is safe to free.** `mat_inverse_lu` on a singular matrix returns an empty matrix, and `matrix_free` on it does nothing harmful.

These leaks and bugs were fixed when this guide was written:
- **Leaks:** `lu_solve`, `mat_inverse_lu`, `svd_thin`, `cond2_svd`, `cond2_svd_thin`, `cond2_sym` and the SVD path of `lstsq` used to leak memory.
- **Default allocator:** `svd_jacobi` used to default to the temporary allocator. It now defaults to `context.allocator`, like every other procedure.
- **`lstsq` crash:** `lstsq` used to read out of bounds when X had fewer rows than columns.
- **`lstsq` QR results:** with `.Auto` or `.QR`, `lstsq` used to return useless coefficients of about 10¹³ for a rank-deficient X.

[`tests/linalg_memory_test.odin`](../tests/linalg_memory_test.odin) guards against all of these.

## 13. A complete example

This program fits a two-factor regression with `lstsq`. It then computes the classical OLS standard errors σ̂²·(XᵀX)⁻¹ with the Cholesky inverse, checks the conditioning of X, and runs the example under a tracking allocator:

```odin
package main

import "core:fmt"
import "core:math"
import "core:mem"
import l "wotan/linalg"

main :: proc() {
    track: mem.Tracking_Allocator
    mem.tracking_allocator_init(&track, context.allocator)
    context.allocator = mem.tracking_allocator(&track)
    defer {
        for _, e in track.allocation_map {fmt.printf("leak: %v bytes at %v\n", e.size, e.location)}
        mem.tracking_allocator_destroy(&track)
    }

    // 8 observations: intercept, market excess return, size factor
    X := l.matrix_from_flat([]f64{
        1,  0.012,  0.004,
        1, -0.008,  0.002,
        1,  0.020, -0.006,
        1,  0.004,  0.010,
        1, -0.015, -0.003,
        1,  0.009,  0.007,
        1, -0.002, -0.009,
        1,  0.017,  0.001,
    }, 8, 3)
    defer l.matrix_free(&X)
    y := []f64{0.015, -0.009, 0.019, 0.010, -0.017, 0.013, -0.006, 0.019}

    fit := l.lstsq(&X, y)
    defer l.lstsq_result_free(&fit, context.allocator)
    if !fit.success {fmt.println("unreliable fit"); return}

    n, p := X.rows, X.cols
    ssr := l.dot_simd(fit.residuals, fit.residuals)
    sigma2 := ssr / f64(n - p)

    XtX := l.xtx_simd(&X)
    defer l.matrix_free(&XtX)
    XtX_inv := l.spd_inverse(&XtX)
    defer l.matrix_free(&XtX_inv)

    names := []string{"alpha", "beta_mkt", "beta_smb"}
    fmt.printf("%-9s %10s %10s %8s\n", "", "coef", "std err", "t")
    for j in 0 ..< p {
        se := math.sqrt(sigma2 * XtX_inv.data[j * p + j])
        fmt.printf("%-9s %10.5f %10.5f %8.2f\n", names[j], fit.beta[j], se, fit.beta[j] / se)
    }
    fmt.printf("method %v, rank %d, cond2(X) = %.1f\n", fit.method_used, fit.rank, l.cond2_svd(&X))
}
```

```
                coef    std err        t
alpha        0.00012    0.00030     0.40
beta_mkt     1.08751    0.02419    44.95
beta_smb     0.46952    0.04595    10.22
method QR, rank 3, cond2(X) = 167.4
```

The coefficients, standard errors and t-values agree with `statsmodels.OLS(y, X).fit()` to the digits shown. A condition number of 167 comes from the factor columns being about 100 times smaller than the intercept column. It costs about two of the 16 digits, which is harmless. In practice, `ml.ols_fit` from the [ML guide](ml_guide.md#4-linear-regression) computes this whole table and more.

## 14. Quick reference

| Task | Procedure |
|---|---|
| New zero matrix / from a slice / free | `matrix_new(f64, r, c)`, `matrix_from_flat(data, r, c)`, `matrix_free(&M)` |
| Element pointer | `matrix_at(&M, i, j)` |
| From a DataFrame | `matrix_from_df(&df, cols)`, `vector_from_df(&df, col)` |
| Products | `matmul_dyn_simd`, `matvec_dyn_simd`, `dot_simd` (scalar: `matmul_dyn`, `matvec_dyn`, `dot`) |
| Accumulating products | `matmul_nn_acc_simd`, `matmul_nt_acc_simd`, `matmul_tn_acc_simd`; threads: `matmul_max_threads` |
| XᵀX, Xᵀy | `xtx_simd` (also `xtx`, `xtx_blocked_simd`), `xty` |
| Transpose, Kronecker | `matrix_transpose`, `kron` / `kron_simd` |
| Covariance, correlation | `covariance`, `correlation` |
| LU | `lu_decompose`, `lu_solve`, `lu_det`, `lu_extract_LU` |
| Inverse | `mat_inverse_lu` (returns ok), `matrix_inverse` (panics if singular), `spd_inverse` (SPD) |
| Cholesky | `cholesky(&M, mode)`, `cholesky_try`, `solve_spd_cholesky`, `forward_substitute_simd`, `backward_substitute_simd` |
| QR | `qr_decompose`, `qr_decompose_thin`, `upper_tri_solve` |
| Least squares | `lstsq(&X, y, method)`, `lstsq_result_free` |
| Symmetric eigen | `eigh(&A, order)`, `jacobi_eigen_symmetric` (in place, unsorted) |
| General eigenvalues | `eigvals_general`, `eigvals_modulus_max` |
| SVD | `svd`, `svd_thin`, `svd_jacobi` |
| Condition number | `cond2_svd`, `cond2_svd_thin`, `cond2_sym`, `rcond2_svd`, `rcond2_sym` |
| Vector kernels | `vec_*_simd`, `axpy_simd`, `ger_simd`, `sum_simd`, `rotate_pair_simd` |
