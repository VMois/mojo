from benchmark import Benchmark
from sys.intrinsics import strided_load
from utils.list import VariadicList
from math import div_ceil, min
from memory import memset_zero
from random import rand, random_float64
from sys.info import simdwidthof
from time import now
from algorithm import vectorize, parallelize, vectorize_unroll
from runtime.llcl import Runtime

struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        rand(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)


# fn matmul_naive(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
#     for m in range(C.rows):
#         for k in range(A.cols):
#             for n in range(C.cols):
#                 C[m, n] += A[m, k] * B[k, n]


# Mojo has SIMD vector types, we can vectorize the Matmul code as follows.
alias nelts = 16 * simdwidthof[DType.float32]()  # The SIMD vector width.


# fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
#     for m in range(C.rows):
#         for k in range(A.cols):
#             for nv in range(0, C.cols, nelts):
#                 C.store[nelts](
#                     m, nv, C.load[nelts](m, nv) + A[m, k] * B.load[nelts](k, nv)
#                 )

#             # Handle remaining elements with scalars.
#             for n in range(nelts * (C.cols // nelts), C.cols):
#                 C[m, n] += A[m, k] * B[k, n]


# Simplify the code by using the builtin vectorize function
# from Functional import vectorize
fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

# Parallelize the code by using the builtin parallelize function
# from Functional import parallelize
fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

    parallelize[calc_row](rt, C.rows)

@always_inline
fn benchmark[
    func: fn (Matrix, Matrix, Matrix, Runtime) -> None
](M: Int, N: Int, K: Int, str: String):
    var C = Matrix(M, N)
    C.zero()
    var A = Matrix(M, K)
    var B = Matrix(K, N)

    with Runtime() as rt:
        @always_inline
        @parameter
        fn test_fn():
            _ = func(C, A, B, rt)

        let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
        # Prevent the matrices from being freed before the benchmark run
        _ = (A, B, C)
        let gflops = ((2 * M * N * K) / secs) / 1e9
        print(str)
        print(gflops, "GFLOP/s")


fn main():
    alias M = 512
    print("Nelts size: ", nelts)
    benchmark[matmul_vectorized_1](
        M,
        M,
        M,
        (
            "Throughput of a 512x512 matrix multiplication in Mojo using the"
            " stdlib `vectorize`: "
        ),
    )
    benchmark[matmul_parallelized](
        M,
        M,
        M,
        (
            "Throughput of a 512x512 {vectorized + parallelized} matrix"
            " multiplication in Mojo: "
        ),
    )
