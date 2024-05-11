// git clone https://github.com/NVIDIA/cutlass.git
// nvcc -O3 --use_fast_math --expt-relaxed-constexpr -arch=sm_80 -I/home/ubuntu/cutlass/include sgemm.cu -o sgemm && ./sgemm

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cute/tensor.hpp>
#include "helpers.h"

#define TILE_SIZE 16
#define BLOCK_SIZE 16

void MatMulCPU(float *A, float *B, float *C, int N, int K)
{
  for (int row = 0; row < N; row++)
  {
    for (int col = 0; col < N; col++)
    {
      float sum = 0.0;
      for (int k = 0; k < K; k++)
      {
        sum += A[col * K + k] * B[row * K + k];
      }
      C[row * N + col] = sum;
    }
  }
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                                                                        TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                                                                                        TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                                                                                        TC *C, CStride dC, CSmemLayout, TiledMma mma, int KEY_BLOCK)
{
  using namespace cute;

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  for (int key_block = 0; key_block < KEY_BLOCK; ++key_block)
  {
    // todo put matmul function here
    auto query_block = blockIdx.z;
    auto cta_coord = make_coord(query_block, key_block, _);

    // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K)
    // Allocate registers same shape/layout as partitioned data
    Tensor tArA = make_fragment_like(tAsA); // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K)
    // Allocate registers same shape/layout as partitioned data
    Tensor tBrB = make_fragment_like(tBsB); // (CPY,CPY_N,CPY_K)

    // Copy gmem to rmem for k_tile=0
    copy(copy_a, tAgA(_, _, _, 0), tArA);
    copy(copy_b, tBgB(_, _, _, 0), tBrB);
    //
    // Define A/B partitioning and C accumulators
    //

    // TUTORIAL: Example of partitioning via a TiledMMA

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)
    // Clear the accumulators
    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
      // Copy rmem to smem with tA|tB thread-partitioned tensors
      __syncthreads(); // Wait for all threads to consume smem
      copy(tArA, tAsA);
      copy(tBrB, tBsB);
      __syncthreads(); // Wait for all threads to consume smem

      // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
      int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
      copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
      copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);

      gemm(mma, tCsA, tCsB, tCrC);

      axpby(1.0, tCrC, 0, tCgC);
    }
  }
}

// TA, TB, TC = the type of variable we are dealing with
void MatMulCaller(float *A, float *B, float *C, int N, int K)
{
  using namespace cute;
  // Define shapes (dynamic)
  // auto N = int(N);
  auto M = int(N);
  // auto K = int(K);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(K, Int<1>{}); // (dM, dK)
  auto dB = make_stride(K, Int<1>{}); // (dN, dK)
  auto dC = make_stride(Int<1>{}, M); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK),
                        make_stride(Int<1>{}, bM + Int<1>{})); // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape(bN, bK),
                        make_stride(Int<1>{}, bN + Int<1>{})); // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN));                   // (m,n) -> smem_idx

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                    Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape<_1, _1>>{});                 // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                    Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape<_1, _1>>{});                 // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                                 Layout<Shape<_4, _2, _1>>{}); // 16x16x1 TiledMMA

  // TiledMMA mmaC = make_tiled_mma(UniversalFMA<float, float, float>{},
  //                                Layout<Shape<_16, _16, _1>>{}); // 16x16x1 TiledMMA

  dim3 dimBlock(size(mmaC));
  // int a = size(mmaC);
  // printf("%d\n", a);
  dim3 dimGrid(1, 1, size(ceil_div(M, bM)));
  int KEY_BLOCK = ceil_div(N, bN);

  cudaStream_t stream = 0;
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler,
                                                A, dA, sA, copyA,
                                                B, dB, sB, copyB,
                                                C, dC, sC, mmaC, KEY_BLOCK);
}

int main(int argc, char **argv)
{
  srand(0);
  int N = 1024; // pow(2, 10); // 512x512 matrix
  int K = 64;
  float *C = (float *)malloc(N * N * sizeof(float));
  float *C_cpu = (float *)malloc(N * N * sizeof(float)); // CPU result storage
  float *A = make_random_float(N * K);
  float *B = make_random_float(N * K);

  // Compute matrix multiplication on CPU for validation
  MatMulCPU(A, B, C_cpu, N, K);

  // CUDA-related allocations and computations
  float *d_C;
  float *d_A;
  float *d_B;
  cudaCheck(cudaMalloc(&d_C, N * N * sizeof(float)));
  cudaCheck(cudaMalloc(&d_A, N * K * sizeof(float)));
  cudaCheck(cudaMalloc(&d_B, N * K * sizeof(float)));
  cudaCheck(cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice));

  // printf("%d", __CUDA_ARCH__);
  float elapsed_time = benchmark_kernel(100, MatMulCaller, d_A, d_B, d_C, N, K);
  printf("Elapsed time: %f ms\n", elapsed_time);
  // Validate results
  float tol = 1e-2;
  validate_result(d_C, C_cpu, "MatMulKernel", N * N, tol);

  // Clean up
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(C);
  free(C_cpu);
  free(A);
  free(B);
}