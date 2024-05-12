/*
Kernels for attention forward pass.

Compile example:
nvcc -O3 --use_fast_math -I<PATH_TO_CUTLASS>/cutlass/include attention_forward.cu --expt-relaxed-constexpr -o attention_forward -lcublas
nvcc -O3 --use_fast_math -I/home/ubuntu/cutlass/include attention_forward.cu --expt-relaxed-constexpr -o attention_forward -lcublas --generate-line-info  && ./attention_forward 6
nvcc -O3 --use_fast_math -I/home/ubuntu/cutlass/include -arch=sm_80 attention_forward.cu --expt-relaxed-constex
pr -o attention_forward -lcublas --generate-line-info  && ./attention_forward 1

version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
./attention_forward 1

version 2 is a naive implementation of flash attention, taken, adapted from
https://github.com/tspeterkim/flash-attention-minimal
and with help from
https://github.com/leloykun/flash-hyperbolic-attention-minimal
sadly, this flash attention version seems about 3X slower than the naive version
./attention_forward 2

version 3 is a cuBLAS + softmax version, similar to the PyTorch implementation
cuBLAS is used both to calculate the QK^T and the final weighted sum
the softmax is calculated using a custom, efficient kernel as well
this turns out to be ~20X faster than (1) nice
./attention_forward 3

version 4 is a further optimized kernel that fuses the scale operation,
uses a directly autoregressive softmax, and uses the online softmax algorithm.
./attention_forward 4
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cute/tensor.hpp>
#include "common.h"

// ----------------------------------------------------------------------------
// CUDA setup

static cublasHandle_t cublas_handle;

// ----------------------------------------------------------------------------
// CPU code reference

void attention_forward_cpu(float *out, float *preatt, float *att,
                           const float *inp,
                           int B, int T, int C, int NH)
{
  // input is (B, T, 3C) Q,K,V
  // preatt, att are (B, NH, T, T)
  // output is (B, T, C)
  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.0 / sqrtf(hs);

  for (int b = 0; b < B; b++)
  {
    for (int t = 0; t < T; t++)
    {
      for (int h = 0; h < NH; h++)
      {
        const float *query_t = inp + b * T * C3 + t * C3 + h * hs;
        float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;

        // pass 1: calculate query dot key and maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= t; t2++)
        {
          const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

          // (query_t) dot (key_t2)
          float val = 0.0f;
          for (int i = 0; i < hs; i++)
          {
            val += query_t[i] * key_t2[i];
          }
          val *= scale;
          if (val > maxval)
          {
            maxval = val;
          }

          preatt_bth[t2] = val;
        }
        // pad with -INFINITY outside of autoregressive region for debugging comparisons
        for (int t2 = t + 1; t2 < T; t2++)
        {
          preatt_bth[t2] = -INFINITY;
        }

        // pass 2: calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++)
        {
          float expv = expf(preatt_bth[t2] - maxval);
          expsum += expv;
          att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++)
        {
          if (t2 <= t)
          {
            att_bth[t2] *= expsum_inv;
          }
          else
          {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
          }
        }

        // pass 4: accumulate weighted values into the output of attention
        float *out_bth = out + b * T * C + t * C + h * hs;
        for (int i = 0; i < hs; i++)
        {
          out_bth[i] = 0.0f;
        }
        for (int t2 = 0; t2 <= t; t2++)
        {
          const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
          float att_btht2 = att_bth[t2];
          for (int i = 0; i < hs; i++)
          {
            out_bth[i] += att_btht2 * value_t2[i];
          }
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void attention_query_key_kernel1(float *preatt, const float *inp,
                                            int B, int T, int C, int NH)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = B * NH * T * T;

  if (idx < total_threads)
  {
    int t2 = idx % T;
    int t = (idx / T) % T;
    if (t2 > t)
    {
      // autoregressive mask
      preatt[idx] = -INFINITY;
      return;
    }
    int h = (idx / (T * T)) % NH;
    int b = idx / (NH * T * T);

    int C3 = C * 3;
    int hs = C / NH; // head size
    const float *query_t = inp + b * T * C3 + t * C3 + h * hs;
    const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

    // (query_t) dot (key_t2)
    float val = 0.0f;
    for (int i = 0; i < hs; i++)
    {
      val += query_t[i] * key_t2[i];
    }
    val *= 1.0 / sqrtf(hs);

    preatt[idx] = val;
  }
}

__global__ void attention_softmax_kernel1(float *att, const float *preatt,
                                          int B, int T, int NH)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = B * T * NH;

  if (idx < total_threads)
  {
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    const float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
    float *att_bth = att + b * NH * T * T + h * T * T + t * T;

    // find maxval
    float maxval = -10000.0f; // TODO something better
    for (int t2 = 0; t2 <= t; t2++)
    {
      if (preatt_bth[t2] > maxval)
      {
        maxval = preatt_bth[t2];
      }
    }

    // calculate the exp and keep track of sum
    float expsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++)
    {
      float expv = expf(preatt_bth[t2] - maxval);
      expsum += expv;
      att_bth[t2] = expv;
    }
    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // normalize to get the softmax
    for (int t2 = 0; t2 < T; t2++)
    {
      if (t2 <= t)
      {
        att_bth[t2] *= expsum_inv;
      }
      else
      {
        // causal attention mask. not strictly necessary to set to zero here
        // only doing this explicitly for debugging and checking to PyTorch
        att_bth[t2] = 0.0f;
      }
    }
  }
}

__global__ void attention_value_kernel1(float *out, const float *att, const float *inp,
                                        int B, int T, int C, int NH)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = B * T * NH;

  if (idx < total_threads)
  {
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    int C3 = C * 3;
    int hs = C / NH; // head size

    float *out_bth = out + b * T * C + t * C + h * hs;
    const float *att_bth = att + b * NH * T * T + h * T * T + t * T;

    for (int i = 0; i < hs; i++)
    {
      out_bth[i] = 0.0f;
    }
    for (int t2 = 0; t2 <= t; t2++)
    {
      const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
      float att_btht2 = att_bth[t2];
      for (int i = 0; i < hs; i++)
      {
        out_bth[i] += att_btht2 * value_t2[i];
      }
    }
  }
}

template <class QKLayout, class PreattLayout, class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class MSmemLayout>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void attention_query_key_kernel2(QKLayout qk_layout, PreattLayout preatt_layout, ProblemShape shape_MNK, CtaTiler cta_tiler,
                                                                                                        TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                                                                                                        TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                                                                                                        TC *C, CStride dC, CSmemLayout sS_layout, TiledMma mma, int KEY_BLOCK, float softmax_scale,
                                                                                                        MSmemLayout sM_layout)
{
  using namespace cute;

  Tensor mQ = make_tensor(make_gmem_ptr(A), qk_layout);     // (B, NH, T, HS)
  Tensor mK = make_tensor(make_gmem_ptr(B), qk_layout);     // (B, NH, T, HS)
  Tensor mP = make_tensor(make_gmem_ptr(C), preatt_layout); // (B, NH, T, T)

  //   Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (T, HS)
  //   Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (T, HS)
  //   Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (T, T)

  // Get the appropriate blocks for this thread block
  // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (blk_Q,blk_K,hs) <- wrong
  auto batch = blockIdx.x;
  auto head = blockIdx.y;
  auto query_block = blockIdx.z;

  auto query_coord = make_coord(query_block, _, _);                                         // (blk_Q,blk_K,hs)
  Tensor gA = local_tile(mQ(batch, head, _, _), cta_tiler, query_coord, Step<_1, X, _1>{}); // (BLK_Q,HS, hs_tile) hs_tile is set to be only one in this setup
                                                                                            // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_Q,HS)
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_Q,CPY_HEAD_DIM,hs_tile)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_Q,CPY_HEAD_DIM)
  copy(copy_a, tAgA(_, _, _, 0), tAsA);

  __shared__ TA smemM[cosize_v<MSmemLayout>];
  Tensor sM = make_tensor(make_smem_ptr(smemM), sM_layout); // (BLK_K)

  for (int key_block = 0; key_block < KEY_BLOCK; ++key_block)
  {
    __syncthreads();
    auto cta_coord = make_coord(query_block, key_block, _); // (blk_Q,blk_K,hs)

    // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    Tensor gB = local_tile(mK(batch, head, _, _), cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_K,HS, hs_tile)
    Tensor gC = local_tile(mP(batch, head, _, _), cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_Q,BLK_K)

    __shared__ TB smemB[cosize_v<BSmemLayout>];
    __shared__ TA smemS[cosize_v<CSmemLayout>];

    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_K,HS)
    Tensor sS = make_tensor(make_smem_ptr(smemS), sS_layout); // (BLK_Q, BLK_K)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_K,CPY_HEAD_DIM,hs_tile)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_K,CPY_HEAD_DIM)

    // Copy gmem to rmem for hs_tile=0, remember we only need to do this once because there is only 1 tile for headim
    copy(copy_b, tBgB(_, _, _, 0), tBsB);

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_Q,MMA_HEAD_DIM)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_K,MMA_HEAD_DIM)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_Q,MMA_K)
    Tensor tCsS = thr_mma.partition_C(sS); // (MMA,MMA_Q,MMA_K)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_Q,MMA_K)
    // Clear the accumulators
    clear(tCrC);
    __syncthreads();
    // int aa = size<0>(tCgC);
    // int bb = size<1>(tCgC);
    // int cc = size<2>(tCgC);
    // printf("%d, %d, %d\n", aa, bb, cc);

    if (query_block < key_block)
    {
      for (int x = 0; x < size<0>(tCsS); ++x)
      {
        for (int y = 0; y < size<1>(tCsS); ++y)
        {
          for (int z = 0; z < size<2>(tCsS); ++z)
          {
            tCsS(x, y, z) = -INFINITY;
          }
        }
      }
      copy(tCsS, tCgC);
    }
    else if (query_block == key_block)
    {
      // We know that the diagonal is all -INFINITY
      gemm(mma, tCsA, tCsB, tCrC);
      axpby(softmax_scale, tCrC, 0, tCgC);
      // int aa = size<0>(tCgC);
      // int bb = size<1>(tCgC);
      // int cc = size<2>(tCgC);
      // printf("%d, %d, %d \n", aa, bb, cc);
      // int aa = size<0>(gC);
      // int bb = size<1>(gC);
      // // int cc = size<2>(gC);
      // printf("%d, %d \n", aa, bb);

      for (int x = 0; x < size<0>(gC); ++x)
      {
        for (int y = 0; y < size<1>(gC); ++y)
        {
          if (x < y)
          {
            gC(x, y) = -INFINITY;
          }
        }
      }
    }
    else
    {
      gemm(mma, tCsA, tCsB, tCrC);
      axpby(softmax_scale, tCrC, 0, tCgC);
    }

    // SM (BLK_K) = max(sM(i), sS(i, _))
    float maxval = -FLT_MAX;
    for (int i = 0; i < size<0>(sM); ++i)
    {
      maxval = fmax(maxval, sS(i, threadIdx.x % size<1>(sS)));
      sM(i) = maxval;
    }
    // float maxval = -FLT_MAX;
    // maxval = fmaxf(maxval, );
  }
}

// TA, TB, TC = the type of variable we are dealing with
void attention_forward2(float *out, float *preatt, float *att,
                        const float *inp,
                        int B, int T, int C, int NH,
                        const int block_size)
// void MatMulCaller(float *A, float *B, float *C, int B, int T, int C, int NH, int N, int K)
{
  using namespace cute;

  const int HS = C / NH;
  // preatt layout
  Layout preatt_layout = make_layout(make_shape(B, NH, T, T), make_stride(T * T * NH, T * T, T, 1));
  Layout qk_layout = make_layout(make_shape(B, NH, T, HS), make_stride(T * C * 3, HS, C * 3, 1));

  auto prob_shape = make_shape(T, T, HS); // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(HS, Int<1>{}); // (dQ, dHs)
  auto dB = make_stride(HS, Int<1>{}); // (dK, dHs)
  auto dC = make_stride(Int<1>{}, T);  // (dQ, dK)

  // Define CTA tile sizes (static), those are the block sizes
  auto bQ = Int<32>{};
  auto bK = Int<32>{};
  auto bHs = Int<64>{};
  auto cta_tiler = make_shape(bQ, bK, bHs); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bQ, bHs),
                        make_stride(Int<1>{}, bQ + Int<1>{})); // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape(bK, bHs),
                        make_stride(Int<1>{}, bK + Int<1>{})); // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bQ, bK));                   // (m,n) -> smem_idx

  auto sM = make_layout(make_shape(bK), make_stride(Int<1>{}));

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                    Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape<_1, _1>>{});                 // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                    Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape<_1, _1>>{});                 // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                                 Layout<Shape<_2, _2, _1>>{}); // 16x16x1 TiledMMA

  // TiledMMA mmaC = make_tiled_mma(UniversalFMA<float, float, float>{},
  //                                Layout<Shape<_16, _16, _1>>{}); // 16x16x1 TiledMMA

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(B, NH, size(ceil_div(T, bQ)));
  int KEY_BLOCK = ceil_div(T, bK);
  cudaStream_t stream = 0;
  const int d = C / NH;
  const float softmax_scale = 1.0 / sqrt(d);
  attention_query_key_kernel2<<<dimGrid, dimBlock, 0, stream>>>(qk_layout, preatt_layout, prob_shape, cta_tiler,
                                                                inp, dA, sA, copyA,
                                                                inp + C, dB, sB, copyB,
                                                                preatt, dC, sC, mmaC, KEY_BLOCK, softmax_scale,
                                                                sM);
}

// ----------------------------------------------------------------------------
// kernel launcher

void attention_forward1(float *out, float *preatt, float *att,
                        const float *inp,
                        int B, int T, int C, int NH,
                        const int block_size)
{
  // attention calculation
  int total_threads = B * NH * T * T;
  int num_blocks = ceil_div(total_threads, block_size);
  attention_query_key_kernel1<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH);
  // softmax and value accumulation
  // total_threads = B * T * NH;
  // num_blocks = ceil_div(total_threads, block_size);
  // attention_softmax_kernel1<<<num_blocks, block_size>>>(att, preatt, B, T, NH);
  // attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH);
}

// kernel version dispatch
void attention_forward(int kernel_num,
                       float *out, float *vaccum, float *qkvr, float *preatt, float *att,
                       const float *inp,
                       int B, int T, int C, int NH,
                       const int block_size)
{

  attention_forward2(out, preatt, att, inp, B, T, C, NH, block_size);
}
// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  srand(0);

  int B = 8;
  int T = 1024;
  int C = 768;
  int NH = 12;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cublasCreate(&cublas_handle);

  // create host memory of random numbers
  float *out = (float *)malloc(B * T * C * sizeof(float));
  float *preatt = (float *)malloc(B * NH * T * T * sizeof(float));
  float *att = (float *)malloc(B * NH * T * T * sizeof(float));
  float *inp = make_random_float(B * T * 3 * C);

  // move to GPU
  float *d_out;
  float *d_vaccum;
  float *d_qkvr;
  float *d_preatt;
  float *d_att;
  float *d_inp;
  cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
  cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
  cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
  cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

  // read kernel_num from command line
  int kernel_num = 1;
  if (argc > 1)
  {
    kernel_num = atoi(argv[1]);
  }
  printf("Using kernel %d\n", kernel_num);
  int block_sizes[] = {32, 64, 128, 256, 512};

  // first check the correctness of the kernel
  attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
  {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);
    attention_forward(kernel_num, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
    validate_result(d_preatt, preatt, "preatt", B * NH * T * T, 1e-2f);
    // all kernels should produce the correct output out
    // validate_result(d_out, out, "out", B * T * C, 1e-4f);
    // // but as for preatt and att, things get a bit more complicated:
    // if (kernel_num != 2) {
    //     // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
    //     // that estimates the softmax online and never materializes preatt/att
    //     validate_result(d_att, att, "att", B * NH * T * T, 1e-4f);
    // }
    // if (kernel_num != 2 && kernel_num != 4 && kernel_num != 5) {
    //     // kernel 4 (knowingly) fails preatt because it fuses the scale normalization
    //     // into the softmax, so preatt is off by 1.0f / sqrt(HS)
    //     // but att and out (checked below) should match.
    //     validate_result(d_preatt, preatt, "preatt", B * NH * T * T, 1e-4f);
    // }
  }
  printf("All results match. Starting benchmarks.\n\n");

  // benchmark speed of the kernel
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
  {
    int block_size = block_sizes[j];
    int repeat_times = 100;

    float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
                                          kernel_num, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp,
                                          B, T, C, NH, block_size);

    printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
  }

  // free memory
  free(out);
  free(preatt);
  free(att);
  free(inp);
  cudaCheck(cudaFree(d_out));
  cudaCheck(cudaFree(d_vaccum));
  cudaCheck(cudaFree(d_qkvr));
  cudaCheck(cudaFree(d_preatt));
  cudaCheck(cudaFree(d_att));
  cudaCheck(cudaFree(d_inp));
  cublasDestroy(cublas_handle);

  return 0;
}