#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "common/util/vectorized_pointwise.h"
#include "common/gemm/cublaslt_gemm.h"

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

using byte = uint8_t;
using int32 = int32_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

namespace detail {

struct Empty {};

__device__ inline fp32 identity(fp32 value, const Empty&) {
  return value;
}

}  // namespace detail

extern "C" {

void cast_to_fp8(const void* input_ptr, const void* scale_ptr, void *amax_ptr,
                 void *scale_inv_ptr, void* output_ptr, size_t dim0,
                 size_t dim1, int dtype) {

  const size_t N = dim0 * dim1;
  using IType = float;
  using OType = fp8e4m3;
  IType* input = nullptr;
  OType* output = nullptr;
  fp32* scale = nullptr;
  fp32* scale_inv = nullptr;
  fp32* amax = nullptr;
  checkCUDA(cudaMalloc((void**)&input, N * sizeof(IType)));
  checkCUDA(cudaMalloc((void**)&output, N * sizeof(OType)));
  checkCUDA(cudaMalloc((void**)&scale, 1 * sizeof(fp32)));
  checkCUDA(cudaMalloc((void**)&scale_inv, 1 * sizeof(fp32)));
  checkCUDA(cudaMalloc((void**)&amax, 1 * sizeof(fp32)));
  checkCUDA(cudaMemcpy(input, input_ptr, N * sizeof(IType),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(scale, scale_ptr, 1 * sizeof(fp32),
                       cudaMemcpyHostToDevice));


  constexpr int nvec = 32 / sizeof(IType);
  cudaStream_t stream = 0;
  if (dtype == 0) {
    transformer_engine::VectorizedUnaryKernelLauncher<
        nvec, detail::Empty, detail::identity>(input, output, scale, scale_inv,
                                               amax, N, {}, stream);
  } else {

    fp8e5m2* output_cast = reinterpret_cast<fp8e5m2*>(output);
    transformer_engine::VectorizedUnaryKernelLauncher<
        nvec, detail::Empty, detail::identity>(
            input, output_cast, scale, scale_inv, amax, N, {}, stream);
  }

  checkCUDA(cudaMemcpy(output_ptr, output, N * sizeof(OType),
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(scale_inv_ptr, scale_inv, 1 * sizeof(fp32),
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(amax_ptr, amax, 1 * sizeof(fp32),
                       cudaMemcpyDeviceToHost));
}

void fp8_gemm(const void* A_ptr,
              const void* A_scale_inverse_ptr,
              int A_dtype,
              const void* B_ptr,
              const void* B_scale_inverse_ptr,
              int B_dtype,
              void* D_ptr,
              int A_dim0, int A_dim1,
              int B_dim0, int B_dim1,
              bool transa, bool transb, bool grad,
              bool accumulate, bool use_split_accumulate) {

  const int m = transa ? A_dim0 : A_dim1;
  const int k = transa ? A_dim1 : A_dim0;
  const int n = transb ? B_dim1 : B_dim0;

  const size_t A_N = m * k;
  const size_t B_N = k * n;
  const size_t D_N = m * n;
  using AType = fp8e4m3;
  using DType = float;
  AType* A = nullptr;
  AType* B = nullptr;
  fp32* D = nullptr;
  fp32* A_scale_inv = nullptr;
  fp32* B_scale_inv = nullptr;
  fp32* bias_ptr = nullptr;
  int workspaceSize = 33'554'432;
  void* workspace = nullptr;
  checkCUDA(cudaMalloc((void**)&A, A_N * sizeof(AType)));
  checkCUDA(cudaMalloc((void**)&B, B_N * sizeof(AType)));
  checkCUDA(cudaMalloc((void**)&D, D_N * sizeof(DType)));
  checkCUDA(cudaMalloc((void**)&A_scale_inv, 1 * sizeof(fp32)));
  checkCUDA(cudaMalloc((void**)&B_scale_inv, 1 * sizeof(fp32)));
  // checkCUDA(cudaMalloc((void**)&bias_ptr, n * sizeof(bf16)));
  checkCUDA(cudaMalloc((void**)&workspace, workspaceSize));
  checkCUDA(cudaMemcpy(A, A_ptr, A_N * sizeof(AType),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(B, B_ptr, B_N * sizeof(AType),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(A_scale_inv, A_scale_inverse_ptr, 1 * sizeof(fp32),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(B_scale_inv, B_scale_inverse_ptr, 1 * sizeof(fp32),
                       cudaMemcpyHostToDevice));


  int lda, ldb, ldd;
  if (transa && !transb) {  // TN
    lda = k;
    ldb = k;
    ldd = m;
  } else if (!transa && !transb) {  // NN
    lda = m;
    ldb = k;
    ldd = m;
  } else if (!transa && transb) {  // NT
    lda = m;
    ldb = n;
    ldd = m;
  } else {  // TT
    printf("TT layout not allowed.\n");
    exit(0);
  }

  auto A_type = CUDA_R_8F_E4M3;
  auto B_type = CUDA_R_8F_E4M3;
  if (A_dtype != 0 ) A_type = CUDA_R_8F_E5M2;
  if (B_dtype != 0 ) B_type = CUDA_R_8F_E5M2;

  auto D_type = CUDA_R_32F;
  auto bias_type = CUDA_R_16BF;
  cudaStream_t stream = 0;

  transformer_engine::cublas_gemm(A,
                                  A_scale_inv,
                                  B,
                                  B_scale_inv,
                                  D,
                                  /*bias_ptr=*/bias_ptr,
                                  /*pre_gelu_out=*/nullptr,
                                  m, n, k,
                                  lda, ldb, ldd,
                                  A_type,
                                  B_type,
                                  D_type,
                                  bias_type,
                                  (transa) ? CUBLAS_OP_T : CUBLAS_OP_N,
                                  (transb) ? CUBLAS_OP_T : CUBLAS_OP_N,
                                  /*bias=*/bias_ptr != nullptr,
                                  /*gelu=*/false,
                                  /*grad=*/false,
                                  workspace,
                                  workspaceSize,
                                  /*use_fp8=*/true,
                                  accumulate,
                                  use_split_accumulate,
                                  stream);

  checkCUDA(cudaMemcpy(D_ptr, D, D_N * sizeof(DType),
                       cudaMemcpyDeviceToHost));

  checkCUDA(cudaFree(A));
  checkCUDA(cudaFree(B));
  checkCUDA(cudaFree(D));
  checkCUDA(cudaFree(A_scale_inv));
  checkCUDA(cudaFree(B_scale_inv));
  // checkCUDA(cudaFree(bias_ptr));
  checkCUDA(cudaFree(workspace));
  
}

}

