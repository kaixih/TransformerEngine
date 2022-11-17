#include "../common.h"
#include "../logging.h"
#include "../transformer_engine.h"
#include <cublasLt.h>

namespace transformer_engine {

void cublas_gemm(void* A,
                 void* A_scale_inverse,
                 void* B,
                 void *B_scale_inverse,
                 void* D,
                 void* bias_ptr,
                 void* pre_gelu_out,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 cudaDataType_t A_type,
                 cudaDataType_t B_type,
                 cudaDataType_t D_type,
                 cudaDataType_t bias_type,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 bool bias,
                 bool gelu,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool use_fp8,
                 bool accumulate,
                 bool use_split_accumulator,
                 cudaStream_t stream
) {
    // check consistency of arguments:
    // if fp8 is desired, context cannot be null
    // fp8 + gelu fusion is unavailable right now.
    if (use_fp8) {
      NVTE_CHECK(!gelu, "fp8 gemm + gelu fusion is unavailable right now!");
    }
    printf("XXX inside cublas_gemm\n");

    float one = 1.0;
    float zero = 0.0;
    float beta = (accumulate) ? one : zero;

    cublasLtHandle_t handle;
    NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

    cublasLtMatmulDesc_t       operationDesc = nullptr;
    cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    int                             returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

    int64_t ld_gelumat = (int64_t) ldd;

    // default to tf32 except for e5m2 inputs where the config is not supported
    cublasComputeType_t gemm_compute_type = (A_type == CUDA_R_8F_E5M2 || B_type == CUDA_R_8F_E5M2)
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_32F_FAST_TF32;

    // Create matrix descriptors. Not setting any extra attributes.
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type,
                                                 transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m,
                                                 lda));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type,
                                                 transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k,
                                                 ldb));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

    NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                     &transa, sizeof(transa)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                     &transb, sizeof(transb)));

    // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
    // Note: gelu fusion isn't available right now, and we don't need
    // amax(D) either (next op is high precision).
    if (use_fp8) {
        // Split accumulator.
        const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                         &fastAccuMode,
                                                         sizeof(fastAccuMode)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                         &A_scale_inverse,
                                                         sizeof(A_scale_inverse)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                         &B_scale_inverse,
                                                         sizeof(B_scale_inverse)));
        if (bias) {
            NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                             CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                             &bias_type, sizeof(bias_type)));
        }
    }
    printf("XXX inside cublas_gemm after fp8 branch\n");

    if (bias && gelu) {
        if (grad) {
            epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
        } else {
            epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
        }
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                         &bias_ptr, sizeof(bias_ptr)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                                operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                &pre_gelu_out, sizeof(pre_gelu_out)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                         &ld_gelumat, sizeof(ld_gelumat)));
    } else if (bias) {
        if (grad) {
            // grad output is always input B
            epilogue = CUBLASLT_EPILOGUE_BGRADB;
        } else {
            epilogue = CUBLASLT_EPILOGUE_BIAS;
        }
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                         &bias_ptr, sizeof(bias_ptr)));
    } else if (gelu) {
        if (grad) {
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
        }
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                                operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                &pre_gelu_out, sizeof(pre_gelu_out)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                         &ld_gelumat, sizeof(ld_gelumat)));
    }
    printf("XXX inside cublas_gemm after gelu bias branch\n");

    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue, sizeof(epilogue)));
    printf("XXX inside cublas_gemm after set attr\n");

    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
                            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            &workspaceSize, sizeof(workspaceSize)));
    printf("XXX inside cublas_gemm after preference\n");

    NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Ddesc,
                                                     Ddesc, preference, 1, &heuristicResult,
                                                     &returnedResults));
    printf("XXX inside cublas_gemm after heur\n");

    if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");
    printf("XXX inside cublas_gemm returnedResults \n");

    // D = alpha * (A * B) + beta * C
    NVTE_CHECK_CUBLAS(cublasLtMatmul(handle,
                                     operationDesc,
                                     static_cast<const void*>(&one),         /* alpha */
                                     A,                                      /* A */
                                     Adesc,
                                     B,                                      /* B */
                                     Bdesc,
                                     static_cast<const void*>(&beta),        /* beta */
                                     D,                                      /* C */
                                     Ddesc,
                                     D,                                      /* D */
                                     Ddesc,
                                     &heuristicResult.algo,                  /* algo */
                                     workspace,                              /* workspace */
                                     workspaceSize,
                                     stream));                               /* stream */


    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}

}  // namespace transformer_engine
