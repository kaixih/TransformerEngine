#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "common/util/vectorized_pointwise.h"
#include "common/gemm/cublaslt_gemm.h"
#include "pybind11/pybind11.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
using bfloat16 = nv_bfloat16;

namespace detail {

struct Empty {};

__device__ inline float identity(float value, const Empty&) {
  return value;
}

}  // namespace detail

void CheckTensorIsOnGPU(TFE_TensorHandle* tensor) {
  TF_Status* status = TF_NewStatus();
  const char* device_type = TFE_TensorHandleDeviceType(tensor, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CHECK_EQ(std::string(device_type), std::string("GPU"))
    << "Tensor must be on the GPU, but got device_type=" << device_type;
  TF_DeleteStatus(status);
}

std::vector<int64_t> TensorShapeAsVector(TFE_TensorHandle* tensor,
                                         TF_Status* status) {
  std::vector<int64_t> shape(TFE_TensorHandleNumDims(tensor, status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(tensor, i, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  return shape;
}

PYBIND11_MODULE(_pywrap_transformer_engine, m) {
  m.attr("FP8_E4M3") = py::int_(static_cast<int>(CUDA_R_8F_E4M3));
  m.attr("FP8_E5M2") = py::int_(static_cast<int>(CUDA_R_8F_E5M2));
  m.def("cast_to_fp8", [](const pybind11::handle& input,
                          const pybind11::handle& scale,
                          const pybind11::handle& output,
                          const pybind11::handle& scale_inv,
                          const pybind11::handle& amax,
                          const int output_dtype) {
    // Get eager tensors.
    CHECK(EagerTensor_CheckExact(input.ptr())) << "Input tensor must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(scale.ptr())) << "Scale tensor must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(output.ptr())) << "Output tensor must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(scale_inv.ptr())) << "Scale_inv tensor must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(amax.ptr())) << "amax tensor must be an EagerTensor.";
    TFE_TensorHandle* input_tensor = EagerTensor_Handle(input.ptr());
    TFE_TensorHandle* scale_tensor = EagerTensor_Handle(scale.ptr());
    TFE_TensorHandle* output_tensor = EagerTensor_Handle(output.ptr());
    TFE_TensorHandle* scale_inv_tensor = EagerTensor_Handle(scale_inv.ptr());
    TFE_TensorHandle* amax_tensor = EagerTensor_Handle(amax.ptr());

    // Check types.
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(input_tensor)) << "Input tensor must have type float32.";
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(scale_tensor)) << "Scale tensor must have type float32.";
    CHECK_EQ(TF_INT8, TFE_TensorHandleDataType(output_tensor)) << "Output tensor must have type int8.";
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(scale_inv_tensor)) << "Scale_inv tensor must have type float32.";
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(amax_tensor)) << "Amax tensor must have type float32.";
    auto dtype = static_cast<cudaDataType_t>(output_dtype);
    CHECK(dtype == CUDA_R_8F_E4M3 || dtype ==CUDA_R_8F_E5M2) << "output_dtype must be \"FP8_E4M3\" or \"FP8_E5M2\".";

    // Check device placement.
    CheckTensorIsOnGPU(input_tensor);
    CheckTensorIsOnGPU(scale_tensor);
    CheckTensorIsOnGPU(output_tensor);
    CheckTensorIsOnGPU(scale_inv_tensor);
    CheckTensorIsOnGPU(amax_tensor);

    // Get and check num elements.
    TF_Status* status = TF_NewStatus();
    int64_t num_elements = TFE_TensorHandleNumElements(input_tensor, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    int64_t scale_num_elements = TFE_TensorHandleNumElements(scale_tensor, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    CHECK_EQ(scale_num_elements, 1) << "Scale must be a scalar.";
    int64_t output_num_elements = TFE_TensorHandleNumElements(output_tensor, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    CHECK_EQ(num_elements, output_num_elements) << "Input and output must have same number of elements.";
    int64_t scale_inv_num_elements = TFE_TensorHandleNumElements(scale_inv_tensor, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    CHECK_EQ(scale_inv_num_elements, 1) << "Scale_inv must be a scalar.";
    int64_t amax_num_elements = TFE_TensorHandleNumElements(amax_tensor, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    CHECK_EQ(amax_num_elements, 1) << "Amax must be a scalar.";

    // Get device pointers
    float* input_data = static_cast<float*>(TFE_TensorHandleDevicePointer(input_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float* scale_data = static_cast<float*>(TFE_TensorHandleDevicePointer(scale_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    fp8e4m3* output_data = static_cast<fp8e4m3*>(TFE_TensorHandleDevicePointer(output_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float* scale_inv_data = static_cast<float*>(TFE_TensorHandleDevicePointer(scale_inv_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float* amax_data = static_cast<float*>(TFE_TensorHandleDevicePointer(amax_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TF_DeleteStatus(status);

    constexpr int nvec = 32 / sizeof(float);
    cudaStream_t stream = 0;
    if (dtype == CUDA_R_8F_E4M3) {
      transformer_engine::VectorizedUnaryKernelLauncher<
          nvec, detail::Empty, detail::identity>(input_data, output_data, scale_data, scale_inv_data,
                                                 amax_data, num_elements, {}, stream);
    } else {
      fp8e5m2* output_cast = reinterpret_cast<fp8e5m2*>(output_data);
      transformer_engine::VectorizedUnaryKernelLauncher<
          nvec, detail::Empty, detail::identity>(
              input_data, output_cast, scale_data, scale_inv_data, amax_data, num_elements, {}, stream);
    }
  });
  m.def("fp8_gemm", [](const pybind11::handle& a_handle,
                       const pybind11::handle& a_scale_inv_handle,
                       const int a_dtype_in,
                       const pybind11::handle& b_handle,
                       const pybind11::handle& b_scale_inv_handle,
                       const int b_dtype_in,
                       const pybind11::handle& d_handle,
                       const bool use_bias,
                       const pybind11::handle& bias_handle,
                       const bool transa, const bool transb, const bool grad,
                       const bool accumulate, const bool use_split_accumulate) {
    // Get eager tensors.
    CHECK(EagerTensor_CheckExact(a_handle.ptr())) << "Input a must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(a_scale_inv_handle.ptr())) << "Input a_scale_inv must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(b_handle.ptr())) << "Input n must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(b_scale_inv_handle.ptr())) << "Input b_scale_inv tensor must be an EagerTensor.";
    CHECK(EagerTensor_CheckExact(d_handle.ptr())) << "Output d must be an EagerTensor.";
    
    TFE_TensorHandle* a_tensor = EagerTensor_Handle(a_handle.ptr());
    TFE_TensorHandle* a_scale_inv_tensor = EagerTensor_Handle(a_scale_inv_handle.ptr());
    TFE_TensorHandle* b_tensor = EagerTensor_Handle(b_handle.ptr());
    TFE_TensorHandle* b_scale_inv_tensor = EagerTensor_Handle(b_scale_inv_handle.ptr());
    TFE_TensorHandle* d_tensor = EagerTensor_Handle(d_handle.ptr());
    TFE_TensorHandle* bias_tensor;
    

    // Check types.
    CHECK_EQ(TF_INT8, TFE_TensorHandleDataType(a_tensor)) << "Input a must have type int8.";
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(a_scale_inv_tensor)) << "Input a_scale_inv must have type float32.";
    CHECK_EQ(TF_INT8, TFE_TensorHandleDataType(b_tensor)) << "Input b must have type int8.";
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(b_scale_inv_tensor)) << "Input b_scale_inv  must have type float32.";
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(d_tensor)) << "Output d must have type float32.";
    auto a_dtype = static_cast<cudaDataType_t>(a_dtype_in);
    auto b_dtype = static_cast<cudaDataType_t>(b_dtype_in);
    CHECK(a_dtype == CUDA_R_8F_E4M3 || a_dtype == CUDA_R_8F_E5M2) << "output_dtype must be \"FP8_E4M3\" or \"FP8_E5M2\".";
    CHECK(b_dtype == CUDA_R_8F_E4M3 || b_dtype == CUDA_R_8F_E5M2) << "output_dtype must be \"FP8_E4M3\" or \"FP8_E5M2\".";

    // Check device placement.
    CheckTensorIsOnGPU(a_tensor);
    CheckTensorIsOnGPU(a_scale_inv_tensor);
    CheckTensorIsOnGPU(b_tensor);
    CheckTensorIsOnGPU(b_scale_inv_tensor);
    CheckTensorIsOnGPU(d_tensor);

    if (use_bias) {
      CHECK(EagerTensor_CheckExact(bias_handle.ptr())) << "Bias must be an EagerTensor.";
      bias_tensor = EagerTensor_Handle(bias_handle.ptr());
      CHECK_EQ(TF_BFLOAT16, TFE_TensorHandleDataType(bias_tensor)) << "Bias must have type bfloat16.";
      CheckTensorIsOnGPU(d_tensor);
    }

    // Get dimensions.
    TF_Status* status = TF_NewStatus();
    auto a_shape = TensorShapeAsVector(a_tensor, status);
    auto b_shape = TensorShapeAsVector(b_tensor, status);
    auto d_shape = TensorShapeAsVector(d_tensor, status);
    CHECK_EQ(a_shape.size(), 2) << "Input a must be rank 2.";
    CHECK_EQ(b_shape.size(), 2) << "Input b must be rank 2.";
    CHECK_EQ(d_shape.size(), 2) << "Output d must be rank 2.";
    const int m = transa ? a_shape[0] : a_shape[1];
    const int k = transa ? a_shape[1] : a_shape[0];
    const int n = transb ? b_shape[1] : b_shape[0];

    // Get device pointers.
    fp8e4m3* a_data = static_cast<fp8e4m3*>(TFE_TensorHandleDevicePointer(a_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float* a_scale_inv_data = static_cast<float*>(TFE_TensorHandleDevicePointer(a_scale_inv_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    fp8e4m3* b_data = static_cast<fp8e4m3*>(TFE_TensorHandleDevicePointer(b_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float* b_scale_inv_data = static_cast<float*>(TFE_TensorHandleDevicePointer(b_scale_inv_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float* d_data = static_cast<float*>(TFE_TensorHandleDevicePointer(d_tensor, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    bfloat16* bias_data = nullptr;
    if (use_bias) {
      bias_data = static_cast<bfloat16*>(TFE_TensorHandleDevicePointer(bias_tensor, status));
    }
    TF_DeleteStatus(status);

    int workspaceSize = 33'554'432;
    void* workspace = nullptr;
    checkCUDA(cudaMalloc((void**)&workspace, workspaceSize));

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
      LOG(FATAL) << "TT layout not allowed.";
    }

    auto D_type = CUDA_R_32F;
    auto bias_type = CUDA_R_16BF;
    cudaStream_t stream = 0;
    transformer_engine::cublas_gemm(a_data,
                                    a_scale_inv_data,
                                    b_data,
                                    b_scale_inv_data,
                                    d_data,
                                    /*bias_ptr=*/bias_data,
                                    /*pre_gelu_out=*/nullptr,
                                    m, n, k,
                                    lda, ldb, ldd,
                                    a_dtype,
                                    b_dtype,
                                    D_type,
                                    bias_type,
                                    (transa) ? CUBLAS_OP_T : CUBLAS_OP_N,
                                    (transb) ? CUBLAS_OP_T : CUBLAS_OP_N,
                                    /*bias=*/bias_data != nullptr,
                                    /*gelu=*/false,
                                    /*grad=*/false,
                                    workspace,
                                    workspaceSize,
                                    /*use_fp8=*/true,
                                    accumulate,
                                    use_split_accumulate,
                                    stream);

    checkCUDA(cudaFree(workspace));
  });
}
