#include <string>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/transpose.h>
#include <transformer_engine/gemm.h>
#include "pybind11/pybind11.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace transformer_engine {

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
    GEMM1_INPUT  = 0,
    GEMM1_WEIGHT = 1,
    GEMM2_INPUT  = 2,
    GEMM2_WEIGHT = 3
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors {
    GRAD_OUTPUT1 = 0,
    GRAD_OUTPUT2 = 1
};

}  // namespace transformer_engine

void CheckTensorIsOnGPU(TFE_TensorHandle* tensor, TF_Status* status) {
  const char* device_type = TFE_TensorHandleDeviceType(tensor, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CHECK_EQ(std::string(device_type), std::string("GPU"))
    << "Tensor must be on the GPU, but got device_type=" << device_type;
}

std::vector<size_t> TensorShapeAsVector(TFE_TensorHandle* tensor,
                                         TF_Status* status) {
  std::vector<size_t> shape(TFE_TensorHandleNumDims(tensor, status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  // Hande scalars
  // TODO: empty tensor?
  if (shape.size() == 0) {
    return {1};
  }
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(tensor, i, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  return shape;
}

inline transformer_engine::DType GetTransformerEngineDType(TF_DataType t) {
  switch (t) {
    case TF_HALF:
      return transformer_engine::DType::kFloat16;
    case TF_FLOAT:
      return transformer_engine::DType::kFloat32;
    case TF_BFLOAT16:
      return transformer_engine::DType::kBFloat16;
    case TF_INT8:
      return transformer_engine::DType::kByte;
    default:
      CHECK(false) << "TensorFlow dtype is not supported: " << t;
  }
}

inline TF_DataType GetTensorFlowDType(transformer_engine::DType  t) {
  switch (t) {
    case transformer_engine::DType::kFloat32:
      return TF_FLOAT;
    default:
      CHECK(false) << "Transformer Engine dtype is not supported: " << (int)t;
  }
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(TFE_TensorHandle* tensor, transformer_engine::DType dtype) {
  TF_Status* status = TF_NewStatus();
  CheckTensorIsOnGPU(tensor, status);
  void* data_ptr = TFE_TensorHandleDevicePointer(tensor, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  std::vector<size_t> shape = TensorShapeAsVector(tensor, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  return transformer_engine::TensorWrapper(data_ptr, shape, dtype);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(TFE_TensorHandle* tensor) {
  transformer_engine::DType dtype = GetTransformerEngineDType(TFE_TensorHandleDataType(tensor));
  return makeTransformerEngineTensor(tensor, dtype);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(const pybind11::handle tensor) {
  CHECK(EagerTensor_CheckExact(tensor.ptr())) << "All inputs must be EagerTensors.";
  return makeTransformerEngineTensor(EagerTensor_Handle(tensor.ptr()));
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(const pybind11::handle tensor, transformer_engine::DType dtype) {
  CHECK(EagerTensor_CheckExact(tensor.ptr())) << "All inputs must be EagerTensors.";
  return makeTransformerEngineTensor(EagerTensor_Handle(tensor.ptr()), dtype);
}

tensorflow::Allocator* GetAllocator() {
  static tensorflow::Allocator* allocator = nullptr;
  if (allocator == nullptr) {
    tensorflow::GPUOptions gpu_options;
    tsl::TfDeviceId device_id(0);
    allocator = tensorflow::GPUProcessState::singleton()->GetGPUAllocator(
        gpu_options, device_id, /*total_bytes=*/1, /*peer_gpu_ids=*/{});
  }
  return allocator;
}

TFE_Context* GetContext(TF_Status* status) {
  // Cache TF context.
  static TFE_Context* context = nullptr;
  if (context == nullptr) {
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    context = TFE_NewContext(opts, status);
  }
  return context;
}

void Deallocator(void* data, size_t unused, void* tensor_handle) {
  GetAllocator()->DeallocateRaw(data);
}

TFE_TensorHandle* AllocateTensor(std::vector<int64_t> shape, TF_DataType dtype) {
  TF_Status* status = TF_NewStatus();
  TFE_Context* ctx = GetContext(status); 

  // Allocate GPU memory.
  size_t num_bytes = TF_DataTypeSize(dtype);
  for (int i = 0; i < shape.size(); ++i) num_bytes *= shape[i];
  void* data = GetAllocator()->AllocateRaw(
      tensorflow::Allocator::kAllocatorAlignment, num_bytes);

  // Get first GPU device name.
  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_devices = TF_DeviceListCount(devices);
  const char* device_name = nullptr;
  for (int i = 0; i < num_devices; ++i) {
    const char* name = TF_DeviceListName(devices, i, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    if (std::string(name).find("GPU") != std::string::npos) {
      device_name = name;
      break;
    }
  }
  CHECK_NE(device_name, nullptr) << "No GPU device found.";

  TFE_TensorHandle* tensor = TFE_NewTensorHandleFromDeviceMemory(
      ctx, device_name, dtype, shape.data(), shape.size(), data, num_bytes, &Deallocator, nullptr, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  return tensor;
}

PYBIND11_MODULE(_pywrap_transformer_engine, m) {
  py::enum_<transformer_engine::DType>(m, "DType")
    .value("kByte", transformer_engine::DType::kByte)
    .value("kInt32", transformer_engine::DType::kInt32)
    .value("kFloat32", transformer_engine::DType::kFloat32)
    .value("kFloat16", transformer_engine::DType::kFloat16)
    .value("kBFloat16", transformer_engine::DType::kBFloat16)
    .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)
    .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);

  py::enum_<transformer_engine::FP8FwdTensors>(m, "FP8FwdTensors", py::arithmetic())
    .value("GEMM1_INPUT", transformer_engine::FP8FwdTensors::GEMM1_INPUT)
    .value("GEMM1_WEIGHT", transformer_engine::FP8FwdTensors::GEMM1_WEIGHT)
    .value("GEMM2_INPUT", transformer_engine::FP8FwdTensors::GEMM2_INPUT)
    .value("GEMM2_WEIGHT", transformer_engine::FP8FwdTensors::GEMM2_WEIGHT);

  py::enum_<transformer_engine::FP8BwdTensors>(m, "FP8BwdTensors", py::arithmetic())
    .value("GRAD_OUTPUT1", transformer_engine::FP8BwdTensors::GRAD_OUTPUT1)
    .value("GRAD_OUTPUT2", transformer_engine::FP8BwdTensors::GRAD_OUTPUT2);

  m.def("cast_to_fp8", [](const pybind11::handle& input,
                          const pybind11::handle& scale,
                          const pybind11::handle& amax,
                          const transformer_engine::DType output_dtype) {
    // Get NVTE tensors.
    auto input_tensor = makeTransformerEngineTensor(input);
    auto scale_tensor = makeTransformerEngineTensor(scale);
    auto amax_tensor = makeTransformerEngineTensor(amax);

    // Allocate output tensors
    auto in_shape = input_tensor.shape();
    std::vector<int64_t> shape_vec(in_shape.data, in_shape.data + in_shape.ndim);
    TFE_TensorHandle* output_eager = AllocateTensor(shape_vec, TF_INT8);
    auto output_tensor = makeTransformerEngineTensor(output_eager, output_dtype);
    TFE_TensorHandle* scale_inv_eager = AllocateTensor({1}, TF_FLOAT);
    auto scale_inv_tensor = makeTransformerEngineTensor(scale_inv_eager);
    TFE_TensorHandle* amax_out_eager = AllocateTensor({1}, TF_FLOAT);
    auto amax_out_tensor = makeTransformerEngineTensor(amax_out_eager);
    cudaMemcpy(amax_out_tensor.dptr(), amax_tensor.dptr(), sizeof(float), cudaMemcpyDeviceToDevice);

    cudaStream_t stream = 0;
    nvte_fp8_quantize(input_tensor.data(), scale_tensor.data(), output_tensor.data(), amax_out_tensor.data(), scale_inv_tensor.data(), stream);
    PyObject* result(PyList_New(3));
    PyList_SET_ITEM(result, 0, EagerTensorFromHandle(output_eager));
    PyList_SET_ITEM(result, 1, EagerTensorFromHandle(amax_out_eager));
    PyList_SET_ITEM(result, 2, EagerTensorFromHandle(scale_inv_eager));
    return tensorflow::PyoOrThrow(result);
  });
  m.def("fp8_cast_transpose_fused", [](const pybind11::handle& input,
                          const pybind11::handle& scale,
                          const pybind11::handle& amax,
                          const transformer_engine::DType output_dtype) {
    // Get NVTE tensors.
    auto input_tensor = makeTransformerEngineTensor(input);
    auto scale_tensor = makeTransformerEngineTensor(scale);
    auto amax_tensor = makeTransformerEngineTensor(amax);

    // Allocate output tensors
    auto in_shape = input_tensor.shape();
    CHECK_EQ(in_shape.ndim, 2);
    std::vector<int64_t> shape_vec(in_shape.data, in_shape.data + in_shape.ndim);
    TFE_TensorHandle* output_cast_eager = AllocateTensor({shape_vec[0], shape_vec[1]}, TF_INT8);
    auto output_cast_tensor = makeTransformerEngineTensor(output_cast_eager, output_dtype);
    TFE_TensorHandle* output_transpose_eager = AllocateTensor({shape_vec[1], shape_vec[0]}, TF_INT8);
    auto output_transpose_tensor = makeTransformerEngineTensor(output_transpose_eager, output_dtype);
    TFE_TensorHandle* scale_inv_eager = AllocateTensor({1}, TF_FLOAT);
    auto scale_inv_tensor = makeTransformerEngineTensor(scale_inv_eager);
    TFE_TensorHandle* amax_out_eager = AllocateTensor({1}, TF_FLOAT);
    auto amax_out_tensor = makeTransformerEngineTensor(amax_out_eager);
    cudaMemcpy(amax_out_tensor.dptr(), amax_tensor.dptr(), sizeof(float), cudaMemcpyDeviceToDevice);

    cudaStream_t stream = 0;
    nvte_cast_transpose(input_tensor.data(), scale_tensor.data(), output_cast_tensor.data(), output_transpose_tensor.data(), amax_out_tensor.data(), scale_inv_tensor.data(), stream);

    PyObject* result(PyList_New(4));
    PyList_SET_ITEM(result, 0, EagerTensorFromHandle(output_cast_eager));
    PyList_SET_ITEM(result, 1, EagerTensorFromHandle(output_transpose_eager));
    PyList_SET_ITEM(result, 2, EagerTensorFromHandle(amax_out_eager));
    PyList_SET_ITEM(result, 3, EagerTensorFromHandle(scale_inv_eager));
    return tensorflow::PyoOrThrow(result);
  });
  m.def("fp8_cast_transpose_bgrad_fused", [](
      const pybind11::handle& grad_output,
      const pybind11::handle& scale,
      const pybind11::handle& amax,
      const transformer_engine::DType output_dtype) {
    // Get NVTE tensors.
    auto grad_output_tensor = makeTransformerEngineTensor(grad_output);
    auto scale_tensor = makeTransformerEngineTensor(scale);
    auto amax_tensor = makeTransformerEngineTensor(amax);

    // Allocate output tensors
    auto in_shape = grad_output_tensor.shape();
    CHECK_EQ(in_shape.ndim, 2);
    std::vector<int64_t> shape_vec(in_shape.data, in_shape.data + in_shape.ndim);
    TFE_TensorHandle* grad_bias_eager = AllocateTensor({shape_vec[1]}, GetTensorFlowDType(grad_output_tensor.dtype()));
    auto grad_bias = makeTransformerEngineTensor(grad_bias_eager, grad_output_tensor.dtype());
    TFE_TensorHandle* grad_output_cast_eager = AllocateTensor({shape_vec[0], shape_vec[1]}, TF_INT8);
    auto grad_output_cast = makeTransformerEngineTensor(grad_output_cast_eager, output_dtype);
    TFE_TensorHandle* grad_output_transpose_eager = AllocateTensor({shape_vec[1], shape_vec[0]}, TF_INT8);
    auto grad_output_transpose = makeTransformerEngineTensor(grad_output_transpose_eager, output_dtype);
    TFE_TensorHandle* scale_inv_eager = AllocateTensor({1}, TF_FLOAT);
    auto scale_inv_tensor = makeTransformerEngineTensor(scale_inv_eager);
    TFE_TensorHandle* amax_out_eager = AllocateTensor({1}, TF_FLOAT);
    auto amax_out_tensor = makeTransformerEngineTensor(amax_out_eager);
    cudaMemcpy(amax_out_tensor.dptr(), amax_tensor.dptr(), sizeof(float), cudaMemcpyDeviceToDevice);

    cudaStream_t stream = 0;
    transformer_engine::TensorWrapper workspace;

    // First call will populate workspace shape and dtype.
    nvte_cast_transpose_dbias(
        grad_output_tensor.data(), scale_tensor.data(), grad_output_cast.data(),
        grad_output_transpose.data(), amax_out_tensor.data(), grad_bias.data(),
        scale_inv_tensor.data(), workspace.data(), stream);

    // Allocate workspace
    auto workspace_shape = workspace.shape();
    std::vector<int64_t> workspace_shape_vec(workspace_shape.data, workspace_shape.data + workspace_shape.ndim);
    TFE_TensorHandle* workspace_data = AllocateTensor(workspace_shape_vec, GetTensorFlowDType(workspace.dtype()));
    workspace = makeTransformerEngineTensor(workspace_data);
    
    nvte_cast_transpose_dbias(
        grad_output_tensor.data(), scale_tensor.data(), grad_output_cast.data(),
        grad_output_transpose.data(), amax_out_tensor.data(), grad_bias.data(),
        scale_inv_tensor.data(), workspace.data(), stream);
    TFE_DeleteTensorHandle(workspace_data);

    PyObject* result(PyList_New(5));
    PyList_SET_ITEM(result, 0, EagerTensorFromHandle(grad_bias_eager));
    PyList_SET_ITEM(result, 1, EagerTensorFromHandle(grad_output_cast_eager));
    PyList_SET_ITEM(result, 2, EagerTensorFromHandle(grad_output_transpose_eager));
    PyList_SET_ITEM(result, 3, EagerTensorFromHandle(amax_out_eager));
    PyList_SET_ITEM(result, 4, EagerTensorFromHandle(scale_inv_eager));
    return tensorflow::PyoOrThrow(result);
  });
  m.def("fp8_gemm", [](const pybind11::handle& a_handle,
                       const pybind11::handle& a_scale_inv_handle,
                       const transformer_engine::DType a_dtype_in,
                       const pybind11::handle& b_handle,
                       const pybind11::handle& b_scale_inv_handle,
                       const transformer_engine::DType b_dtype_in,
                       const pybind11::handle& workspace_handle,
                       const bool use_bias,
                       const pybind11::handle& bias_handle,
                       const bool transa, const bool transb, const bool grad,
                       const bool accumulate, const bool use_split_accumulate) {
    // Get NVTE tensors.
    auto a = makeTransformerEngineTensor(a_handle, a_dtype_in );
    auto a_scale_inv = makeTransformerEngineTensor(a_scale_inv_handle);
    auto b = makeTransformerEngineTensor(b_handle, b_dtype_in);
    auto b_scale_inv = makeTransformerEngineTensor(b_scale_inv_handle);

    transformer_engine::TensorWrapper bias(nullptr, {static_cast<size_t>(1)}, transformer_engine::DType::kBFloat16);
    if (use_bias) {
      bias = makeTransformerEngineTensor(bias_handle.ptr());
    }
    
    transformer_engine::TensorWrapper pre_gelu_out(nullptr, {static_cast<size_t>(1)}, transformer_engine::DType::kBFloat16);
    auto workspace = makeTransformerEngineTensor(workspace_handle);

    // Allocate output tensor
    auto a_shape = a.shape();
    auto b_shape = a.shape();
    CHECK_EQ(a_shape.ndim, 2);
    CHECK_EQ(b_shape.ndim, 2);
    TFE_TensorHandle* d_eager = AllocateTensor({static_cast<int64_t>(b_shape.data[0]), static_cast<int64_t>(a_shape.data[0])}, TF_FLOAT);
    auto d = makeTransformerEngineTensor(d_eager, transformer_engine::DType::kFloat32);

    cudaStream_t stream = 0;
    nvte_cublas_gemm(a.data(),
                     a_scale_inv.data(),
                     b.data(),
                     b_scale_inv.data(),
                     d.data(),
                     bias.data(),
                     pre_gelu_out.data(),
                     transa,
                     transb,
                     grad,
                     workspace.data(),
                     accumulate,
                     use_split_accumulate,
                     stream);

    return tensorflow::PyoOrThrow(EagerTensorFromHandle(d_eager));
  });
}
