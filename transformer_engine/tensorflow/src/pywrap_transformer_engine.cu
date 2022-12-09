#include <string>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include "pybind11/pybind11.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/core/protobuf/config.pb.h"

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

void Deallocator(void* data, size_t unused, void* tensor_handle) {
  tensorflow::GPUOptions gpu_options;
  tensorflow::TfDeviceId device_id(0);
  tensorflow::Allocator* allocator = tensorflow::GPUProcessState::singleton()->GetGPUAllocator(
      gpu_options, device_id, /*total_bytes=*/1, /*peer_gpu_ids=*/{});
  allocator->DeallocateRaw(data);
}


TFE_TensorHandle* AllocateTensor(std::vector<int64_t> shape, TF_DataType dtype, size_t size) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);

  // Allocate GPU memory.
  tensorflow::GPUOptions gpu_options;
  tensorflow::TfDeviceId device_id(0);
  tensorflow::Allocator* allocator = tensorflow::GPUProcessState::singleton()->GetGPUAllocator(
      gpu_options, device_id, /*total_bytes=*/1, /*peer_gpu_ids=*/{});
  size_t num_bytes = size;
  for (int i = 0; i < shape.size(); ++i) num_bytes *= shape[i];
  void* data = allocator->AllocateRaw(tensorflow::Allocator::kAllocatorAlignment, num_bytes);

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
  py::enum_<transformer_engine::DType>(m, "DType", py::arithmetic(), "NVTE Datatypes")
    .value("Byte", transformer_engine::DType::kByte)
    .value("Int32", transformer_engine::DType::kInt32)
    .value("Float32", transformer_engine::DType::kFloat32)
    .value("Float16", transformer_engine::DType::kFloat16)
    .value("BFloat16", transformer_engine::DType::kBFloat16)
    .value("Float8E4M3", transformer_engine::DType::kFloat8E4M3)
    .value("Float8E5M2", transformer_engine::DType::kFloat8E5M2)
    .export_values();
  m.def("cast_to_fp8", [](const pybind11::handle& input,
                          const pybind11::handle& scale,
                          const pybind11::handle& amax,
                          const pybind11::handle& scale_inv,
                          const transformer_engine::DType output_dtype) {
    // Get NVTE tensors.
    auto input_tensor = makeTransformerEngineTensor(input);
    auto scale_tensor = makeTransformerEngineTensor(scale);
    auto scale_inv_tensor = makeTransformerEngineTensor(scale_inv);
    auto amax_tensor = makeTransformerEngineTensor(amax);

    // Allocate output tensor
    auto in_shape = input_tensor.shape();
    std::vector<int64_t> shape_vec(in_shape.data, in_shape.data + in_shape.ndim);
    TFE_TensorHandle* output_eager = AllocateTensor(shape_vec, TF_INT8, sizeof(uint8_t));
    auto output_tensor = makeTransformerEngineTensor(output_eager, output_dtype);

    cudaStream_t stream = 0;
    nvte_fp8_quantize(input_tensor.data(), scale_tensor.data(), output_tensor.data(), amax_tensor.data(), scale_inv_tensor.data(), stream);
    return tensorflow::PyoOrThrow(EagerTensorFromHandle(output_eager));
  });
  // TODO(trevor): can we expose the dtype as the enum class defined in
  // transformer_engine::DType?
  // Since the native TF tensor doesn't support the fp8 formats (i.e. e5m2 or
  // e4m3) yet, we pass the a_dtype and b_dtype explicitly and the tensor with
  // int8 storage format. d_dtype and bias_dtype can be obtained from the passed
  // tensors.
  m.def("fp8_gemm", [](const pybind11::handle& a_handle,
                       const pybind11::handle& a_scale_inv_handle,
                       const transformer_engine::DType a_dtype_in,
                       const pybind11::handle& b_handle,
                       const pybind11::handle& b_scale_inv_handle,
                       const transformer_engine::DType b_dtype_in,
                       const pybind11::handle& d_handle,
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
    auto d = makeTransformerEngineTensor(d_handle, transformer_engine::DType::kFloat32);

    transformer_engine::TensorWrapper bias(nullptr, {static_cast<size_t>(1)}, transformer_engine::DType::kBFloat16);
    if (use_bias) {
      bias = makeTransformerEngineTensor(bias_handle.ptr());
    }
    
    transformer_engine::TensorWrapper pre_gelu_out(nullptr, {static_cast<size_t>(1)}, transformer_engine::DType::kBFloat16);
    auto workspace = makeTransformerEngineTensor(workspace_handle);

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
  });
}
