/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/gpu/kernel/arrays/gather_gpu_kernel.h"
#include <memory>
#include "mindspore/core/ops/gather.h"
#include "kernel/kernel_get_value.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
const size_t kInputNum = 3;
bool GatherFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (kernel_name_ == ops::kNameGather) {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::Gather>(base_operator);
    MS_EXCEPTION_IF_NULL(kernel_ptr);
    batch_dims_ = kernel_ptr->get_batch_dims();
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  indices_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  axis_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);
  axis_type_ = inputs.at(kIndex2)->GetDtype();
  return true;
}

int GatherFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  if (!TryGetIntValue(inputs, kIndex2, kernel_name_, &axis_)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', cant get axis.";
    return KRET_RESIZE_FAILED;
  }

  input_shapes_ = inputs[kIndexZero]->GetShapeVector();
  indices_shapes_ = inputs[kIndexOne]->GetShapeVector();
  output_shapes_ = outputs[kIndexZero]->GetShapeVector();
  if (batch_dims_ < 0) {
    batch_dims_ += SizeToLong(indices_shapes_.size());
  }
  is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(indices_shapes_, kernel_name_, "indices") ||
                   CHECK_SHAPE_NULL(output_shapes_, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }
  int dims = SizeToInt(input_shapes_.size());
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << "," << dims
                      << "), but got " << axis_;
  }
  Reshape();
  return KRET_OK;
}

std::vector<KernelAttr> GatherFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T, typename S, typename G>
bool GatherFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices_addr = GetDeviceAddress<S>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto input_dim1 = input_shapes_[IntToSize(axis_)];
  auto status = Gather(input_addr, indices_addr, output_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2],
                       dims_[kIndex3], LongToSize(input_dim1), reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define GATHER_GPU_REG(MS_T, MS_S, MS_A, T, S, A)                                            \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddInputAttr(MS_A).AddOutputAttr(MS_T), \
    &GatherFwdGpuKernelMod::LaunchKernel<T, S, A>

#define GATHER_GPU_INDEX_REG(MS_T, T)                                                  \
  {GATHER_GPU_REG(MS_T, kNumberTypeInt32, kNumberTypeInt32, T, int32_t, int32_t)},     \
    {GATHER_GPU_REG(MS_T, kNumberTypeInt64, kNumberTypeInt64, T, int64_t, int64_t)},   \
    {GATHER_GPU_REG(MS_T, kNumberTypeInt32, kNumberTypeInt64, T, int32_t, int64_t)}, { \
    GATHER_GPU_REG(MS_T, kNumberTypeInt64, kNumberTypeInt32, T, int64_t, int32_t)      \
  }

std::vector<std::pair<KernelAttr, GatherFwdGpuKernelMod::GatherFunc>> GatherFwdGpuKernelMod::func_list_ = {{
  GATHER_GPU_INDEX_REG(kNumberTypeComplex64, mindspore::utils::Complex<float>),
  GATHER_GPU_INDEX_REG(kNumberTypeComplex128, mindspore::utils::Complex<double>),
  GATHER_GPU_INDEX_REG(kNumberTypeFloat64, double),
  GATHER_GPU_INDEX_REG(kNumberTypeFloat32, float),
  GATHER_GPU_INDEX_REG(kNumberTypeFloat16, half),
  GATHER_GPU_INDEX_REG(kNumberTypeInt64, int64_t),
  GATHER_GPU_INDEX_REG(kNumberTypeInt32, int32_t),
  GATHER_GPU_INDEX_REG(kNumberTypeInt16, int16_t),
  GATHER_GPU_INDEX_REG(kNumberTypeInt8, int8_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt64, uint64_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt32, uint32_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt16, uint16_t),
  GATHER_GPU_INDEX_REG(kNumberTypeUInt8, uint8_t),
  GATHER_GPU_INDEX_REG(kNumberTypeBool, bool),
}};

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Gather,
                                 []() { return std::make_shared<GatherFwdGpuKernelMod>(kGather); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SparseGatherV2,
                                 []() { return std::make_shared<GatherFwdGpuKernelMod>(kSparseGatherV2); });
}  // namespace kernel
}  // namespace mindspore
