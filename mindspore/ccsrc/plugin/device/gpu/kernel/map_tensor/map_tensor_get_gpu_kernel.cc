/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/map_tensor/map_tensor_get_gpu_kernel.h"
#include <utility>
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, MapTensorGetGpuKernelMod::MapTensorGetLaunchFunc>>
  MapTensorGetGpuKernelMod::map_tensor_get_func_list_ = {{KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt32)
                                                            .AddOutputAttr(kNumberTypeFloat32),
                                                          &MapTensorGetGpuKernelMod::LaunchKernel<int32_t, float>},
                                                         {KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt64)
                                                            .AddOutputAttr(kNumberTypeFloat32),
                                                          &MapTensorGetGpuKernelMod::LaunchKernel<int64_t, float>}};

std::vector<KernelAttr> MapTensorGetGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    map_tensor_get_func_list_.begin(), map_tensor_get_func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MapTensorGetGpuKernelMod::MapTensorGetLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool MapTensorGetGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = prim->name();
  insert_default_value_ = GetValue<bool>(prim->GetAttr(kAttrInsertDefaultValue));
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetOutputNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = map_tensor_get_func_list_[index].second;

  input_key_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  output_type_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  return true;
}

int MapTensorGetGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();

  MS_EXCEPTION_IF_NULL(inputs.at(kIndex1));
  const auto &keys_shape = inputs.at(kIndex1)->GetShapeVector();
  MS_EXCEPTION_IF_NULL(outputs.at(kIndex0));
  const auto &output_shape = outputs.at(kIndex0)->GetShapeVector();

  if (IsDynamic(keys_shape) || IsDynamic(output_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }

  InitSizeLists(keys_shape, output_shape);
  return KRET_OK;
}

template <typename KeyType, typename ValueType>
bool MapTensorGetGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetOutputNum, kernel_name_);

  // The real hash table should be accessed by user data.
  if (input_user_data_.empty()) {
    MS_LOG(EXCEPTION) << "The hash table user data is not set yet.";
  }

  auto user_data = input_user_data_[kIndex0];
  MS_EXCEPTION_IF_NULL(user_data);
  auto hash_table_ptr = user_data->get<GPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(hash_table_ptr);
  return hash_table_ptr->Find(static_cast<KeyType *>(inputs.at(kIndex1)->addr),
                              inputs.at(kIndex1)->size / sizeof(KeyType), insert_default_value_,
                              static_cast<ValueType *>(outputs.at(kIndex0)->addr), stream_ptr);
}

void MapTensorGetGpuKernelMod::InitSizeLists(const ShapeVector &keys_shape, const ShapeVector &output_shape) {
  // Return size 1 as the first input size for MapTensorGet. Real memory should be assigned by MindRT.
  input_size_list_.push_back(kSizeOne);

  auto keys_size = std::accumulate(keys_shape.begin(), keys_shape.end(), 1, std::multiplies{});
  MS_EXCEPTION_IF_ZERO("keys size", keys_size);
  input_size_list_.push_back(keys_size * input_key_type_size_);

  auto output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies{});
  MS_EXCEPTION_IF_ZERO("output size", output_size);
  output_size_list_.push_back(output_size * output_type_size_);
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MapTensorGet, MapTensorGetGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
