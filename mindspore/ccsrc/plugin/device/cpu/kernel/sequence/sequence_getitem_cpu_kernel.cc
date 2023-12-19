/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/sequence_getitem_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool SequenceGetItemCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SequenceGetItemCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty() || inputs[0] == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input is invalid, input size:" << inputs.size();
  }
  tuple_shape_ = inputs[0]->GetShapeVector();
  if (tuple_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input tuple size must greater 0";
  }
  return KRET_OK;
}

template <typename T>
bool SequenceGetItemCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs) {
  const auto input_addr = GetDeviceAddress<T>(inputs, 0);
  MS_EXCEPTION_IF_NULL(input_addr);
  auto index = *(GetDeviceAddress<int64_t>(inputs, 1));
  auto output_addr = GetDeviceAddress<T>(outputs, 0);
  MS_EXCEPTION_IF_NULL(output_addr);
  auto len = static_cast<int64_t>(tuple_shape_[0]);
  if (index >= len || index < -len) {
    MS_EXCEPTION(ValueError) << "index is out of range: " << -len << " <= index < " << len << ", but got " << index
                             << ".";
  }
  if (index < 0) {
    index += len;
  }
  if (tuple_shape_.size() == 1) {
    *output_addr = input_addr[index];
    return true;
  }
  auto output_size = output_size_list_[0];
  size_t element_index_size =
    static_cast<size_t>(std::accumulate(tuple_shape_.begin() + 1, tuple_shape_.end(), 1, std::multiplies<int64_t>()));
  size_t input_addr_offset = element_index_size * LongToSize(index);
  auto cp_ret = memcpy_s(output_addr, output_size, input_addr + input_addr_offset, element_index_size * sizeof(T));
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
  }

  return true;
}

const std::vector<std::pair<KernelAttr, SequenceGetItemCpuKernelMod::KernelRunFunc>>
  &SequenceGetItemCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SequenceGetItemCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt32),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &SequenceGetItemCpuKernelMod::LaunchKernel<int64_t>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RealTupleGetItem, SequenceGetItemCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
