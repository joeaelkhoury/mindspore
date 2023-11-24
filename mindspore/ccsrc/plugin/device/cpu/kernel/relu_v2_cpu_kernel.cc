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

#include "plugin/device/cpu/kernel/relu_v2_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/relu_v2.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore::kernel {
constexpr auto kReLUV2 = "ReLUV2";
constexpr const size_t kReLUV2InputsNum = 1;
constexpr const size_t kReLUV2OutputsNum = 2;
template <typename T>
bool ReLUV2CpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &,
                                      const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReLUV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReLUV2OutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);
  auto *mask = reinterpret_cast<uint8_t *>(outputs[kIndex1]->device_ptr());
  MS_ERROR_IF_NULL_W_RET_VAL(mask, false);

  size_t lens = outputs[0]->size() > 0 ? static_cast<size_t>(outputs[0]->size() / sizeof(T)) : 1;
  auto task = [input, mask, output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      T v = input[i];
      bool p = v > static_cast<T>(0);
      mask[i] = static_cast<uint8_t>(p);
      output[i] = p ? v : static_cast<T>(0);
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_, pool_);
  return true;
}

const std::vector<std::pair<KernelAttr, ReLUV2CpuKernelMod::KernelRunFunc>> &ReLUV2CpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ReLUV2CpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8),
     &ReLUV2CpuKernelMod::LaunchKernel<uint16_t>},
  };
  return func_list;
}

bool ReLUV2CpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kReLUV2InputsNum || outputs.size() != kReLUV2OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kReLUV2InputsNum << " and "
                  << kReLUV2OutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int ReLUV2CpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  if (input_shape.size() < kDim4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dims of input shape must be greater than 4, but got "
                  << input_shape.size();
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLUV2,
                                 []() { return std::make_shared<ReLUV2CpuKernelMod>(kReLUV2); });
}  // namespace mindspore::kernel
