/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <map>
#include "kernel/kernel_get_value.h"
#include "plugin/device/gpu/kernel/arrays/tile_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 2;
constexpr size_t kTileOutputsNum = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
}  // namespace
bool TileGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = base_operator->name();
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int TileGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  input_shape_.clear();
  output_shape_.clear();
  std::transform(input_shape.cbegin(), input_shape.cend(), std::back_inserter(input_shape_), LongToSize);
  std::transform(output_shape.cbegin(), output_shape.cend(), std::back_inserter(output_shape_), LongToSize);
  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape_, kernel_name_, "output");
  if (is_null_input_) {
    return true;
  }
  if (output_shape_.size() < kTileOutputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 1, but got "
                      << output_shape_.size();
  }
  if (output_shape_.size() > TILE_MAX_DIMENSION) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be greater than "
                      << TILE_MAX_DIMENSION << ", but got " << output_shape_.size();
  }
  std::vector<int64_t> multiples;
  TryGetIntValue(inputs, kIndex1, kernel_name_, &multiples);
  size_t filling_value = multiples.size() - input_shape_.size();
  (void)input_shape_.insert(input_shape_.begin(), filling_value, kIndex1);
  return KRET_OK;
}

template <typename T>
bool TileGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto status = CalTile(input_shape_, output_shape_, input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;

std::vector<std::pair<KernelAttr, TileGpuKernelMod::TileLaunchFunc>> TileGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &TileGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &TileGpuKernelMod::LaunchKernel<Complex<double>>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &TileGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &TileGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &TileGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &TileGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &TileGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &TileGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &TileGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &TileGpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &TileGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &TileGpuKernelMod::LaunchKernel<Complex<double>>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &TileGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &TileGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &TileGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &TileGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &TileGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &TileGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &TileGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &TileGpuKernelMod::LaunchKernel<bool>}};

std::vector<KernelAttr> TileGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TileGpuKernelMod::TileLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Tile, TileGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
