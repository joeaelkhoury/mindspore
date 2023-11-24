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

#include "plugin/device/gpu/kernel/nn/pad_v3_gpu_kernel.h"
#include "kernel/kernel_get_value.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadV3ConstantModeInputsNum = 3;

template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreatePadV3KernelPtr(const std::string &kernel_name,
                                                                    const uint32_t &device_id) {
  return std::make_unique<cukernel::PadV3HelperGpuKernel<T, S>>(kernel_name, device_id);
}
using PadV3PtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

#define REG_PAD_V3_TWO_INTPUT_INT32_KERNEL(INPUT_T, T, S)                                     \
  {                                                                                           \
    KernelAttr().AddInputAttr(INPUT_T).AddInputAttr(kNumberTypeInt32).AddOutputAttr(INPUT_T), \
      CreatePadV3KernelPtr<T, S>                                                              \
  }

#define REG_PAD_V3_TWO_INTPUT_INT64_KERNEL(INPUT_T, T, S)                                     \
  {                                                                                           \
    KernelAttr().AddInputAttr(INPUT_T).AddInputAttr(kNumberTypeInt64).AddOutputAttr(INPUT_T), \
      CreatePadV3KernelPtr<T, S>                                                              \
  }

#define REG_PAD_V3_TWO_INTPUT_KERNEL(INPUT_T, T, S) \
  REG_PAD_V3_TWO_INTPUT_INT32_KERNEL(INPUT_T, T, S), REG_PAD_V3_TWO_INTPUT_INT64_KERNEL(INPUT_T, T, S)

#define REG_PAD_V3_THREE_INTPUT_INT32_KERNEL(INPUT_T, T, S)                                                         \
  {                                                                                                                 \
    KernelAttr().AddInputAttr(INPUT_T).AddInputAttr(kNumberTypeInt32).AddInputAttr(INPUT_T).AddOutputAttr(INPUT_T), \
      CreatePadV3KernelPtr<T, S>                                                                                    \
  }

#define REG_PAD_V3_THREE_INTPUT_INT64_KERNEL(INPUT_T, T, S)                                                         \
  {                                                                                                                 \
    KernelAttr().AddInputAttr(INPUT_T).AddInputAttr(kNumberTypeInt64).AddInputAttr(INPUT_T).AddOutputAttr(INPUT_T), \
      CreatePadV3KernelPtr<T, S>                                                                                    \
  }

#define REG_PAD_V3_THREE_INTPUT_KERNEL(INPUT_T, T, S) \
  REG_PAD_V3_THREE_INTPUT_INT32_KERNEL(INPUT_T, T, S), REG_PAD_V3_THREE_INTPUT_INT64_KERNEL(INPUT_T, T, S)

#define REG_PAD_V3_KERNEL(INPUT_T, T, S) \
  REG_PAD_V3_TWO_INTPUT_KERNEL(INPUT_T, T, S), REG_PAD_V3_THREE_INTPUT_KERNEL(INPUT_T, T, S)

const std::vector<std::pair<KernelAttr, PadV3PtrCreatorFunc>> kernel_attr = {
  REG_PAD_V3_KERNEL(kNumberTypeFloat64, double, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeFloat32, float, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeFloat16, half, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeInt64, int64_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeInt32, int32_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeInt16, int16_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeInt8, int8_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeUInt64, uint64_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeUInt32, uint32_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeUInt16, uint16_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeUInt8, uint8_t, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeComplex64, Complex<float>, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeComplex128, Complex<double>, int64_t),
  REG_PAD_V3_KERNEL(kNumberTypeBool, bool, int64_t),
};
}  // namespace

bool PadV3GpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool PadV3GpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  MS_ERROR_IF_NULL(attr_ptr_);
  attr_ptr_->mode = GetValue<std::string>(primitive_->GetAttr(ops::kMode));
  attr_ptr_->paddings_contiguous = GetValue<bool>(primitive_->GetAttr("paddings_contiguous"));
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);
  return true;
}

int PadV3GpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> paddings_val;
  auto paddings_type = inputs[kIndex1]->dtype_id();
  if (paddings_type == kNumberTypeInt32) {
    std::vector<int32_t> paddings_arg = inputs[kIndex1]->GetValueWithCheck<std::vector<int32_t>>();
    for (size_t i = 0; i < paddings_arg.size(); ++i) {
      paddings_val.push_back(static_cast<int64_t>(paddings_arg[i]));
    }
  } else if (paddings_type == kNumberTypeInt64) {
    paddings_val = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  } else {
    MS_LOG(ERROR) << "For Padv3, the paddings value type should be int64 or int32, but got " << paddings_type;
    return KRET_RESIZE_FAILED;
  }

  int64_t paddings_size = SizeToLong(paddings_val.size());
  auto prim = primitive_;
  MS_EXCEPTION_IF_NULL(prim);
  if (!GetValue<bool>(prim->GetAttr("paddings_contiguous"))) {
    constexpr int64_t nTwo = 2;
    std::vector<int64_t> tmp = paddings_val;
    for (int64_t i = 0; i < paddings_size; ++i) {
      if (i % nTwo == 0) {
        paddings_val[LongToSize(i)] = tmp[LongToSize(i) / nTwo];
      } else {
        paddings_val[LongToSize(i)] = tmp[LongToSize((i + paddings_size) / nTwo)];
      }
    }
  }
  attr_ptr_->paddings = paddings_val;

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> x_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> padding_shape = inputs[1]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[0]->GetShapeVector();
  input_shapes.emplace_back(x_shape);
  input_shapes.emplace_back(padding_shape);
  output_shapes.emplace_back(out_shape);
  if (attr_ptr_->mode == ops::kConstant) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPadV3ConstantModeInputsNum, kernel_name_);
    std::vector<int64_t> constant_value_shape = inputs[2]->GetShapeVector();
    input_shapes.emplace_back(constant_value_shape);
  }
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> PadV3GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, PadV3PtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, PadV3, PadV3GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
