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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_CPU_KERNEL_H_

#include <utility>
#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleTrilinear3DCpuKernelMod : public NativeCpuKernelMod {
 public:
  UpsampleTrilinear3DCpuKernelMod() = default;
  ~UpsampleTrilinear3DCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  struct WeightsAndIndices {
    void operator()(int64_t *const input_index0, int64_t *const input_index1, T *const lambda_0, T *const lambda_1) {
      *input_index0 = id0;
      *input_index1 = id1;
      *lambda_0 = lambda0;
      *lambda_1 = lambda1;
    }
    void Step(const int64_t stride) {
      id0 *= stride;
      id1 *= stride;
    }
    int64_t id0;
    int64_t id1;
    T lambda0;
    T lambda1;
  };

  template <typename S>
  void ComputeWeightsAndIndices(WeightsAndIndices<S> *const wi, S scale, int64_t out_idx, int64_t input_size,
                                int64_t output_size, int64_t stride);

  template <typename S>
  void ComputeHelper(WeightsAndIndices<S> *const helper, S scale, int64_t input_size, int64_t output_size,
                     int64_t stride);

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using KernelRunFunc = std::function<bool(UpsampleTrilinear3DCpuKernelMod *, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list_;
  KernelRunFunc kernel_func_;

  TypeId x_type_{kTypeUnknown};
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> y_shape_;
  std::vector<double> scales_{};
  std::vector<int64_t> none_list_{};
  bool align_corners_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_CPU_KERNEL_H_
