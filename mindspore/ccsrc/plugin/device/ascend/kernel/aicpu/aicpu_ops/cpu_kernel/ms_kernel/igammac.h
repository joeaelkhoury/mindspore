/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_IGAMMAC_H_
#define AICPU_KERNELS_NORMALIZED_IGAMMAC_H_

#include "cpu_kernel_utils.h"
#include "utils/bcast.h"

namespace aicpu {

class IgammacCpuKernel : public CpuKernel {
 public:
  IgammacCpuKernel() = default;
  ~IgammacCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static uint32_t IgammacCheckAndBroadCast(CpuKernelContext &ctx, BCalcInfo *calc_info);

  template <typename T>
  static uint32_t IgammacCompute(const CpuKernelContext &ctx, const BCalcInfo &calc_info);
};
}  // namespace aicpu
#endif
