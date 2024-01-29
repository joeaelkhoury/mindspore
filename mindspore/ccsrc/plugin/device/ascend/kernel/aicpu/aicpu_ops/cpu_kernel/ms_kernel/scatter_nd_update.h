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

#ifndef AICPU_KERNELS_NORMALIZED_SCATTERNDUPDATE_H_
#define AICPU_KERNELS_NORMALIZED_SCATTERNDUPDATE_H_

#include <string.h>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class ScatterNdUpdateCpuKernel : public CpuKernel {
 public:
  ScatterNdUpdateCpuKernel() = default;
  ~ScatterNdUpdateCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename var_type>
  uint32_t DTYPE_CHOOSE(const CpuKernelContext &ctx);

  template <typename var_type, typename indices_type>
  uint32_t ScatterNdUpdateComputeRealKernel(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
