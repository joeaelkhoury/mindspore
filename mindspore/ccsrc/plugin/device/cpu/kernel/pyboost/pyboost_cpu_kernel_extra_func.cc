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

#include "plugin/device/cpu/kernel/pyboost/pyboost_cpu_kernel_extra_func.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void PyboostCPUKernelExtraFunc::SetThreadPool(const kernel::KernelModPtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);

  const auto &cpu_kernel = std::dynamic_pointer_cast<kernel::NativeCpuKernelMod>(kernel);
  MS_EXCEPTION_IF_NULL(cpu_kernel);
  auto thread_pool = kernel::GetActorMgrInnerThreadPool();
  cpu_kernel->SetThreadPool(thread_pool);
}

bool PyboostCPUKernelExtraFunc::IsKernelModRegistered(const std::string &op_name) {
  return kernel::Factory<kernel::NativeCpuKernelMod>::Instance().IsRegistered(op_name);
}

REG_PYBOOST_KERNEL_EXTRA_FUN(CPU, PyboostCPUKernelExtraFunc);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
