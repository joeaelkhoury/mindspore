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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_KERNEL_EXECUTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_KERNEL_EXECUTOR_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_graph_executor.h"
#include "plugin/device/ascend/hal/hardware/ascend_somas.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendKernelExecutor : public KernelExecutor {
 public:
  AscendKernelExecutor() = default;
  ~AscendKernelExecutor() override = default;

  void Initialize() override;
  void Destroy() override;

  // Optimize the kernel graph for graph mode.
  void OptimizeGraph(const FuncGraphPtr &graph) const override;

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;

  // Adjust kernel graph before run graph, used in Graph Mode.
  void PreprocessBeforeRun(const FuncGraphPtr &graph) const override;

  // Launch a kernel via 'KernelMod' of the kernel.
  bool LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                    const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                    size_t stream_id) const override;

  // Unify the MindIR, the default behavior uses the common unified MindIR.
  void UnifyMindIR(const KernelGraphPtr &graph) const override;
  void AddMindIRPass(const KernelGraphPtr &graph) const override;

  // Get rank id for distributed training.
  uint32_t GetRankID() const override { return res_manager_->rank_id_; }

  bool ExecuteKernelTask(const pynative::KernelTaskType &task_type, const device::DeviceAddressPtrList &input_addr_list,
                         const TensorStorageInfoPtrList &input_storage_list,
                         const device::DeviceAddressPtrList &output_addr_list) const override;

 private:
  // Launch device aicpu library
  void LaunchDeviceLibrary() const;

  void SetAtomicCleanToNodes(const KernelGraphPtr &graph,
                             const std::map<CNodePtr, std::vector<CNodePtr>> &atomics_node) const;

  // launch
  bool PySyncRuning() const;
  bool MemoryCopyAsync(const CNodePtr &node, const vector<KernelTensor *> &inputs,
                       const vector<KernelTensor *> &outputs) const;
  bool LaunchAtomicClean(const CNodePtr &node, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream) const;

  bool GetKernelRealInputs(const CNodePtr &kernel, const vector<KernelTensor *> &inputs,
                           std::vector<KernelTensor *> *real_inputs) const;
  void PreprocessBeforeRunGraph(const KernelGraphPtr &graph) const;
  void PreprocessBeforeRunSingleOpGraph(const KernelGraphPtr &graph) const;
  static void DoSomas(const KernelGraphPtr &graph);

  // Using node to get it's atomics
  mutable std::map<CNodePtr, std::vector<CNodePtr>> node_atomics_;
  // Persistent cache for single op execution.
  // node_atomics_ will be cleaned up in CompileGraph.
  mutable std::map<CNodePtr, std::vector<CNodePtr>> node_atomics_persistent_cache_;
  mutable std::set<CNodePtr> nop_op_to_memcpy_;
  mutable std::mutex launch_mutex_;
  AscendDeviceResManager *res_manager_{nullptr};
  AscendGraphExecutor *graph_executor_{nullptr};
  bool initialized_ = false;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_KERNEL_EXECUTOR_H_
