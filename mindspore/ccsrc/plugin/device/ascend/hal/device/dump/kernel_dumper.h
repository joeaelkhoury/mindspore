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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_KERNEL_DUMPER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_KERNEL_DUMPER_H_
#include <tuple>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <functional>
#include "include/backend/kernel_graph.h"
#include "mindspore/ccsrc/kernel/kernel.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task.h"
#include "include/backend/debug/data_dump/overflow_dumper.h"

using AddressPtr = mindspore::kernel::AddressPtr;
using mindspore::ge::model_runner::HcclTaskInfo;

namespace aicpu {
namespace dump {
class OpMappingInfo;
class Task;
}  // namespace dump
}  // namespace aicpu

namespace mindspore {
namespace device {
namespace ascend {
const char *const kDumpKernelsDumpOp = "DumpDataInfo";
#define GE_MODULE_NAME_U16 static_cast<uint16_t>(45)

class OpDebugTask {
 public:
  OpDebugTask() = default;
  ~OpDebugTask();

  uint32_t debug_stream_id = 0U;
  uint32_t debug_task_id = 0U;
  void *op_debug_addr = nullptr;
  void *new_op_debug_addr = nullptr;
  friend class KernelDumper;
};

class KernelDumper : public debug::OverflowDumper {
 public:
  KernelDumper() = default;
  ~KernelDumper();

  void OpLoadDumpInfo(const CNodePtr &kernel) override;
  void DumpHcclOutput(const std::shared_ptr<HcclTaskInfo> &task_info, const aclrtStream &stream);
  void Init() override;
  void ExecutorDumpOp(const aicpu::dump::OpMappingInfo &op_mapping_info, const aclrtStream &stream);
#ifndef ENABLE_SECURITY
  void OpDebugRegisterForStream(const CNodePtr &kernel) override;
  void OpDebugUnregisterForStream() override;
#endif
  static std::map<aclrtStream, std::unique_ptr<OpDebugTask>> op_debug_tasks;
  static std::map<uint32_t, bool> is_data_map;
  static std::map<std::string, std::string> stream_task_graphs;
  static std::mutex dumper_mutex_;

  string dump_path_;
  string net_name_;
  string iteration_;

 private:
  // Support multi-thread.
  bool load_flag_;
  uint32_t graph_id_;
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  bool initialed_{false};
  bool already_print_{false};
  bool is_op_debug_;
  uint32_t op_debug_mode_;
  std::mutex dump_mutex_;

  void *dev_load_mem_ = nullptr;
  void *proto_dev_mem_ = nullptr;
  void *proto_size_dev_mem_ = nullptr;
  std::string overflow_dump_filename = "debug_files";
  void *p2p_debug_addr_ = nullptr;
  void SetOpMappingInfo(NotNull<aicpu::dump::OpMappingInfo *> dump_info, const CNodePtr &kernel);
  void SetOpMappingInfo(NotNull<aicpu::dump::OpMappingInfo *> dump_info,
                        const std::shared_ptr<HcclTaskInfo> &task_info);

  void ConstructDumpTask(NotNull<const CNodePtr &> kernel, NotNull<aicpu::dump::Task *> dump_task);
  void DumpKernelOutput(const CNodePtr &kernel, NotNull<aicpu::dump::Task *> task) const;
  void DumpKernelOutput(const std::shared_ptr<HcclTaskInfo> &task_info, NotNull<aicpu::dump::Task *> task);
  void DumpKernelInput(const CNodePtr &kernel, NotNull<aicpu::dump::Task *> task) const;
  std::string StripUniqueId(const std::string node_name) const;

  void SetOpMappingInfoRegister(NotNull<aicpu::dump::OpMappingInfo *> dump_info, const CNodePtr &kernel);
  void MallocP2PDebugMem(const void *const op_debug_addr);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_KERNEL_DUMPER_H_
