/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ge_runtime/task/tbe_task.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "runtime/kernel.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
TbeTask::TbeTask(const ModelContext &model_context, const std::shared_ptr<TbeTaskInfo> &task_info)
    : TaskRepeater<TbeTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      stub_func_(nullptr),
      args_(nullptr),
      args_size_(0) {
  MS_EXCEPTION_IF_NULL(task_info);

  auto stream_list = model_context.stream_list();
  if (stream_list.size() == 1) {
    stream_ = stream_list[0];
  } else if (stream_list.size() > task_info->stream_id()) {
    stream_ = stream_list[task_info->stream_id()];
  } else {
    MS_LOG(EXCEPTION) << "Index: " << task_info->stream_id() << " >= stream_list.size(): " << stream_list.size();
  }
}

TbeTask::~TbeTask() {
  if (args_ != nullptr) {
    auto rt_ret = aclrtFree(args_);
    if (rt_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rt api aclrtFree failed, ret: " << rt_ret;
    }
    args_ = nullptr;
    stub_func_ = nullptr;
    stream_ = nullptr;
  }
}

std::string TbeTask::DebugString() const {
  std::ostringstream buffer;
  buffer << "Tbetask: task name: " << task_info_->op_name() << ", stub_func: " << task_info_->stub_func()
         << ", mindspore stream_id: " << task_info_->stream_id() << ", stream_: " << stream_
         << ", block_dim: " << task_info_->block_dim() << ", dump_flag: " << task_info_->dump_flag();
  auto input_data = task_info_->input_data_addrs();
  auto output_data = task_info_->output_data_addrs();
  auto workspace = task_info_->workspace_addrs();
  // args_size should be obtained from task instead of task_info
  size_t addr_size = input_data.size() + output_data.size() + workspace.size();
  buffer << ", args_size: " << addr_size * sizeof(void *);
  buffer << ", input_data_addr: ";
  for (size_t i = 0; i < input_data.size(); ++i) {
    buffer << "[" << i << "]: " << input_data[i];
    if (i != input_data.size() - 1) {
      buffer << ", ";
    }
  }
  buffer << ", output_data_addr: ";
  for (size_t i = 0; i < output_data.size(); ++i) {
    buffer << "[" << i << "]: " << output_data[i];
    if (i != output_data.size() - 1) {
      buffer << ", ";
    }
  }
  buffer << ", workspace_addr: ";
  for (size_t i = 0; i < workspace.size(); ++i) {
    buffer << "[" << i << "]: " << workspace[i];
    if (i != workspace.size() - 1) {
      buffer << ", ";
    }
  }
  return buffer.str();
}

void TbeTask::Distribute() {
  MS_LOG(INFO) << "InitTbeTask and DistributeTbeTask start.";
  MS_EXCEPTION_IF_NULL(task_info_);
  MS_EXCEPTION_IF_NULL(stream_);
  // Get stub_func
  if (task_info_->stub_func().empty()) {
    MS_LOG(EXCEPTION) << "kernel_info->stub_func is empty!";
  }

  rtError_t rt_ret = rtGetFunctionByName(const_cast<char *>(task_info_->stub_func().c_str()), &stub_func_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtGetFunctionByName failed, ret: " << rt_ret;
  }

  // Get args
  std::vector<void *> tensor_device_addrs;
  tensor_device_addrs.insert(tensor_device_addrs.cend(), task_info_->input_data_addrs().cbegin(),
                             task_info_->input_data_addrs().cend());
  tensor_device_addrs.insert(tensor_device_addrs.cend(), task_info_->output_data_addrs().cbegin(),
                             task_info_->output_data_addrs().cend());
  tensor_device_addrs.insert(tensor_device_addrs.cend(), task_info_->workspace_addrs().cbegin(),
                             task_info_->workspace_addrs().cend());
  args_size_ = static_cast<uint32_t>(tensor_device_addrs.size() * sizeof(void *));

  rt_ret = rtMalloc(&args_, args_size_, RT_MEMORY_HBM, 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtMalloc failed, ret: " << rt_ret << " mem size " << args_size_;
  }

  rt_ret = aclrtMemcpy(args_, args_size_, static_cast<void *>(tensor_device_addrs.data()), args_size_,
                       ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api aclrtMemcpy failed, ret: " << rt_ret;
  }

  auto dump_flag = task_info_->dump_flag() ? RT_KERNEL_DUMPFLAG : RT_KERNEL_DEFAULT;
  rtArgsEx_t args_info = {};
  args_info.args = args_;
  args_info.argsSize = SizeToUint(args_size_);
  rt_ret = rtKernelLaunchWithFlag(stub_func_, task_info_->block_dim(), &args_info, nullptr, stream_, dump_flag);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtKernelLaunch failed, ret: " << rt_ret << " mem size " << args_size_;
  }
  MS_LOG(INFO) << "TbeTask: stub_func = " << task_info_->stub_func()
               << ", [DataDump] task name: " << task_info_->op_name() << " dump_flag: " << dump_flag;
}

REGISTER_TASK(TaskInfoType::TBE, TbeTask, TbeTaskInfo);
}  // namespace mindspore::ge::model_runner
