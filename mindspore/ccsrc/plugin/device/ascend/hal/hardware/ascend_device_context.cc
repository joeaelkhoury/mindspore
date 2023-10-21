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

#include "plugin/device/ascend/hal/hardware/ascend_device_context.h"
#include <map>
#include <memory>
#include <string>
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#include "debug/debugger/proto_exporter.h"
#endif
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "runtime/dev.h"

using mindspore::profiler::ascend::AscendProfiler;
#endif

namespace mindspore {
namespace device {
namespace ascend {
void AscendDeviceContext::Initialize() {
  GilReleaseWithCheck gil_release;
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    runtime_instance_->SetContext();
    return;
  } else {
    MS_LOG(INFO) << "Start Initialize...";
#ifndef ENABLE_SECURITY
    AscendProfiler::GetInstance()->MsprofInitProfiler();
#endif
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = dynamic_cast<AscendKernelRuntime *>(
    device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id));
  MS_EXCEPTION_IF_NULL(runtime_instance_);
#ifndef ENABLE_SECURITY
  runtime_instance_->PreInit();
#endif
  // enable hccl and init hccl not done, skip the rest step.
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) &&
      !distributed::collective::CollectiveManager::instance()->initialized()) {
    return;
  }

  DeviceContext::SetDynKernelExecutor(std::make_shared<GeKernelExecutor>());
  GetKernelExecutor(true)->SetDeviceContext(this);

  auto force_acl = common::GetEnv("MS_DEV_FORCE_ACL");
  if (!force_acl.empty()) {
    DeviceContext::SetKernelExecutor(GetKernelExecutor(true));
    GetKernelExecutor(false)->SetDeviceContext(this);
  }

  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();
  auto ascend_res_manager = dynamic_cast<AscendDeviceResManager *>(device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(ascend_res_manager);
  runtime_instance_ = ascend_res_manager->runtime_instance_;
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
  GetKernelExecutor(false)->Initialize();
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(true));
  GetKernelExecutor(true)->Initialize();
  auto ascend_graph_executor = dynamic_cast<AscendGraphExecutor *>(graph_executor_.get());
  MS_EXCEPTION_IF_NULL(ascend_graph_executor);
  ascend_graph_executor->Initialize();
  initialized_ = true;
  MS_LOG(INFO) << "Initialize success.";
}

void AscendDeviceContext::Destroy() {
#ifndef ENABLE_SECURITY
  AscendProfiler::GetInstance()->MsprofStopProfiler();
#endif
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger && debugger->debugger_enabled()) {
    debugger->SetTrainingDone(true);
    bool ret = debugger->SendMetadata(false);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to SendMetadata when finalize";
    }
  }
#endif
  MS_LOG(INFO) << "Enter Destroy...";
  if (!initialized_) {
    if (deprecated_interface_ != nullptr) {
      (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
    }
    return;
  }

  MS_LOG(INFO) << "Start Destroy ";
  auto ascend_graph_executor = dynamic_cast<AscendGraphExecutor *>(graph_executor_.get());
  MS_EXCEPTION_IF_NULL(ascend_graph_executor);
  ascend_graph_executor->Destroy();
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
  GetKernelExecutor(false)->Destroy();
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(true));
  GetKernelExecutor(true)->Destroy();
  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Destroy();
  if (runtime_instance_) {
    runtime_instance_ = nullptr;
  }
  if (deprecated_interface_ != nullptr) {
    (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
  }
  initialized_ = false;
  MS_LOG(INFO) << "Destroy success.";
}

// @todo move SetRunMode to here when old runtime is delete
bool AscendDeviceContext::PartitionGraph(const FuncGraphPtr &func_graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
}

RunMode AscendDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) && !IsDynamicShapeGraph(func_graph)) {
    return RunMode::kGraphMode;
  } else {
    return RunMode::kKernelMode;
  }
}

DeprecatedInterface *AscendDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>(nullptr);
  }
  return deprecated_interface_.get();
}

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
MS_REGISTER_DEVICE(kDavinciMultiGraphInferenceDevice, AscendDeviceContext);

void AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  for (auto output : outputs) {
    if (!output->isa<CNode>() || !AnfUtils::IsRealKernel(output)) {
      continue;
    }

    if (!common::AnfAlgo::IsNopNode(output)) {
      continue;
    }

    if (!common::AnfAlgo::IsNeedSkipNopOpAddr(output)) {
      continue;
    }

    size_t input_num = common::AnfAlgo::GetInputTensorNum(output);
    if (input_num != 1) {
      MS_LOG(WARNING) << "The input number of nop node :" << output->fullname_with_scope() << " is " << input_num
                      << ", not equal 1";
      continue;
    }

    auto real_input_index = AnfAlgo::GetInputGraphIdxByKernelIdx(output, 0);
    auto pre_node_out_device_address = AnfAlgo::GetPrevNodeOutputAddr(output, real_input_index);
    MS_EXCEPTION_IF_NULL(pre_node_out_device_address);
    auto ptr = pre_node_out_device_address->GetPtr();
    auto size = pre_node_out_device_address->GetSize();
    std::string output_format = AnfAlgo::GetOutputFormat(output, 0);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(output, 0);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
      const_cast<void *>(ptr), size, output_format, output_type, trans::GetRuntimePaddingShape(output, 0));
    // If graph has the flag kFlagEnableZeroCopyInGraph, it means the graph should run in graph mode, the device
    // address of graph output should not be persisted, and its output addr will be replaced after RunGraph.
    // As we fetch the output device address of a nopnode, we can only get the input device address of it, so we
    // have to prevent the ptr persist flag of the device address here.
    if (!graph->has_flag(kFlagEnableZeroCopyInGraph)) {
      device_address->set_is_ptr_persisted(true);
    }
    device_address->set_host_shape(trans::GetRuntimePaddingShape(output, 0));
    AnfAlgo::SetOutputAddr(device_address, 0, output.get());
    common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(false), output);
    MS_LOG(INFO) << "Assign device address to output nop node " << output->fullname_with_scope();
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
