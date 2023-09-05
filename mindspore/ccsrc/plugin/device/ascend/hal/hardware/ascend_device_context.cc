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

#ifdef WITH_BACKEND
namespace {
void SetContextSocVersion(MsContext *ctx) {
  constexpr auto k910AAscendVersion = "Ascend910";
  constexpr auto k910BAscendVersion = "ascend910b";
  const std::map<std::string, std::string> kAscendSocVersions = {
    {"Ascend910A", "ascend910"},    {"Ascend910B", "ascend910"},    {"Ascend910PremiumA", "ascend910"},
    {"Ascend910ProA", "ascend910"}, {"Ascend910ProB", "ascend910"}, {"Ascend910B1", "ascend910b"},
    {"Ascend910B2", "ascend910b"},  {"Ascend910B3", "ascend910b"},  {"Ascend910B4", "ascend910b"}};
  // Get default soc version.
  static std::string version;
  if (version.empty()) {
    const int kSocVersionLen = 50;
    char soc_version[kSocVersionLen] = {0};
    auto ret = rtGetSocVersion(soc_version, kSocVersionLen);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "GetSocVersion failed.";
    }
    version = soc_version;
  }
  auto iter = kAscendSocVersions.find(version);
  if (iter == kAscendSocVersions.end()) {
    MS_LOG(INFO) << "The soc version is not Ascend910 or ascend910b.";
    return;
  }
  if (iter->second == k910BAscendVersion) {
    ctx->set_ascend_soc_version(k910BAscendVersion);
  } else if (iter->second == k910AAscendVersion) {
    ctx->set_ascend_soc_version(k910AAscendVersion);
  }
}
}  // namespace
#endif

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
MS_REGISTER_DEVICE(kDavinciMultiGraphInferenceDevice, AscendDeviceContext);
#ifdef WITH_BACKEND
MSCONTEXT_REGISTER_INIT_FUNC(kAscendDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  auto enable_ge = mindspore::common::GetEnv("MS_ENABLE_GE");
  if (enable_ge == "1") {
    if (ctx->backend_policy() != "ge") {
      (void)ctx->set_backend_policy("ge");
    }
  } else {
    if (ctx->backend_policy() != "ms") {
      (void)ctx->set_backend_policy("ms");
    }
  }
  SetContextSocVersion(ctx);
});
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
