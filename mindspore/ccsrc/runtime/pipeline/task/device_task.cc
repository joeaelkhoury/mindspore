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

#include <string>
#include "include/common/profiler.h"
#include "runtime/pipeline/task/device_task.h"

namespace mindspore {
namespace runtime {
namespace {
constexpr auto kProfilerNamePyboost = "pyboost";
}

void DeviceOpRunTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeDeviceTask,
                                     context_->op_run_info()->base_op_run_info.op_name, false);
  MS_EXCEPTION_IF_NULL(run_func_);
  run_func_(context_);
}

void PyBoostDeviceTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeDeviceTask,
                                     kProfilerNamePyboost, false);
  if (run_func_) {
    run_func_();
  } else {
    MS_LOG(EXCEPTION) << "No run function!";
  }
}

void PassthroughDeviceTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeDeviceTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  run_func_();
}
}  // namespace runtime
}  // namespace mindspore
