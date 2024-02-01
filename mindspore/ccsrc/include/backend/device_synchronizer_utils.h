/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef aMINDSPORE_CCSRC_INCLUDE_DEVICE_SYNCHRONIZER_UTILS_H
#define aMINDSPORE_CCSRC_INCLUDE_DEVICE_SYNCHRONIZER_UTILS_H

#include "runtime/graph_scheduler/actor/kernel_launch_actor.h"

namespace mindspore {
namespace runtime {
static inline void WaitRuntimeFinishLaunch() { KernelLaunchActor::GetInstance()->Wait(); }
}  // namespace runtime
}  // namespace mindspore
#endif  // aMINDSPORE_CCSRC_INCLUDE_DEVICE_SYNCHRONIZER_UTILS_H
