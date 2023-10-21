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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CREATE_NODE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CREATE_NODE_HELPER_H_
#include <string>
#include <memory>

#include "ir/anf.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"

namespace mindspore::opt {
class CreateNodeHelper {
 public:
  CreateNodeHelper() = default;
  ~CreateNodeHelper() = default;
  static AnfNodePtr CreateNodeWithCheck(const AnfNodePtr &node, bool is_ascend_mindir = false);

 private:
  static CNodePtr ConvertToTargetOp(const CNodePtr &origin_op, OpAdaptationInfo *op_adaptation_info);
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CREATE_NODE_HELPER_H_
