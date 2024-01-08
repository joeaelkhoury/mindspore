/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_PLACEHOLDER_FOR_OP_WITH_OPTIONAL_INPUTS_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_PLACEHOLDER_FOR_OP_WITH_OPTIONAL_INPUTS_H_

#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class InsertPlaceholderForOpWithOptionalInputs : public PatternProcessPass {
 public:
  explicit InsertPlaceholderForOpWithOptionalInputs(bool multigraph = true)
      : PatternProcessPass("add_placeholder_for_op_with_optional_inputs", multigraph) {}
  ~InsertPlaceholderForOpWithOptionalInputs() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_PLACEHOLDER_FOR_OP_WITH_OPTIONAL_INPUTS_H_
