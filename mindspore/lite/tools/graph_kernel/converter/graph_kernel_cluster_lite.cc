/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/converter/graph_kernel_cluster_lite.h"

#include <utility>
#include <vector>

#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
std::vector<PrimitivePtr> GraphKernelClusterLite::GetClusterableOpList() {
  std::vector<OpWithLevel> clusterable_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimAdd},
    {kAllTarget, OpLevel_0, prim::kPrimMul},
    {kAllTarget, OpLevel_0, prim::kPrimSub},
    {kAllTarget, OpLevel_0, prim::kPrimSqrt},
    {kAllTarget, OpLevel_0, prim::kPrimRealDiv},
    // ascend device
    {kAscendDevice, OpLevel_0, prim::kPrimMatMul},
    {kAscendDevice, OpLevel_0, prim::kPrimFastGeLU},
    {kAscendDevice, OpLevel_0, prim::kPrimTranspose},
    {kAscendDevice, OpLevel_0, prim::kPrimReshape},
    // cpu device
    {kCPUDevice, OpLevel_0, prim::kPrimLog},
    {kCPUDevice, OpLevel_0, prim::kPrimExp},
    {kCPUDevice, OpLevel_0, prim::kPrimPow},
    {kCPUDevice, OpLevel_0, prim::kPrimNeg},
    {kCPUDevice, OpLevel_0, prim::kPrimRsqrt},
    {kCPUDevice, OpLevel_0, prim::kPrimSin},
    {kCPUDevice, OpLevel_0, prim::kPrimTanh},
    {kCPUDevice, OpLevel_0, prim::kPrimCos},
    {kCPUDevice, OpLevel_0, prim::kPrimGreater},
    {kCPUDevice, OpLevel_0, prim::kPrimGreaterEqual},
    {kCPUDevice, OpLevel_0, prim::kPrimLess},
    {kCPUDevice, OpLevel_0, prim::kPrimLessEqual},
    {kCPUDevice, OpLevel_0, prim::kPrimLogicalAnd},
    {kCPUDevice, OpLevel_0, prim::kPrimLogicalOr},
    {kCPUDevice, OpLevel_0, prim::kPrimLogicalNot},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  return GkUtils::GetValidOps(clusterable_ops_with_level, flags.fusion_ops_level, flags.enable_cluster_ops_only,
                              flags.enable_cluster_ops, flags.disable_cluster_ops);
}

bool GraphKernelClusterLite::IsClusterableOp(const AnfNodePtr &node) {
  if (!GraphKernelCluster::IsClusterableOp(node)) {
    return false;
  }
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  if (device_ == "Ascend") {
    auto type_id = cb->GetOutputInferType(node, 0);
    if (type_id == kNumberTypeInt64) {
      return false;
    }
  }
  // check if the node has dynamic shape
  auto cnode = node->cast<CNodePtr>();
  for (size_t i = 0; i < cnode->size() - 1; i++) {
    if (!cnode->input(i + 1)->isa<Parameter>() && !cnode->input(i + 1)->isa<ValueNode>() &&
        cb->GetInputShape(cnode, i).size() == 0) {
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
