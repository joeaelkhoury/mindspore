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

#include "plugin/device/ascend/optimizer/mindir/aicpu_lib_select.h"
#include <set>
#include <string>
#include "include/common/utils/utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const AnfNodePtr AICpuLibSelectPass::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  static const std::set<std::string> kAICpuOpNames = {kDropoutGenMaskOpName,
                                                      kEnvironCreateOpName,
                                                      kEnvironSetOpName,
                                                      kEnvironGetOpName,
                                                      kEnvironDestroyAllOpName,
                                                      kPriorityReplayBufferCreate,
                                                      kPriorityReplayBufferPush,
                                                      kPriorityReplayBufferSample,
                                                      kPriorityReplayBufferUpdate,
                                                      kPriorityReplayBufferDestroy,
                                                      kReservoirReplayBufferCreate,
                                                      kReservoirReplayBufferPush,
                                                      kReservoirReplayBufferSample,
                                                      kReservoirReplayBufferDestroy,
                                                      kGatherDGradV2OpName,
                                                      kConcatOffsetOpName,
                                                      kSliceGradOpName,
                                                      kRandomShuffleOpName,
                                                      kRangeOpName};
  static const std::set<std::string> kMigrateAicpuKernelOps = {mindspore::kAdaptiveAvgPool2dOpName,
                                                               mindspore::kAdaptiveAvgPool2dGradOpName,
                                                               mindspore::kCacheSwapTableOpName,
                                                               mindspore::kCol2imOpName,
                                                               mindspore::kCumulativeLogsumexpOpName,
                                                               mindspore::kDataFormatVecPermuteOpName,
                                                               mindspore::kFillOpName,
                                                               mindspore::kLogMatrixDeterminantOpName,
                                                               mindspore::kMatrixSolveLsOpName,
                                                               mindspore::kMaskedSelectOpName,
                                                               mindspore::kMaskedSelectGradOpName,
                                                               mindspore::kMedianOpName,
                                                               mindspore::kMedianGradOpName,
                                                               mindspore::kNMSWithMaskOpName,
                                                               mindspore::kReduceSumOpName,
                                                               mindspore::kFFTWithSizeOpName,
                                                               mindspore::kHistogramDOpName,
                                                               mindspore::kIm2colOpName,
                                                               mindspore::kIsInfOpName,
                                                               mindspore::kIsNanOpName,
                                                               mindspore::kMatrixDeterminantOpName,
                                                               mindspore::kMatrixLogarithmOpName,
                                                               mindspore::kMatrixSetDiagV3OpName,
                                                               mindspore::kMultinomialOpName,
                                                               mindspore::kNanToNumOpName,
                                                               mindspore::kQrOpName,
                                                               mindspore::kResizeBicubicOpName,
                                                               mindspore::kNuclearNormOpName,
                                                               mindspore::kQuantileOpName,
                                                               mindspore::kSparseSegmentSqrtNOpName,
                                                               mindspore::kUnsortedSegmentProdOpName};
  static const std::string kEnvOpSoNames = "mindspore_aicpu_kernels";
  static const std::string kCpuKernelSoName = "mindspore_cpu_kernels";

  if (!node->isa<CNode>()) {
    return node;
  }
  auto kernel_name = common::AnfAlgo::GetCNodeName(node);
  if (kAICpuOpNames.find(kernel_name) != kAICpuOpNames.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kEnvOpSoNames), node);
  }
  if (kMigrateAicpuKernelOps.find(kernel_name) != kMigrateAicpuKernelOps.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kCpuKernelSoName), node);
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
