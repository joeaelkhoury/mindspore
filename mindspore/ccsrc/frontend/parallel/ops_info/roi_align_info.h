/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ROI_ALIGN_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ROI_ALIGN_INFO_H_

#include <memory>
#include <vector>
#include <string>

#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
class ROIAlignInfo : public OperatorInfo {
 public:
  ROIAlignInfo(const std::string &name, const Shapes &input_shape, const Shapes &output_shape,
               const PrimitiveAttrs &attrs)
      : OperatorInfo(name, input_shape, output_shape, attrs, std::make_shared<ROIAlignCost>()) {}
  ~ROIAlignInfo() = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
  Status InitForCostModel(const StrategyPtr &strategy, const StrategyPtr &out_strategy) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status CheckStrategyForDynamicShape(const StrategyPtr &strategy) override;

 private:
  Status InferBias();
  Status InferGroup();
  Status ComputeReplaceGraph(const CNodePtr &cnode);
  std::vector<int64_t> CreateRangeVector(int64_t upper_bound) const;

  int64_t features_slice_size_ = 0;
  int64_t rois_slice_size_ = 0;
  int64_t bias_ = 0;
  Group group_;
  OperatorAttrs roi_align_attrs;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ROI_ALIGN_INFO_H_
