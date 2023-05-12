/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/ops_info/fillv2_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status FillV2Info::InferAttrs() {
  if (infer_attrs_completed_) {
    return SUCCESS;
  }
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GetAttrs failed.";
    return FAILED;
  }
  ResetInputsShape();
  infer_attrs_completed_ = true;
  return SUCCESS;
}

Status FillV2Info::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy " << strategy->ToString();
    return FAILED;
  }
  return SUCCESS;
}

Status FillV2Info::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  auto strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << name_ << ": Infer device matric failed, inputs_startegy is empty.";
    return FAILED;
  }
  dev_matrix_shape_ = strategies.at(0);
  return SUCCESS;
}

Status FillV2Info::InferTensorMap() {
  TensorMap tensor_map;
  std::vector<Dimensions> strategies = strategy_->GetInputDim();
  auto input_shape_strategy = strategies.at(0);
  auto size = input_shape_strategy.size();
  for (size_t i = 0; i < size; ++i) {
    tensor_map.push_back(SizeToLong(size - i - 1));
  }
  inputs_tensor_map_.push_back(tensor_map);
  inputs_tensor_map_.emplace_back(TensorMap());
  outputs_tensor_map_.push_back(tensor_map);
  return SUCCESS;
}

std::vector<StrategyPtr> FillV2Info::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_.at(0).size(), 1);
  Shape input1_split;
  Shapes splittable_inputs = {input0_split, input1_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy.";
  }
  return sp_vector;
}

void FillV2Info::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto input_shape = inputs_shape_.at(kIndex0);
    std::vector<Dimensions> stra = strategy_->GetInputDim();
    for (size_t i = 0; i < stra[0].size(); i++) {
      input_shape[i] /= stra[0][i];
    }
    auto func_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto val_tensor_node = NewValueNode(MakeValue(std::make_shared<tensor::Tensor>(input_shape)));
    cnode->set_input(kIndex1, val_tensor_node);
  }
}

Status FillV2Info::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  if (mirror_ops_.size() == kSizeOne) {
    // Insert empty mirror op for shape
    (void)mirror_ops_.insert(mirror_ops_.begin(), OperatorVector());
  }
  return SUCCESS;
}

Shape FillV2Info::GetShapeFromTensor(const tensor::TensorPtr &shape_tensor) {
  MS_EXCEPTION_IF_NULL(shape_tensor);
  auto dim = shape_tensor->DataDim();
  if (IntToSize(dim) != kDim1) {
    MS_LOG(EXCEPTION) << name_ << ": The rank of 'input_shape' must be 1, but got rank " << dim;
  }
  auto size = shape_tensor->DataSize();
  if (size <= 0) {
    MS_LOG(EXCEPTION) << name_ << ": The size of 'input_shape' must be greater than 0, but got size " << size;
  }
  auto dtype = shape_tensor->data_type();
  auto data = shape_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data);
  if (dtype == kNumberTypeInt32) {
    auto shape_data = static_cast<int32_t *>(data);
    Shape shape(shape_data, shape_data + size);
    return shape;
  } else if (dtype == kNumberTypeInt64) {
    auto shape_data = static_cast<int64_t *>(data);
    Shape shape(shape_data, shape_data + size);
    return shape;
  }
  MS_LOG(EXCEPTION) << name_ << ": The dtype of 'input_shape' must be int32 or int64, but got type "
                    << TypeIdToString(dtype);
}

void FillV2Info::ResetInputsShape() {
  auto input_value_shape = input_value_[0];
  if (input_value_shape == nullptr) {
    MS_LOG(EXCEPTION) << name_ << ": The value of input 'shape' must be a constant. "
                      << "If you pass this value via construct, try to define its value in __init__";
  }
  MS_EXCEPTION_IF_NULL(input_value_shape);
  if (input_value_shape->isa<tensor::Tensor>()) {
    auto tensor_shape_ptr = GetValue<tensor::TensorPtr>(input_value_shape);
    auto shape = GetShapeFromTensor(tensor_shape_ptr);
    inputs_shape_[0] = shape;
    is_parameter_[0] = false;
    return;
  } else if (input_value_shape->isa<ValueTuple>()) {
    inputs_shape_.insert(inputs_shape_.begin(), GetValue<Shape>(input_value_shape));
    is_parameter_.insert(is_parameter_.begin(), false);
    return;
  }
  MS_LOG(EXCEPTION) << name_ << ": The type of input 'shape' must be Tensor or Tuple, but got "
                    << input_value_shape->type()->ToString();
}

REGISTER(FillV2Info);
}  // namespace parallel
}  // namespace mindspore
