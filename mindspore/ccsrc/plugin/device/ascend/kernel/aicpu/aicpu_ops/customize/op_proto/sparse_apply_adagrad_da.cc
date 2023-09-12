/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "inc/sparse_apply_adagrad_da.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
namespace {
const int64_t kMaxDimNum = 8;

// Set ref port for ref input without ref output
void SetRefInput(Operator &op, const string &input_name) {
  const OpDescPtr opDesc = OpDescUtils::GetOpDescFromOperator(op);
  int inputIndex = opDesc->GetInputIndexByName(input_name);
  if (inputIndex < 0) {
    OP_LOGE(TbeGetName(op).c_str(), "SetRefInput failed, can not find input %s!", input_name.c_str());
  }
  GeTensorDesc inputTensorDesc = opDesc->GetInputDesc(inputIndex);
  vector<uint32_t> index = {static_cast<uint32_t>(inputIndex)};
  inputTensorDesc.SetRefPortByIndex(index);
  if (opDesc->UpdateInputDesc(static_cast<uint32_t>(inputIndex), inputTensorDesc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "UpdateOutputDesc failed, maybe input name error!");
  }
}

// Check input and attr of the input tensor description.
bool ApplyVerifyFunc(const ge::Operator &op, const std::vector<std::string> &inputTensorList,
                     const std::vector<std::string> &inputScalarList) {
  // check shape of Tensor
  auto var_dims = op.GetInputDescByName(inputTensorList[0].c_str()).GetShape().GetDims();
  if (var_dims.size() > kMaxDimNum) {
    OP_LOGE(TbeGetName(op).c_str(), "Var only support less than 8 dims!");
    return false;
  }
  if (IsUnknown(var_dims)) {
    OP_LOGW(TbeGetName(op).c_str(), "this is dynamic shape, will exit ApplyVerifyFunc");
    return true;
  }
  for (std::size_t i = 1; i < inputTensorList.size(); i++) {
    auto tmp_dims = op.GetInputDescByName(inputTensorList[i].c_str()).GetShape().GetDims();
    if (IsUnknown(tmp_dims)) {
      OP_LOGW(TbeGetName(op).c_str(), "this is dynamic shape, will continue ApplyVerifyFunc");
      continue;
    }
    if (tmp_dims != var_dims) {
      OP_LOGE(TbeGetName(op).c_str(), "the shape of %s must equal with %s", inputTensorList[i].c_str(),
              inputTensorList[0].c_str());
      return false;
    }
  }

  // check shape of Scalar
  for (std::size_t j = 0; j < inputScalarList.size(); j++) {
    auto scalar_dims = op.GetInputDescByName(inputScalarList[j].c_str()).GetShape().GetDims();
    if (scalar_dims.size() > 1) {
      OP_LOGE(TbeGetName(op).c_str(), "The input %s must be scalar!", inputScalarList[j].c_str());
      return false;
    }
  }
  return true;
}
}  // namespace
// ----------------SparseApplyAdagradDA Op-------------------
CUST_IMPLEMT_VERIFIER(SparseApplyAdagradDA, SparseApplyAdagradDAVerify) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter SparseApplyAdagradDA proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("grad_accum");
  inputTensorList.push_back("grad_square_accum");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");
  inputScalarList.push_back("global_step");

  if (!ApplyVerifyFunc(op, inputTensorList, inputScalarList)) {
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  if (WithRank(op.GetInputDesc(4), 1, indices_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input indices must be 1-D.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(SparseApplyAdagradDAInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter SparseApplyAdagradDA op_proto inferfunction!");
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "var", "grad_accum", {"var"})) {
    return GRAPH_FAILED;
  }
  SetRefInput(op, "grad_accum");
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "var", "grad_square_accum", {"var"})) {
    return GRAPH_FAILED;
  }
  SetRefInput(op, "grad_square_accum");
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(SparseApplyAdagradDA, SparseApplyAdagradDAInferShape);
CUST_VERIFY_FUNC_REG(SparseApplyAdagradDA, SparseApplyAdagradDAVerify);
// ----------------SparseApplyAdagradDA END-------------------
}  // namespace ge