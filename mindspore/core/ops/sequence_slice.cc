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

#include "ops/sequence_slice.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
int64_t SequenceSliceGetValue(const std::string &prim_name, const std::string &attr_name, const AbstractBasePtr &abs) {
  auto build_type = abs->BuildType();
  auto build_value = abs->BuildValue();
  if (build_type == kInt32) {
    return GetValue<int32_t>(build_value);
  } else if (build_type == kInt64) {
    return GetValue<int64_t>(build_value);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the type of '" << attr_name
                            << "' should be int32, int64 but got: " << abs->BuildType()->ToString();
  }
}
AbstractBasePtr SliceInferValue(const abstract::AbstractSequencePtr &seq_abs, int64_t start_v, int64_t end,
                                int64_t step) {
  auto elems = seq_abs->elements();
  int64_t len = static_cast<int64_t>(elems.size());
  int64_t start = start_v;
  abstract::AbstractBasePtrList abs{};
  if (step == 0) {
    MS_EXCEPTION(ValueError) << "For 'SequenceSlice', step cannot be 0.";
  } else if (step > 0) {
    if (start <= -len) {
      start = 0;
    } else if (start < 0) {
      start += len;
    }
    if (end > len) {
      end = len;
    } else if (end > -len && end < 0) {
      end += len;
    }
    if (start >= end) {
      return std::make_shared<abstract::AbstractTuple>(abs);
    }
    for (int64_t i = start; i < end; i += step) {
      abs.push_back(std::make_shared<abstract::AbstractScalar>(elems[static_cast<size_t>(i)]->BuildValue(),
                                                               elems[static_cast<size_t>(i)]->BuildType()));
    }
    return std::make_shared<abstract::AbstractTuple>(abs);
  } else {
    // when step < 0
    if (start >= len) {
      start = -1;
    } else if (start >= 0 && start < len) {
      start -= len;
    }
    if (end < -len) {
      end = -1 - len;
    } else if (end >= 0 && end < len) {
      end -= len;
    }
    if (start <= end) {
      return std::make_shared<abstract::AbstractTuple>(abs);
    }
    for (int64_t i = start; i > end; i += step) {
      abs.push_back(std::make_shared<abstract::AbstractScalar>(elems[static_cast<size_t>(i + len)]->BuildValue(),
                                                               elems[static_cast<size_t>(i + len)]->BuildType()));
    }
    return std::make_shared<abstract::AbstractTuple>(abs);
  }
}
AbstractBasePtr SliceInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_num = 4;
  constexpr size_t seq_index = 0;
  constexpr size_t start_index = 1;
  constexpr size_t end_index = 2;
  constexpr size_t step_index = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto first_abs = input_args[seq_index];
  MS_EXCEPTION_IF_NULL(first_abs);
  if (!first_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the first input should be tuple or list but got: " << first_abs->ToString();
  }
  auto seq_abs = first_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    // If the length of input sequence is dynamic length, the length of sliced sequence should also be dynamic length.
    return seq_abs->Clone();
  }
  if (seq_abs->size() != 0 && CheckAndConvertUtils::CheckContainNestedOrIrregularSequence(input_args)) {
    // Sequence ops with nested or irregular sequence input should be convert to PyExecute node.
    return std::make_shared<abstract::AbstractAny>();
  }
  auto start_abs = input_args[start_index];
  MS_EXCEPTION_IF_NULL(start_abs);
  auto end_abs = input_args[end_index];
  MS_EXCEPTION_IF_NULL(end_abs);
  auto step_abs = input_args[step_index];
  MS_EXCEPTION_IF_NULL(step_abs);

  // whether element is tensor with a scalar or a scalar
  for (auto elem : seq_abs->elements()) {
    if (!(elem->isa<abstract::AbstractScalar>() ||
          (elem->isa<abstract::AbstractTensor>() &&
           elem->cast<abstract::AbstractTensorPtr>()->shape()->shape().size() == 0) ||
          (elem->isa<abstract::AbstractTensor>() &&
           elem->cast<abstract::AbstractTensorPtr>()->shape()->shape().size() == 1 &&
           elem->cast<abstract::AbstractTensorPtr>()->shape()->shape()[0] == 1))) {
      return std::make_shared<abstract::AbstractAny>();
    }
  }

  // all value is known
  if (start_abs->BuildValue() != kValueAny && end_abs->BuildValue() != kValueAny &&
      step_abs->BuildValue() != kValueAny) {
    int64_t start_v;
    int64_t end_v;
    int64_t step_v;
    const std::string start_str = "start";
    const std::string end_str = "end";
    const std::string step_str = "step";
    start_v = SequenceSliceGetValue(prim_name, start_str, start_abs);
    end_v = SequenceSliceGetValue(prim_name, end_str, end_abs);
    step_v = SequenceSliceGetValue(prim_name, step_str, step_abs);
    return SliceInferValue(seq_abs, start_v, end_v, step_v);
  }
  auto ret = seq_abs->Clone()->cast<abstract::AbstractSequencePtr>();
  ret->CheckAndConvertToDynamicLenSequence();
  return ret;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SequenceSlice, BaseOperator);
class SequenceSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferInner(primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2, 3}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceSlice, prim::kPrimSequenceSlice, SequenceSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
