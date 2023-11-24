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

#include "ops/npu_clear_float_status_v2.h"
#include <map>
#include <set>
#include <string>
#include "mindspore/core/ops/other_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NPUClearFloatStatusV2InferShape(const PrimitivePtr &,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape())[kShape];
  // dynamic rank
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  // dynamic shape
  if (IsDynamic(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector(input_shape.size(), abstract::Shape::kShapeDimAny));
  }
  const int64_t normal_shape_size = 1;
  const int64_t normal_shape_len = 8;
  if (input_shape.size() != normal_shape_size) {
    MS_EXCEPTION(ValueError) << "Input_x must be a 1-dimensional tensor, but got " << std::to_string(input_shape.size())
                             << "-dimensional tensor.";
  }
  if (input_shape[0] != normal_shape_len) {
    MS_EXCEPTION(ValueError) << "The first dimension of input_x must be 8, but got " << std::to_string(input_shape[0]);
  }
  std::vector<int64_t> output_shape = {normal_shape_len};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr NPUClearFloatStatusV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kInt32};
  TypePtr input_x_type = input_args[0]->GetType();
  (void)types.emplace("input_x", input_x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return kInt32;
}
}  // namespace

MIND_API_OPERATOR_IMPL(NPUClearFloatStatusV2, BaseOperator);
AbstractBasePtr NPUClearFloatStatusV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = NPUClearFloatStatusV2InferType(primitive, input_args);
  auto infer_shape = NPUClearFloatStatusV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGNPUClearFloatStatusV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUClearFloatStatusV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUClearFloatStatusV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUClearFloatStatusV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NPUClearFloatStatusV2, prim::kPrimNPUClearFloatStatusV2, AGNPUClearFloatStatusV2Infer,
                                 false);
}  // namespace ops
}  // namespace mindspore
