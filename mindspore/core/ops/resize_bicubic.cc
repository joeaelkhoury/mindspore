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
#include "ops/resize_bicubic.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void AttrTest(bool a, bool b) {
  if (a && b) {
    MS_EXCEPTION(ValueError) << "The half_pixel_centers must be false when align_corners is true "
                             << ", but half_pixel_centers got True";
  }
}

abstract::ShapePtr ResizeBicubicInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int64_t> output_shape(4, -1);
  const int64_t shape0_dim = 4;
  auto shape0 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (!IsDynamicRank(shape0) && shape0.size() != shape0_dim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the images tensor must be a 4-D tensor. But got "
                             << shape0.size() << "-D";
    constexpr int64_t indexid3 = 3;
    output_shape[0] = shape0[0];
    output_shape[indexid3] = shape0[indexid3];
  }
  auto shape1 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (shape1.size() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size tensor must be a 1-D tensor. But got "
                             << shape1.size() << "-D";
  }
  constexpr int64_t calnum2 = 2;
  if (!IsDynamic(shape1) && shape1[0] != calnum2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size shape must be 2. But got " << shape1[0];
  }
  auto align_corners_ptr = primitive->GetAttr("align_corners");
  bool align_corners = GetValue<bool>(align_corners_ptr);
  auto half_pixel_centers_ptr = primitive->GetAttr("half_pixel_centers");
  bool half_pixel_centers = GetValue<bool>(half_pixel_centers_ptr);
  AttrTest(align_corners, half_pixel_centers);
  if (!input_args[1]->BuildValue()->isa<AnyValue>() && !input_args[1]->BuildValue()->isa<None>()) {
    auto input1_value = input_args[1]->BuildValue();
    if (!input1_value->isa<tensor::Tensor>()) {
      MS_LOG(EXCEPTION) << "For ResizeArea, the inputs[1] must be a tensor, but got: " << input1_value->ToString()
                        << ".";
    }
    auto input1_shape_ptr = static_cast<int32_t *>(input1_value->cast<tensor::TensorPtr>()->data_c());
    output_shape[kInputIndex1] = input1_shape_ptr[kInputIndex0];
    output_shape[kInputIndex2] = input1_shape_ptr[kInputIndex1];
  }
  return std::make_shared<abstract::Shape>(output_shape);
}
TypePtr ResizeBicubicInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid0_types = {kInt8, kUInt8, kInt16, kUInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid1_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_args[0]->BuildType(), valid0_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[1]->BuildType(), valid1_types, prim_name);
  return kFloat32;
}
}  // namespace
MIND_API_OPERATOR_IMPL(ResizeBicubic, BaseOperator);
void ResizeBicubic::set_align_corners(const bool align_corners) {
  (void)this->AddAttr("align_corners", api::MakeValue(align_corners));
}
void ResizeBicubic::set_half_pixel_centers(const bool half_pixel_centers) {
  (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers));
}

bool ResizeBicubic::get_align_corners() const {
  auto value_ptr = GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}
bool ResizeBicubic::get_half_pixel_centers() const {
  auto value_ptr = GetAttr("half_pixel_centers");
  return GetValue<bool>(value_ptr);
}

void ResizeBicubic::Init(const bool align_corners, const bool half_pixel_centers) {
  this->set_align_corners(align_corners);
  this->set_half_pixel_centers(half_pixel_centers);
}

AbstractBasePtr ResizeBicubicInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ResizeBicubicInferType(primitive, input_args);
  auto infer_shape = ResizeBicubicInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ResizeBicubic, prim::kPrimResizeBicubic, ResizeBicubicInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
