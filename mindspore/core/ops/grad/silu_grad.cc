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
#include "ops/grad/silu_grad.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(SiLUGrad, BaseOperator);
class SiLUGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 2;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    return input_args[0]->GetShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    auto dout = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 0, kObjectTypeTensorType);
    auto out = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 1, kObjectTypeTensorType);
    (void)abstract::CheckDtypeSame(prim_name, out, dout);
    auto x_type = dout->GetType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!CheckAndConvertUtils::IsTensor(dout)) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }
    return x_type;
  }
};
abstract::AbstractBasePtr SiLUGradInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  SiLUGradInfer silu_grad_infer;
  auto type = silu_grad_infer.InferType(primitive, input_args);
  auto shape = silu_grad_infer.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(SiLUGrad, prim::kPrimSiLUGrad, SiLUGradInfer, false);
}  // namespace ops
}  // namespace mindspore
