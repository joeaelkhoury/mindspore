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
#include "ops/map_tensor_get_grad.h"
#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MapTensorGetGrad, BaseOperator);
AbstractBasePtr MapTensorGetGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Check number of arguments.
  constexpr int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, kNameMapTensorGetGrad);
  // Check argument abstracts.
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorGetGrad, input_args, kInputIndex0);

  // We skip check other arguments, because grad operations are generated by compiler
  // so we can assume that their are always correct.

  // Grad map tensor has same abstract with the input map tensor.
  return abs_map_tensor->Broaden();
}
REGISTER_PRIMITIVE_EVAL_IMPL(MapTensorGetGrad, prim::kPrimMapTensorGetGrad, MapTensorGetGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
