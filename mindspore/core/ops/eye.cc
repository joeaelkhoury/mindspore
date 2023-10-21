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

#include "ops/eye.h"

#include <complex>
#include <memory>
#include <set>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void ImpleEye(int64_t num_n_, int64_t num_m_, void *target) {
  MS_EXCEPTION_IF_NULL(target);
  int64_t num_min = (num_n_ > num_m_) ? num_m_ : num_n_;
  auto result_data = static_cast<T *>(target);
  T num = static_cast<T>(1);
  for (int64_t i = 0; i < num_min; i++) {
    result_data[(num_m_ + 1) * i] = static_cast<T>(num);
  }
}

abstract::ShapePtr EyeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto n_ptr = input_args[0]->BuildValue();
  auto m_ptr = input_args[1]->BuildValue();
  auto n_v = GetValue<int64_t>(n_ptr);
  auto m_v = GetValue<int64_t>(m_ptr);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::Check("n", n_v, kGreaterEqual, 0, prim_name);
  CheckAndConvertUtils::Check("m", m_v, kGreaterEqual, 0, prim_name);
  std::vector<int64_t> state_shape = {n_v, m_v};
  return std::make_shared<abstract::Shape>(state_shape);
}

void EyeCheck(const std::vector<AbstractBasePtr> &input_args) {
  if (!input_args[0]->isa<abstract::AbstractScalar>()) {
    MS_EXCEPTION(TypeError) << "For Eye, 'n' must be int, but got ValueAny!";
  }
  if (!input_args[1]->isa<abstract::AbstractScalar>()) {
    MS_EXCEPTION(TypeError) << "For Eye, 'm' must be int, but got ValueAny!";
  }
  auto n_ptr_ = input_args[0]->BuildValue();
  auto m_ptr_ = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(n_ptr_);
  MS_EXCEPTION_IF_NULL(m_ptr_);
  if (!n_ptr_->isa<Int64Imm>() && !n_ptr_->isa<Int32Imm>()) {
    MS_EXCEPTION(TypeError) << "For Eye, the dtype of n is invalid!";
  }
  if (!m_ptr_->isa<Int64Imm>() && !m_ptr_->isa<Int32Imm>()) {
    MS_EXCEPTION(TypeError) << "For Eye, the dtype of m is invalid!";
  }
  auto dtype_value_c = input_args[2]->BuildValue();
  MS_EXCEPTION_IF_NULL(dtype_value_c);
  if (!dtype_value_c->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "For Eye, the dtype of Eye is invalid!";
  }
}

TypePtr EyeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  EyeCheck(input_args);
  auto dtype_value = input_args[2]->BuildValue();
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "For Eye, the dtype of Eye is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto dtype_ret = CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_types, prim_name);
  return dtype_ret;
}

ValuePtr EyeInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  if (input_args.empty()) {
    return nullptr;
  }
  auto n_pt = input_args[0]->BuildValue();
  auto m_pt = input_args[1]->BuildValue();
  auto dtype_value_ = input_args[2]->BuildValue();
  if (n_pt == nullptr || m_pt == nullptr || dtype_value_ == nullptr) {
    return nullptr;
  }
  EyeCheck(input_args);
  auto num_n_ = GetValue<int64_t>(n_pt);
  auto num_m_ = GetValue<int64_t>(m_pt);
  CheckAndConvertUtils::Check("n", num_n_, kGreaterEqual, 0, prim_name);
  CheckAndConvertUtils::Check("m", num_m_, kGreaterEqual, 0, prim_name);
  auto type_id = dtype_value_->cast<TypePtr>()->type_id();
  auto shape = EyeInferShape(prim, input_args);
  auto result_tensor = std::make_shared<tensor::Tensor>(type_id, shape->shape());
  auto result_datac = result_tensor->data_c();
  switch (type_id) {
    case kNumberTypeInt8: {
      ImpleEye<int8_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeInt16: {
      ImpleEye<int16_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeInt32: {
      ImpleEye<int32_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeInt64: {
      ImpleEye<int64_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeUInt8: {
      ImpleEye<uint8_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeUInt16: {
      ImpleEye<uint16_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeUInt32: {
      ImpleEye<uint32_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeUInt64: {
      ImpleEye<uint64_t>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeFloat16: {
      ImpleEye<float16>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeFloat32: {
      ImpleEye<float>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeFloat64: {
      ImpleEye<double>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeComplex64: {
      ImpleEye<std::complex<float>>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeComplex128: {
      ImpleEye<std::complex<double>>(num_n_, num_m_, result_datac);
      break;
    }
    case kNumberTypeBool: {
      ImpleEye<bool>(num_n_, num_m_, result_datac);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "Eye unsupported current data type . ";
    }
  }
  return result_tensor;
}
}  // namespace

AbstractBasePtr EyeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, op_name);
  auto infer_type = EyeInferType(primitive, input_args);
  auto infer_shape = EyeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Eye, BaseOperator);
class MIND_API AGEyeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EyeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EyeInferType(primitive, input_args);
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EyeInferValue(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return EyeInfer(engine, primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {0, 1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Eye, prim::kPrimEye, AGEyeInfer, true);
}  // namespace ops
}  // namespace mindspore
