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

#include "pipeline/pynative/forward/do_cast.h"
#include <memory>
#include <utility>
#include <algorithm>
#include "mindspore/core/ops/array_ops.h"
#include "pipeline/pynative/pynative_utils.h"
#include "include/common/utils/stub_tensor.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pynative {
namespace {
template <typename S>
ValuePtr Cast(S in, const TypeId &dst_type_id) {
  switch (dst_type_id) {
    case kNumberTypeInt32:
      return std::make_shared<tensor::Tensor>(static_cast<int>(in), kInt32);
    case kNumberTypeFloat16:
      return std::make_shared<tensor::Tensor>(static_cast<float16>(in), kFloat16);
    case kNumberTypeFloat32:
      return std::make_shared<tensor::Tensor>(static_cast<float>(in), kFloat32);
    case kNumberTypeBool:
      return std::make_shared<tensor::Tensor>(static_cast<bool>(in), kBool);
    case kNumberTypeInt64:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(in), kInt64);
    case kNumberTypeFloat64:
      return std::make_shared<tensor::Tensor>(static_cast<double>(in), kFloat64);
    case kNumberTypeInt16:
      return std::make_shared<tensor::Tensor>(static_cast<int16_t>(in), kInt16);
    case kNumberTypeInt8:
      return std::make_shared<tensor::Tensor>(static_cast<int8_t>(in), kInt8);
    case kNumberTypeUInt64:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(in), kUInt64);
    case kNumberTypeUInt32:
      return std::make_shared<tensor::Tensor>(static_cast<uint32_t>(in), kUInt32);
    case kNumberTypeUInt16:
      return std::make_shared<tensor::Tensor>(static_cast<uint16_t>(in), kUInt16);
    case kNumberTypeUInt8:
      return std::make_shared<tensor::Tensor>(static_cast<uint8_t>(in), kUInt8);
    case kNumberTypeBFloat16:
      return std::make_shared<tensor::Tensor>(static_cast<bfloat16>(in), kBFloat16);
    default:
      MS_LOG(DEBUG) << "Not support cast to dst type: " << TypeIdToType(dst_type_id)->ToString();
      return nullptr;
  }
}

// This function is used to convert scalar value to another scalar value with destination data type.
// The scope of scalar type includes common data types, such as `FP64`, `FP32`, `FP16, `Int64`, `Int32`, ...
// The following sort is based on the hot spots of the data type.
ValuePtr ScalarToDstDtypeValue(const ValuePtr &src_value, const TypeId &dst_type_id) {
  MS_EXCEPTION_IF_NULL(src_value);
  if (src_value->isa<Int64Imm>()) {
    const auto &int64_v = src_value->cast<Int64ImmPtr>();
    return Cast<int64_t>(int64_v->value(), dst_type_id);
  } else if (src_value->isa<FP32Imm>()) {
    const auto &fp32_v = src_value->cast<FP32ImmPtr>();
    return Cast<float>(fp32_v->value(), dst_type_id);
  } else if (src_value->isa<Int32Imm>()) {
    const auto &int32_v = src_value->cast<Int32ImmPtr>();
    return Cast<int32_t>(int32_v->value(), dst_type_id);
  } else if (src_value->isa<FP64Imm>()) {
    const auto &fp64_v = src_value->cast<FP64ImmPtr>();
    return Cast<double>(fp64_v->value(), dst_type_id);
  } else if (src_value->isa<BoolImm>()) {
    const auto &bool_v = src_value->cast<BoolImmPtr>();
    return Cast<bool>(bool_v->value(), dst_type_id);
  } else if (src_value->isa<Int16Imm>()) {
    const auto &int16_v = src_value->cast<Int16ImmPtr>();
    return Cast<int16_t>(int16_v->value(), dst_type_id);
  } else {
    MS_LOG(DEBUG) << "Now, the value [" << src_value->ToString() << "] is not supported to cast directly.";
    return nullptr;
  }
}
}  // namespace

void CastOperation::DoCast(const FrontendOpRunInfoPtr &op_run_info) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeCast,
                                     op_run_info->base_op_run_info.op_name, true);
  // Mixed precision conversion tensors which has cast dtype
  SetTensorMixPrecisionCast(op_run_info);
  // Implicit transform
  SetImplicitCast(op_run_info);
}

void CastOperation::ClearRes() {
  implicit_cast_map_.clear();
  type_prim_cache_.clear();
}

bool CastOperation::IsValueTypeInvalid(const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  return !v->isa<tensor::Tensor>() && !v->isa<tensor::CSRTensor>() && !v->isa<IntegerImm>() && !v->isa<FloatImm>() &&
         !v->isa<BoolImm>();
}

void CastOperation::GetDstType(const FrontendOpRunInfoPtr &op_run_info,
                               const mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                               mindspore::HashMap<SignatureEnumDType, TypeId> *dst_type) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  constexpr size_t index_size = 2;
  for (auto it = type_indexes.begin(); it != type_indexes.end(); (void)++it) {
    const auto &type = it->first;
    const auto &indexes = it->second;
    if (type == SignatureEnumDType::kDTypeEmptyDefaultValue || indexes.size() < index_size) {
      continue;
    }
    size_t priority = 0;
    TypeId max_type = TypeId::kTypeUnknown;
    bool has_scalar_float32 = false;
    bool has_scalar_int64 = false;
    bool has_tensor_int8 = false;
    // Find the maximum priority of the same dtype
    for (size_t index : indexes) {
      if (index >= op_run_info->none_init_inputs_num) {
        MS_LOG(EXCEPTION) << "The index " << index << " exceeds the size of none_init_inputs_num "
                          << op_run_info->none_init_inputs_num;
      }
      const auto &v = op_run_info->op_grad_info->input_value[index];
      if (v->isa<FloatImm>()) {
        has_scalar_float32 = true;
      }
      if (!v->isa<BoolImm>() && v->isa<IntegerImm>()) {
        has_scalar_int64 = true;
      }
      if (v->isa<tensor::Tensor>()) {
        auto arg = v->cast<tensor::TensorPtr>();
        TypeId arg_type_id = arg->data_type();
        auto type_priority = prim::type_map.find(arg_type_id);
        if (type_priority == prim::type_map.end()) {
          continue;
        }
        if (arg_type_id == kNumberTypeInt8) {
          has_tensor_int8 = true;
        }
        if (type_priority->second > priority) {
          max_type = type_priority->first;
          priority = type_priority->second;
        }
      }
    }
    max_type = JudgeMaxType(max_type, has_scalar_float32, has_scalar_int64, has_tensor_int8);
    MS_EXCEPTION_IF_NULL(dst_type);
    (void)dst_type->emplace(std::make_pair(type, max_type));
  }
}

ValuePtr CastOperation::DoNormalCast(const FrontendOpRunInfoPtr &cast_run_info, const ValuePtr &v,
                                     const TypeId &type_id) const {
  MS_EXCEPTION_IF_NULL(v);
  // Step 1: Cast scalar value to another scalar value with destination data type.
  // It is used to avoid to call `cast infer value function` or launch cast op to backend.
  ValuePtr dst_value = ScalarToDstDtypeValue(v, type_id);
  if (dst_value != nullptr) {
    MS_LOG(DEBUG) << "Source value: " << v->ToString() << " cast to value: " << dst_value->ToString();
    cast_run_info->real_out = dst_value;
    return dst_value;
  }

  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    if (type_id == tensor->data_type()) {
      cast_run_info->real_out = v;
      return cast_run_info->real_out;
    }
  }

  MS_EXCEPTION_IF_NULL(cast_run_info);
  constexpr auto input_size = 2;
  cast_run_info->op_grad_info->op_prim = GetPrimByTypeId(type_id);
  PyNativeAlgo::Common::GetConstInputToAttr(
    cast_run_info->op_grad_info->op_prim, cast_run_info->base_op_run_info.op_name,
    cast_run_info->base_op_run_info.device_target, false, &cast_run_info->input_to_attr);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(v);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(GetDstTypeValue(type_id));
  cast_run_info->input_size = input_size;
  PyNativeAlgo::PyParser::PrepareOpGradInfo(cast_run_info);
  PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->RunOpFrontend(cast_run_info);
  return cast_run_info->real_out;
}

ValuePtr CastOperation::DoAutoCast(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, const TypeId &type_id,
                                   const std::string &op_name, size_t index) const {
  MS_EXCEPTION_IF_NULL(v);
  // Step 1: Cast scalar value to another scalar value with destination data type.
  // It is used to avoid to call `cast infer value function` or launch cast op to backend.
  ValuePtr dst_value = ScalarToDstDtypeValue(v, type_id);
  if (dst_value != nullptr) {
    MS_LOG(DEBUG) << "Source value: " << v->ToString() << " cast to value: " << dst_value->ToString();
    return dst_value;
  }
  // When step 1 does not work, creating a cast op to get destination data type value.
  MS_EXCEPTION_IF_NULL(op_run_info);
  constexpr auto input_size = 2;
  const auto &cast_run_info = std::make_shared<FrontendOpRunInfo>();
  auto cast_prim = GetPrimByTypeId(type_id);
  cast_run_info->requires_grad = op_run_info->requires_grad;
  cast_run_info->base_op_run_info.op_name = prim::kPrimCast->name();
  cast_run_info->base_op_run_info.is_mixed_precision_cast = true;
  cast_run_info->base_op_run_info.next_op_name = op_name;
  cast_run_info->base_op_run_info.next_input_index = index;
  cast_run_info->base_op_run_info.use_dynamic_shape_process = op_run_info->base_op_run_info.use_dynamic_shape_process;
  cast_run_info->cell_obj_id = op_run_info->cell_obj_id;
  cast_run_info->base_op_run_info.device_target =
    PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->GetCurrentDeviceTarget(cast_prim);
  bool is_dynamic_shape =
    cast_run_info->base_op_run_info.has_dynamic_output || cast_run_info->base_op_run_info.use_dynamic_shape_process;
  PyNativeAlgo::Common::GetConstInputToAttr(cast_prim, cast_run_info->base_op_run_info.op_name,
                                            cast_run_info->base_op_run_info.device_target, is_dynamic_shape,
                                            &cast_run_info->input_to_attr);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(v);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(GetDstTypeValue(type_id));
  cast_run_info->input_size = input_size;
  cast_run_info->op_grad_info->op_prim = cast_prim;
  PyNativeAlgo::PyParser::PrepareOpGradInfo(cast_run_info);
  PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->RunOpFrontend(cast_run_info);
  return cast_run_info->real_out;
}

ValuePtr CastOperation::DoParamMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info, bool *is_cast,
                                                const ValuePtr &v, const std::string &op_name, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(is_cast);
  MS_EXCEPTION_IF_NULL(v);
  if (op_run_info->mix_type != kNotSet) {
    auto dst_dtype = kFloat16;
    if (op_run_info->mix_type == kFP32) {
      dst_dtype = kFloat32;
    } else if (op_run_info->mix_type == kBF16) {
      dst_dtype = kBFloat16;
    }
    const auto &tensor = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto source_dtype = tensor->Dtype();
    if (source_dtype != nullptr && (IsSubType(source_dtype, kFloat) || IsSubType(source_dtype, kBFloat)) &&
        *source_dtype != *dst_dtype) {
      MS_LOG(DEBUG) << "MixPrecision cast for " << op_run_info->base_op_run_info.op_name << " " << index
                    << "th input, and to type " << dst_dtype->ToString();
      *is_cast = true;
      return DoAutoCast(op_run_info, tensor, dst_dtype->type_id(), op_name, index);
    }
  }
  return v;
}

ValuePtr CastOperation::DoParamMixPrecisionCastTuple(const FrontendOpRunInfoPtr &op_run_info, bool *is_cast,
                                                     const ValueSequencePtr &value_seq, const std::string &op_name,
                                                     size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(is_cast);
  MS_EXCEPTION_IF_NULL(value_seq);
  size_t tuple_size = value_seq->size();
  const auto &value_tuple = value_seq->value();
  ValuePtrList result(tuple_size, nullptr);
  for (size_t i = 0; i < tuple_size; i++) {
    if (value_tuple[i]->isa<tensor::MetaTensor>()) {
      MS_LOG(DEBUG) << "Call cast for item " << i;
      result[i] = DoParamMixPrecisionCast(op_run_info, is_cast, value_tuple[i], op_name, index);
    } else if (value_tuple[i]->isa<ValueSequence>()) {
      result[i] =
        DoParamMixPrecisionCastTuple(op_run_info, is_cast, value_tuple[i]->cast<ValueSequencePtr>(), op_name, index);
    } else {
      result[i] = value_tuple[i];
    }
  }
  if (value_seq->isa<ValueList>()) {
    return std::make_shared<ValueList>(result);
  } else {
    return std::make_shared<ValueTuple>(result);
  }
}

void CastOperation::DoSignatureCast(const FrontendOpRunInfoPtr &op_run_info,
                                    const mindspore::HashMap<SignatureEnumDType, TypeId> &dst_type,
                                    const std::vector<SignatureEnumDType> &dtypes) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info->op_prim);
  const auto &signature = op_run_info->signatures;
  auto &input_args = op_run_info->op_grad_info->input_value;
  size_t input_args_size = input_args.size();
  if (dtypes.size() > input_args_size) {
    MS_LOG(EXCEPTION) << "Signature dtypes size[" << dtypes << "] is greater than input_args_size[" << input_args_size
                      << "].";
  }
  for (size_t i = 0; i < dtypes.size(); ++i) {
    // No need to implicit cast if no dtype.
    if (dtypes.empty() || dtypes[i] == SignatureEnumDType::kDTypeEmptyDefaultValue) {
      continue;
    }
    auto it = dst_type.find(dtypes[i]);
    if (it == dst_type.end() || it->second == kTypeUnknown) {
      continue;
    }
    const auto &v = input_args[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      if (i >= signature.size()) {
        MS_EXCEPTION(ValueError) << "Signature size is not equal to index, signature size " << signature.size()
                                 << ", index " << i;
      }
      sig = signature[i].rw;
    }
    TypeId arg_type_id = kTypeUnknown;
    if (v->isa<tensor::MetaTensor>()) {
      const auto &arg = v->cast<tensor::MetaTensorPtr>();
      arg_type_id = arg->data_type();
    }
    // Implicit cast
    bool is_same_type = false;
    if (arg_type_id != kTypeUnknown) {
      is_same_type = (prim::type_map.find(arg_type_id) == prim::type_map.end() || arg_type_id == it->second);
    }
    if (sig == SignatureEnumRW::kRWWrite && arg_type_id != kTypeUnknown && !is_same_type) {
      prim::RaiseExceptionForConvertRefDtype(op_run_info->op_grad_info->op_prim, TypeIdToMsTypeStr(arg_type_id),
                                             TypeIdToMsTypeStr(it->second), i);
    }
    if (is_same_type) {
      continue;
    }

    if (IsValueTypeInvalid(v)) {
      std::string type_str = v->type() == nullptr ? "None, value is \"" + v->ToString() + "\"" : v->type()->ToString();
      MS_EXCEPTION(TypeError) << "For '" << op_run_info->op_grad_info->op_prim->name() << "', the " << (i + 1)
                              << "th input " << signature[i].name << " can not be implicitly converted. "
                              << "Its type is " << type_str << ". Only support Tensor or Scalar.";
    }
    MS_LOG(DEBUG) << "Implicit cast for " << op_run_info->base_op_run_info.op_name << " " << i
                  << "th input, and to type " << TypeIdToType(it->second)->ToString();
    input_args[i] = DoAutoCast(op_run_info, v, it->second, op_run_info->base_op_run_info.op_name, i);
  }
}

void CastOperation::SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (op_run_info->async_status.disable_mix_precision) {
    // Pure function running, mix precision cast is disable, or cell not set mix precision
    MS_LOG(DEBUG) << "No mix precision for " << op_run_info->base_op_run_info.op_name;
    return;
  }
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info->op_prim);
  const auto &signature = op_run_info->signatures;
  for (size_t i = 0; i < op_run_info->none_init_inputs_num; i++) {
    const auto &v = op_run_info->op_grad_info->input_value[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      if (i >= signature.size()) {
        MS_EXCEPTION(ValueError) << "Signature size is not equal to index, signature size " << signature.size()
                                 << ", index " << i;
      }
      sig = signature[i].rw;
    }
    // mix precision for non param
    bool is_cast = false;
    ValuePtr cast_output = nullptr;
    if (v->isa<tensor::MetaTensor>()) {
      auto meta_tensor = v->cast<tensor::MetaTensorPtr>();
      if (meta_tensor && meta_tensor->is_parameter()) {
        // If parameter write(not kRWRead), no need cast
        if (sig != SignatureEnumRW::kRWRead) {
          continue;
        }
      }
      cast_output = DoParamMixPrecisionCast(op_run_info, &is_cast, v, op_run_info->op_grad_info->op_prim->name(), i);
    } else if (v->isa<ValueSequence>()) {
      // mix precision for tuple inputs
      cast_output = DoParamMixPrecisionCastTuple(op_run_info, &is_cast, v->cast<ValueSequencePtr>(),
                                                 op_run_info->op_grad_info->op_prim->name(), i);
    }
    if (is_cast) {
      MS_EXCEPTION_IF_NULL(cast_output);
      op_run_info->op_grad_info->input_value[i] = cast_output;
    }
  }
}

void CastOperation::SetImplicitCast(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  const auto &it = implicit_cast_map_.find(prim->name());
  if (it == implicit_cast_map_.end()) {
    std::vector<SignatureEnumDType> dtypes;
    bool has_dtype_sig = GetSignatureType(op_run_info->signatures, &dtypes);
    if (!has_dtype_sig) {
      PrimSignature sig_value{has_dtype_sig, {}, {}};
      implicit_cast_map_[prim->name()] = sig_value;
      return;
    }
    const auto &signature = op_run_info->signatures;
    auto sig_size = signature.size();
    // Ignore monad signature
    for (const auto &sig : signature) {
      if (sig.default_value != nullptr && sig.default_value->isa<Monad>()) {
        --sig_size;
      }
    }
    if (sig_size > 0 && sig_size != op_run_info->none_init_inputs_num) {
      MS_EXCEPTION(ValueError) << op_run_info->base_op_run_info.op_name << " inputs number "
                               << op_run_info->none_init_inputs_num << " does not match the requires "
                               << "signature size " << sig_size;
    }
    mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> type_indexes;
    mindspore::HashMap<SignatureEnumDType, TypeId> dst_type;
    GetTypeIndex(dtypes, &type_indexes);
    GetDstType(op_run_info, type_indexes, &dst_type);
    DoSignatureCast(op_run_info, dst_type, dtypes);
    PrimSignature sig_value{has_dtype_sig, dtypes, type_indexes};
    implicit_cast_map_[prim->name()] = sig_value;
  } else {
    if (!it->second.has_dtype_sig) {
      MS_LOG(DEBUG) << op_run_info->base_op_run_info.op_name << " have no dtype sig";
      return;
    }
    MS_LOG(DEBUG) << "Do signature for " << op_run_info->base_op_run_info.op_name << " with cache";
    mindspore::HashMap<SignatureEnumDType, TypeId> dst_type;
    GetDstType(op_run_info, it->second.type_indexes, &dst_type);
    DoSignatureCast(op_run_info, dst_type, it->second.dtypes);
  }
}
}  // namespace pynative
}  // namespace mindspore
