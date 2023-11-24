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

#include "transform/acl_ir/acl_convert.h"
#include <map>
#include <algorithm>
#include "transform/acl_ir/acl_adapter_info.h"
#include "include/common/utils/convert_utils.h"
#include "transform/acl_ir/acl_helper.h"
#include "ops/op_utils.h"
#include "include/backend/device_address.h"

namespace mindspore::transform {
namespace {
#define AT_ALL_MINDSPORE_TYPE_AND_ACL_DATATYPE_PAIR(_) \
  _(kNumberTypeBool, ACL_BOOL)                         \
  _(kNumberTypeInt, ACL_INT32)                         \
  _(kNumberTypeInt8, ACL_INT8)                         \
  _(kNumberTypeInt16, ACL_INT16)                       \
  _(kNumberTypeInt32, ACL_INT32)                       \
  _(kNumberTypeInt64, ACL_INT64)                       \
  _(kNumberTypeUInt, ACL_UINT32)                       \
  _(kNumberTypeUInt8, ACL_UINT8)                       \
  _(kNumberTypeUInt16, ACL_UINT16)                     \
  _(kNumberTypeUInt32, ACL_UINT32)                     \
  _(kNumberTypeUInt64, ACL_UINT64)                     \
  _(kNumberTypeFloat, ACL_FLOAT)                       \
  _(kNumberTypeFloat16, ACL_FLOAT16)                   \
  _(kNumberTypeFloat32, ACL_FLOAT)                     \
  _(kNumberTypeFloat64, ACL_DOUBLE)                    \
  _(kNumberTypeBFloat16, ACL_BF16)                     \
  _(kNumberTypeDouble, ACL_DOUBLE)                     \
  _(kNumberTypeComplex, ACL_DT_UNDEFINED)              \
  _(kNumberTypeComplex64, ACL_COMPLEX64)               \
  _(kNumberTypeComplex128, ACL_COMPLEX128)             \
  _(kNumberTypeInt4, ACL_DT_UNDEFINED)                 \
  _(kNumberTypeGLUInt, ACL_DT_UNDEFINED)

static const std::map<std::string, aclFormat> kMsFormatToAclFormat = {{kOpFormat_NCHW, ACL_FORMAT_NCHW},
                                                                      {kOpFormat_NHWC, ACL_FORMAT_NHWC},
                                                                      {kOpFormat_ND, ACL_FORMAT_ND},
                                                                      {kOpFormat_DEFAULT, ACL_FORMAT_ND},
                                                                      {kOpFormat_NC1HWC0, ACL_FORMAT_NC1HWC0},
                                                                      {kOpFormat_NDC1HWC0, ACL_FORMAT_NDC1HWC0},
                                                                      {kOpFormat_FRAC_Z, ACL_FORMAT_FRACTAL_Z},
                                                                      {kOpFormat_FRAC_NZ, ACL_FORMAT_FRACTAL_NZ},
                                                                      {kOpFormat_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D},
                                                                      {kOpFormat_NCDHW, ACL_FORMAT_NCDHW}};

static const std::map<aclDataType, std::string> kAclDatatypeToStr = {
  {ACL_FLOAT, "float"},   {ACL_FLOAT16, "float16"},     {ACL_INT8, "int8"},
  {ACL_INT32, "int32"},   {ACL_UINT8, "uint8"},         {ACL_INT16, "int16"},
  {ACL_UINT16, "uint16"}, {ACL_UINT32, "uint32"},       {ACL_INT64, "int64"},
  {ACL_UINT64, "uint64"}, {ACL_DOUBLE, "double"},       {ACL_BOOL, "bool"},
  {ACL_STRING, "string"}, {ACL_COMPLEX64, "complex64"}, {ACL_COMPLEX128, "complex128"},
  {ACL_BF16, "bf16"}};

static const std::map<aclFormat, std::string> kAclFormatToStr = {
  {ACL_FORMAT_NCHW, "NCHW"},       {ACL_FORMAT_NHWC, "NHWC"},           {ACL_FORMAT_ND, "ND"},
  {ACL_FORMAT_NC1HWC0, "NC1HWC0"}, {ACL_FORMAT_FRACTAL_Z, "FRACTAL_Z"}, {ACL_FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
  {ACL_FORMAT_HWCN, "HWCN"},       {ACL_FORMAT_NDHWC, "NDHWC"},         {ACL_FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
  {ACL_FORMAT_NCDHW, "NCDHW"},     {ACL_FORMAT_NDC1HWC0, "NDC1HWC0"},   {ACL_FRACTAL_Z_3D, "FRACTAL_Z_3D"}};

std::string aclDatatypeToStr(aclDataType type) {
  auto iter = kAclDatatypeToStr.find(type);
  if (iter != kAclDatatypeToStr.end()) {
    return iter->second;
  }
  return "undefined";
}

std::string aclFormatToStr(aclFormat fmt) {
  auto iter = kAclFormatToStr.find(fmt);
  if (iter != kAclFormatToStr.end()) {
    return iter->second;
  }
  return "undefined";
}

template <typename T>
inline std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  for (auto iter = values.begin(); iter != values.end(); ++iter) {
    ss << *iter;
    if (iter != values.end() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}

std::string AclTensorDescString(const AclDumpString &desc) {
  std::stringstream ss;
  ss << "[TensorDesc] ";
  ss << "Name = " << desc.tensor_name;
  ss << ", DataType = " << desc.data_type;
  ss << ", Origin Format = " << desc.ori_format;
  ss << ", Origin Shape = " << desc.ori_shape;
  ss << ", Device Format = " << desc.dev_format;
  ss << ", Device Shape = " << desc.dev_shape;
  ss << ", Tensor Type = ";
  if (desc.tensor_type == AclDumpString::TensorType::kDeviceTensor) {
    ss << "Device Tensor";
  } else if (desc.tensor_type == AclDumpString::TensorType::kNullTensor) {
    ss << "Null Tensor";
  } else {
    ss << "Host Tensor";
  }
  return ss.str();
}

void DumpAclString(const aclDataType data_type, const ShapeVector &ori_shape, const ShapeVector &dev_shape,
                   const aclFormat ori_format, const aclFormat dev_format, AclDumpString *dump) {
  if (dump == nullptr) {
    return;
  }
  dump->data_type = aclDatatypeToStr(data_type);
  dump->ori_format = aclFormatToStr(ori_format);
  dump->dev_format = aclFormatToStr(dev_format);
  dump->ori_shape = VectorToString(ori_shape);
  dump->dev_shape = VectorToString(dev_shape);
  dump->tensor_type = AclDumpString::TensorType::kDeviceTensor;
  if (ori_format == ACL_FORMAT_UNDEFINED && dev_format == ACL_FORMAT_UNDEFINED) {
    dump->tensor_type = AclDumpString::TensorType::kNullTensor;
  } else if (dev_format == ACL_FORMAT_UNDEFINED) {
    dump->tensor_type = AclDumpString::TensorType::kHostTensor;
  }
}

AddressPtr GetAddrPtr(KernelTensor *ptr) {
  MS_EXCEPTION_IF_NULL(ptr);
  return std::make_shared<mindspore::kernel::Address>(ptr->device_ptr(), ptr->size());
}
}  // namespace

template <typename ConvertType>
void AttrHelper<ConvertType>::ConvertValueToDstType(const ValuePtr &value, const TypeId src_type) {
  MS_EXCEPTION_IF_NULL(value);
  auto sub_converter = static_cast<ConvertType *>(this);
  switch (src_type) {
    case kNumberTypeInt32: {
      sub_converter->ConvertValue(value, AttrDeclType<int32_t>());
      break;
    }
    case kNumberTypeInt64: {
      sub_converter->ConvertValue(value, AttrDeclType<int64_t>());
      break;
    }
    case kNumberTypeFloat32: {
      sub_converter->ConvertValue(value, AttrDeclType<float>());
      break;
    }
    case kNumberTypeFloat64: {
      sub_converter->ConvertValue(value, AttrDeclType<double>());
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "Unsupported type: " << src_type;
    }
  }
}

template <typename ConvertType>
template <typename T>
void AttrHelper<ConvertType>::ConvertValueToRealType(const ValuePtr &value, const std::string &attr_name,
                                                     T trans_struct) {
  MS_EXCEPTION_IF_NULL(value);
  attr_name_ = attr_name;

  auto sub_converter = static_cast<ConvertType *>(this);
  // Set datatype
  if (value->isa<Scalar>()) {
    if constexpr (std::is_same<T, TensorParams *>::value) {
      auto scalar_type = value->type();
      MS_EXCEPTION_IF_NULL(scalar_type);
      TypeId scalar_type_id = scalar_type->type_id();
      trans_struct->data_type = scalar_type_id;
    }
  }

  if (value->isa<BoolImm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<bool>(), trans_struct);
  } else if (value->isa<Int64Imm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<int64_t>(), trans_struct);
  } else if (value->isa<Int32Imm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<int32_t>(), trans_struct);
  } else if (value->isa<FP32Imm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<float>(), trans_struct);
  } else if (value->isa<StringImm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<std::string>(), trans_struct);
  } else if (value->isa<GeDataTypeImm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<::ge::DataType>(), trans_struct);
  } else if (value->isa<ValueSequence>()) {
    ConvertListAttr(value, trans_struct);
  } else {
    MS_LOG(EXCEPTION) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
                      << ", perhaps you should add more supported type.";
  }
}

template <typename ConvertType>
template <typename T>
void AttrHelper<ConvertType>::ConvertListAttr(const ValuePtr &value, T trans_struct) {
  const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
  ShapeVector shape;
  TypePtr type_ptr = nullptr;
  bool is_ge_datatype = false;
  auto sub_converter = static_cast<ConvertType *>(this);
  GetValueSequenceDataTypeAndShape(value_sequence, &type_ptr, &shape, &is_ge_datatype);
  if (is_ge_datatype) {
    sub_converter->ConvertValue(value, AttrDeclType<std::vector<::ge::DataType>>(), trans_struct);
    return;
  }
  if (type_ptr == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(type_ptr);
  TypeId type_id = type_ptr->type_id();
  if constexpr (std::is_same<T, TensorParams *>::value) {
    trans_struct->data_type = type_id;
    trans_struct->ori_shape = shape;
    trans_struct->dev_shape = shape;
  }

  if (shape.size() > 1) {
    if (type_id == TypeId::kNumberTypeInt64) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<std::vector<int64_t>>>(), shape, trans_struct);
    } else {
      MS_LOG(EXCEPTION) << "Currently not support to convert input with value: " << value->ToString()
                        << ", perhaps you should add more supported type: " << TypeIdToString(type_id);
    }
  } else {
    if (type_id == TypeId::kNumberTypeBool) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<uint8_t>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeFloat) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<float>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeFloat32) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<float>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeInt32) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<int32_t>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeInt64) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<int64_t>>(), trans_struct);
    } else if (type_id == TypeId::kObjectTypeString) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<std::string>>(), trans_struct);
    } else {
      MS_LOG(EXCEPTION) << "Currently not support to convert input with value: " << value->ToString()
                        << ", perhaps you should add more supported type: " << TypeIdToString(type_id);
    }
  }
}

template <typename ConvertType>
void AttrHelper<ConvertType>::GetValueSequenceDataTypeAndShape(const ValuePtrList &value_sequence, TypePtr *data_type,
                                                               ShapeVector *shape, bool *is_ge_datatype) {
  MS_EXCEPTION_IF_NULL(data_type);
  MS_EXCEPTION_IF_NULL(shape);
  if (value_sequence.size() == 0) {
    MS_LOG(WARNING) << "value sequence is empty, failed to get data type";
    return;
  }
  (void)shape->push_back(value_sequence.size());
  auto val = value_sequence[0];
  if (val->isa<GeDataTypeImm>()) {
    *is_ge_datatype = true;
    return;
  }
  if (val->isa<Scalar>()) {
    *data_type = val->type();
  }
  if (val->isa<ValueSequence>()) {
    const auto &sub_sequence = val->cast<ValueSequencePtr>()->value();
    GetValueSequenceDataTypeAndShape(sub_sequence, data_type, shape, is_ge_datatype);
  }
}
void AclConverter::ConvertValueDependToHostInput(const std::string &kernel_name,
                                                 const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<TensorParams> &input_params,
                                                 const std::set<int64_t> &value_depend_args,
                                                 AclInputToHost *inputs_on_host) {
  MS_LOG(DEBUG) << "Start convert value_depend to acl host_input";
  for (const auto input_idx : value_depend_args) {
    const auto &input = inputs[input_idx];
    const auto &param = input_params[input_idx];
    KernelTensorValuePtr value_ptr = input->GetValue()->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(value_ptr);
    tensor::TensorPtr tensor = nullptr;
    if (!transform::AclHelper::IsInputDtypeSupport(kernel_name, param.data_type, input_idx)) {
      ValueDependToInputConverter value_convert;
      auto cast_map = value_convert.GetValueDependCastMap();
      auto iter = cast_map.find(param.data_type);
      if (iter == cast_map.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << kernel_name << " input(" << input_idx
                                   << ") data type not support and can not add Cast.";
      }
      if (!transform::AclHelper::IsInputDtypeSupport(kernel_name, iter->second, input_idx)) {
        MS_LOG(INTERNAL_EXCEPTION) << kernel_name << " input(" << input_idx << ") data type not support.";
      }
      value_convert.ConvertValueToDstType(value_ptr, param.data_type);
      tensor = value_convert.GetTensor();
    } else {
      tensor = std::make_shared<tensor::Tensor>(param.data_type, param.ori_shape);
      auto ret = memcpy_s(tensor->data_c(), tensor->Size(), value_ptr->GetDataPtr(), value_ptr->GetDataSize());
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
      }
    }
    inputs_on_host->emplace(input_idx, tensor);
  }
}

void AclConverter::ConvertToAclInput(const PrimitivePtr &prim, const AclInputToHost &host_inputs,
                                     const std::vector<KernelTensor *> &inputs,
                                     const std::vector<TensorParams> &input_params) {
  auto &prim_name = prim->name();
  auto info = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto flags = info->GetInputMappingFlags();
  if ((flags & GeTensorInfo::kEmptyParam) != 0) {
    return;
  }

  // Special const input.
  bool set_const = false;
  if (AclAdapterManager::GetInstance().CheckAclAdapter(info->op_type())) {
    set_const = AclAdapterManager::GetInstance().GetOpInfo(info->op_type()).is_const_input();
  }

  // NOTE: Key of `input_params` is input index of mindspore operator prototype in MS->GE op adapter
  // this lambda function is used to process operator not containing dynamic input
  auto get_input_param = [&inputs, &host_inputs, &prim](
                           size_t ms_proto_idx, size_t ms_real_idx) -> std::pair<AddressPtr, tensor::TensorPtr> {
    auto host_input_tensor = host_inputs.get(ms_proto_idx);
    if (host_input_tensor != nullptr) {
      return {nullptr, host_input_tensor};
    }
    if (ms_real_idx >= inputs.size()) {
      MS_LOG(EXCEPTION) << "Failed to find input " << ms_proto_idx << " for " << prim->name();
    }
    return {GetAddrPtr(inputs[ms_real_idx]), nullptr};
  };

  // NOTE: num of real inputs params may less than `info->GetNumInputsOfMsOpProto()`, e.g. Conv2D without bias
  size_t num_real_inputs = inputs.size();
  size_t ms_real_idx = 0;
  for (size_t ms_idx = 0; ms_idx < info->GetNumInputsOfMsOpProto(); ++ms_idx) {
    // skip attribute convert input
    auto attr_iter = info->attr_input_map().find(ms_idx);
    if ((attr_iter != info->attr_input_map().end()) && (prim->attrs().count(attr_iter->second) > 0)) {
      MS_LOG(DEBUG) << "Skip input " << ms_idx << " converted from attribute " << attr_iter->second;
      continue;
    }

    if (ms_real_idx >= num_real_inputs) {
      break;
    }

    auto opt_ge_input_info = info->GetOptGeInputByMsInputIndex(ms_idx);
    // mindpore input mapped to GE attribute
    if (!opt_ge_input_info.has_value()) {
      MS_LOG(DEBUG) << "Not found matched GE input for mindspore input idx:" << ms_idx << " of primitive " << prim_name;
      ms_real_idx += 1;
      continue;
    }

    auto &ge_input_info = opt_ge_input_info.value();
    size_t count = (ge_input_info.type == Ms2GeParamInfo::DYNAMIC ? num_folded_inputs_ : 1);
    size_t ge_start_idx = (ge_input_info.is_after_dynamic ? ge_input_info.index + count - 1 : ge_input_info.index);
    AclDumpString dump_str;
    AclDumpString *dump_str_pointer = transform::AclHelper::IsPrintDebugString() ? &dump_str : nullptr;

    for (size_t i = 0; i < count; i++) {
      auto [dev_address, host_tensor] = get_input_param(ms_idx, ms_real_idx);
      std::string arg_name =
        (ge_input_info.type == Ms2GeParamInfo::DYNAMIC ? ge_input_info.name + std::to_string(i) : ge_input_info.name);
      size_t acl_real_input_idx = ge_start_idx + i;
      MS_LOG(DEBUG) << "Fill acl real input " << acl_real_input_idx << " use ms real input " << ms_real_idx;
      auto [acl_desc, acl_data] =
        (host_tensor != nullptr)
          ? ConvertTensorToAclDesc(host_tensor, input_params[ms_real_idx], arg_name, dump_str_pointer)
          : ConvertTensorToAclDesc(dev_address, input_params[ms_real_idx], arg_name, dump_str_pointer);
      if (set_const && (host_tensor != nullptr)) {
        (void)aclSetTensorConst(acl_desc, host_tensor->data_c(), host_tensor->Size());
      }
      runner_.SetInput(acl_real_input_idx, acl_desc, acl_data);
      if (transform::AclHelper::IsPrintDebugString()) {
        input_str_[acl_real_input_idx] = dump_str;
      }
      ms_real_idx += 1;
    }
  }
}

void AclConverter::ConvertToAclOutput(const std::string &kernel_name, const std::vector<KernelTensor *> &outputs,
                                      const std::vector<TensorParams> &output_params) {
  // Get output real index
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto flags = info->GetOutputMappingFlags();

  // pre-allocate output buffer
  size_t num_max_outputs = ((flags & GeTensorInfo::kDynamicParam) ? outputs.size() : info->GetNumOutputsOfMsOpProto());
  if (transform::AclHelper::IsPrintDebugString()) {
    output_str_.clear();
    output_str_.resize(num_max_outputs);
  }
  runner_.ResizeOpOutputs(num_max_outputs);

  if ((flags & GeTensorInfo::kEmptyParam) != 0) {
    return;
  }

  // NOTE: suppose there is only one dynamic output
  size_t num_folded_outputs =
    ((flags & GeTensorInfo::kDynamicParam) ? outputs.size() - info->GetNumOutputsOfMsOpProto() + 1 : 0);
  // NOTE: num of real outputs params may larger than `info->GetNumOutputsOfMsOpProto()`, e.g. ApplyAdagradV2, ApplyAdam
  size_t num_real_outputs = outputs.size();
  size_t ms_real_idx = 0;
  for (size_t ms_idx = 0; ms_idx < info->GetNumOutputsOfMsOpProto(); ++ms_idx) {
    if (ms_real_idx >= num_real_outputs) {
      break;
    }

    auto opt_ge_output_info = info->GetOptGeOutputByMsOutputIndex(ms_idx);
    // mindpore op contains extra output parameters, e.g. ApplyAdagradV2, ApplyAdam
    if (!opt_ge_output_info.has_value()) {
      MS_LOG(DEBUG) << "Not found matched GE input for mindspore input idx:" << ms_idx << " of primitive "
                    << kernel_name;
      ms_real_idx += 1;
      continue;
    }

    auto &ge_output_info = opt_ge_output_info.value();
    size_t count = (ge_output_info.type == Ms2GeParamInfo::DYNAMIC ? num_folded_outputs : 1);
    size_t ge_start_idx = (ge_output_info.is_after_dynamic ? ge_output_info.index + count - 1 : ge_output_info.index);
    AclDumpString dump_str;
    AclDumpString *dump_str_pointer = transform::AclHelper::IsPrintDebugString() ? &dump_str : nullptr;

    for (size_t i = 0; i < count; i++) {
      std::string arg_name = (ge_output_info.type == Ms2GeParamInfo::DYNAMIC ? ge_output_info.name + std::to_string(i)
                                                                             : ge_output_info.name);

      size_t acl_real_output_idx = ge_start_idx + i;
      MS_LOG(DEBUG) << "Fill acl real output " << acl_real_output_idx << " use ms real output " << ms_real_idx;
      auto [acl_desc, acl_data] = ConvertTensorToAclDesc(GetAddrPtr(outputs[ms_real_idx]), output_params[ms_real_idx],
                                                         arg_name, dump_str_pointer);
      runner_.SetOutput(acl_real_output_idx, acl_desc, acl_data);
      if (transform::AclHelper::IsPrintDebugString()) {
        output_str_[acl_real_output_idx] = dump_str;
      }
      ms_real_idx += 1;
    }
  }
}

void AclConverter::ConvertAttrToAclInput(const mindspore::HashMap<std::string, ValuePtr> &attrs,
                                         const std::string &kernel_name, AclInputToHost *inputs_on_host) {
  MS_LOG(DEBUG) << "Start convert attr to acl input";
  MS_EXCEPTION_IF_NULL(inputs_on_host);
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto &[input_idx, ms_attr_name] : info->attr_input_map()) {
    auto iter = attrs.find(ms_attr_name);
    if (iter == attrs.end()) {
      MS_LOG(DEBUG) << "Not found attr " << ms_attr_name << " for primitive " << kernel_name << ", ignore it.";
      continue;
    }

    auto opt_ge_input_info = info->GetOptGeInputByMsInputIndex(input_idx);
    // mindpore input mapped to GE attribute
    if (!opt_ge_input_info.has_value()) {
      MS_LOG(DEBUG) << "Not found matched GE input for mindspore input idx:" << input_idx << " of primitive "
                    << kernel_name;
      continue;
    }

    auto &ge_input_info = opt_ge_input_info.value();
    if (ge_input_info.type == Ms2GeParamInfo::DYNAMIC) {
      MS_LOG(EXCEPTION) << "Mindspore attribute " << ms_attr_name << " mapped to a dynamic GE input";
    }
    size_t acl_real_input_idx =
      (ge_input_info.is_after_dynamic ? ge_input_info.index + num_folded_inputs_ - 1 : ge_input_info.index);

    AttrToInputConverter attr_coverter;
    TensorParams new_params;
    attr_coverter.ConvertValueToRealType(iter->second, ms_attr_name, &new_params);
    auto input_tensor = attr_coverter.GetTensor();
    inputs_on_host->emplace(input_idx, input_tensor);
    AclDumpString dump_str;
    AclDumpString *dump_str_pointer = transform::AclHelper::IsPrintDebugString() ? &dump_str : nullptr;
    auto [acl_desc, acl_data] = ConvertTensorToAclDesc(input_tensor, new_params, ge_input_info.name, dump_str_pointer);

    if (AclAdapterManager::GetInstance().CheckAclAdapter(info->op_type())) {
      auto set_const = AclAdapterManager::GetInstance().GetOpInfo(info->op_type()).is_const_input();
      if (set_const && (input_tensor != nullptr)) {
        auto const_ret = aclSetTensorConst(acl_desc, input_tensor->data_c(), input_tensor->Size());
        if (const_ret != ACL_SUCCESS) {
          MS_LOG(EXCEPTION) << "AclSetTensorConst failed! error op is " << info->op_type()
                            << " with error code:" << const_ret;
        }
      }
    }

    runner_.SetInput(acl_real_input_idx, acl_desc, acl_data);
    if (transform::AclHelper::IsPrintDebugString()) {
      input_str_[acl_real_input_idx] = dump_str;
    }
    MS_LOG(DEBUG) << "Fill acl real input " << acl_real_input_idx << " with attribute " << ms_attr_name
                  << " of primitive " << kernel_name;
  }
  MS_LOG(DEBUG) << "Convert attr to acl input over";
}

std::string AclConverter::GetFormatFromInputAttrMap(const std::vector<KernelTensor *> &inputs,
                                                    const std::string &kernel_name) {
  MS_LOG(DEBUG) << "Start GetFormatFromInputAttrMap";
  std::string format;
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto &[input_idx, attr_name] : info->input_attr_map()) {
    if (attr_name == kAttrDataFormat) {
      // adapter dyn_num input
      auto cur_input_idx = num_folded_inputs_ == 0 ? input_idx : input_idx + num_folded_inputs_ - 1;
      MS_LOG(DEBUG) << "Operator " << kernel_name << " converts input " << input_idx << " to attribute " << attr_name;
      if (cur_input_idx >= inputs.size()) {
        MS_LOG(DEBUG) << "Operator " << kernel_name << " index " << cur_input_idx
                      << " is out of range of inputs, size of which is " << inputs.size() << ", ignore it.";
        continue;
      }
      MS_EXCEPTION_IF_NULL(inputs[cur_input_idx]);
      auto format_enum = ops::GetScalarValue<int64_t>(inputs[cur_input_idx]->GetValue());
      format = FormatEnumToString(static_cast<mindspore::Format>(format_enum.value()));
    }
  }
  return format;
}

void AclConverter::ConvertInputToAclAttr(const std::vector<KernelTensor *> &inputs, const std::string &kernel_name) {
  MS_LOG(DEBUG) << "Start convert input to acl attr";
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto &[input_idx, attr_name] : info->input_attr_map()) {
    // adapter dyn_num input
    auto cur_input_idx = num_folded_inputs_ == 0 ? input_idx : input_idx + num_folded_inputs_ - 1;
    MS_LOG(DEBUG) << "Operator " << kernel_name << " converts input " << input_idx << " to attribute " << attr_name;
    if (input_idx >= inputs.size()) {
      MS_LOG(DEBUG) << "Operator " << kernel_name << " index " << input_idx
                    << " is out of range of inputs, size of which is " << inputs.size() << ", ignore it.";
      continue;
    }
    MS_EXCEPTION_IF_NULL(inputs[cur_input_idx]);
    ValuePtr attr_value = inputs[cur_input_idx]->GetValue();
    info->GetGeAttrValueByMsInputValue(input_idx + 1, &attr_value);

    AttrConverter attr_coverter;
    attr_coverter.ConvertValueToRealType(attr_value, attr_name, this);
  }
  MS_LOG(DEBUG) << "Convert input to acl attr over";
}

void AclConverter::ConvertInputToAclAttr(const AclInputToHost &inputs, const std::string &kernel_name) {
  MS_LOG(DEBUG) << "Start convert input to acl attr";
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto &[input_idx, attr_name] : info->input_attr_map()) {
    auto input_tensor = inputs.get(input_idx);
    if (input_tensor == nullptr) {
      MS_LOG(ERROR) << "failed to get input [" << input_idx << "] for primitive " << kernel_name;
      continue;
    }
    auto tensor_value = CreateValueFromTensor(input_tensor);
    info->GetGeAttrValueByMsInputValue(input_idx + 1, &tensor_value);

    AttrConverter attr_coverter;
    attr_coverter.ConvertValueToRealType(tensor_value, attr_name, this);
  }
  MS_LOG(DEBUG) << "Convert input to acl attr over";
}

void AclConverter::ConvertToAclAttr(const mindspore::HashMap<std::string, ValuePtr> &attrs,
                                    const std::string &prim_name, std::vector<std::string> *ms_attr_str) {
  MS_LOG(DEBUG) << "Start convert mindspore attr to acl attr";
  auto info = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto &ms_ge_attr_map = info->attr_map();

  for (const auto &[ms_attr_name, ge_attr_name] : ms_ge_attr_map) {
    ValuePtr attr_value = nullptr;
    if (attrs.count(ms_attr_name) != 0) {
      attr_value = attrs.at(ms_attr_name);
    }
    info->GetGeAttrValueByMsAttrValue(ms_attr_name, &attr_value);

    // Dump Info
    if (ms_attr_str != nullptr) {
      std::stringstream ss;
      ss << "attr name: " << ms_attr_name << ", value: " << attr_value->ToString();
      (void)ms_attr_str->emplace_back(ss.str());
    }

    AttrConverter attr_coverter;
    attr_coverter.ConvertValueToRealType(attr_value, ge_attr_name, this);
  }
  MS_LOG(DEBUG) << "convert mindspore attr to acl attr over";
}

void AclConverter::ConvertToAclOpType(const std::string &prim_name) {
  auto info = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto op_type = info->op_type();
  runner_.SetName(op_type);
}

void AclConverter::ResizeAclOpInputs(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto info = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto flags = info->GetInputMappingFlags();
  size_t num_max_inputs = info->GetNumInputsOfMsOpProto();

  if (flags & GeTensorInfo::kDynamicParam) {
    std::vector<int64_t> dyn_input_sizes = {};
    if (prim->HasAttr(kAttrDynInputSizes)) {
      dyn_input_sizes = GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrDynInputSizes));
    }
    if (dyn_input_sizes.empty()) {
      MS_LOG(EXCEPTION) << "Attribute " << kAttrDynInputSizes << " of primitive " << prim_name << " is "
                        << dyn_input_sizes << ", of which size is empty";
    }
    num_folded_inputs_ = LongToSize(dyn_input_sizes[0]);
    num_max_inputs = info->GetNumInputsOfMsOpProto() + num_folded_inputs_ - 1;
  }

  if (transform::AclHelper::IsPrintDebugString()) {
    input_str_.clear();
    input_str_.resize(num_max_inputs);
  }
  runner_.ResizeOpInputs(num_max_inputs);
}

aclDataType AclConverter::ConvertType(TypeId type) {
  static constexpr aclDataType kDataTypeToAclDataTypeTable[static_cast<int64_t>(kNumberTypeEnd)] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_MINDSPORE_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
  };
  if (type == kMetaTypeNone || type == kTypeUnknown) {
    return ACL_DT_UNDEFINED;
  }
  if (type <= kNumberTypeBegin || type >= kNumberTypeEnd) {
    MS_LOG(EXCEPTION) << "Invalid datatype:" << type;
  }
  auto acl_type = kDataTypeToAclDataTypeTable[type - kNumberTypeBegin - 1];
  if (acl_type == ACL_DT_UNDEFINED) {
    MS_LOG(EXCEPTION) << "Invalid datatype:" << type;
  }
  return acl_type;
}

aclFormat AclConverter::ConvertFormat(const std::string &format) {
  auto iter = kMsFormatToAclFormat.find(format);
  if (iter == kMsFormatToAclFormat.end()) {
    MS_LOG(EXCEPTION) << "Invalid format:" << format;
  }
  return iter->second;
}

std::pair<aclTensorDesc *, aclDataBuffer *> AclConverter::CreateTensorDesc(const tensor::TensorPtr &tensor,
                                                                           const ShapeVector &dev_shape,
                                                                           const std::string &dev_format,
                                                                           const std::string &desc_name) {
  MS_EXCEPTION_IF_NULL(tensor);
  AclTensorDescMaker acl_tensor;
  aclTensorDesc *acl_desc = nullptr;

  const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);

  auto acl_data_type = ConvertType(tensor->data_type());
  auto acl_ori_format = ConvertFormat(device_address->format());
  auto acl_dev_format = ConvertFormat(dev_format);
  acl_desc = acl_tensor.Create(acl_data_type, tensor->shape(), acl_ori_format)
               .SetShape(dev_shape)
               .SetFormat(acl_dev_format)
               .SetName(desc_name)
               .Get();
  MS_EXCEPTION_IF_NULL(acl_desc);

  // Create buf.
  auto buffer_maker =
    std::make_shared<AclTensorBufferMaker>(device_address->GetMutablePtr(), device_address->GetSize());
  auto acl_data = buffer_maker->Get();
  MS_EXCEPTION_IF_NULL(acl_data);
  return std::make_pair(acl_desc, acl_data);
}

std::pair<aclTensorDesc *, aclDataBuffer *> AclConverter::ConvertTensorToAclDesc(const AddressPtr &address,
                                                                                 const TensorParams &params,
                                                                                 const std::string &desc_name,
                                                                                 AclDumpString *dump_str) const {
  AclTensorDescMaker tensor;
  if (dump_str != nullptr) {
    dump_str->tensor_name = desc_name;
  }

  // Create desc.
  aclTensorDesc *acl_desc = nullptr;
  if (params.data_type == kMetaTypeNone) {
    acl_desc = tensor.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(ACL_DT_UNDEFINED, params.ori_shape, params.dev_shape, ACL_FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED,
                  dump_str);
  } else {
    auto acl_data_type = ConvertType(params.data_type);
    auto acl_ori_format = ConvertFormat(params.ori_format);
    auto acl_dev_format = ConvertFormat(params.dev_format);
    acl_desc = tensor.Create(acl_data_type, params.ori_shape, acl_ori_format)
                 .SetShape(params.dev_shape)
                 .SetFormat(acl_dev_format)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(acl_data_type, params.ori_shape, params.dev_shape, acl_ori_format, acl_dev_format, dump_str);
  }
  MS_EXCEPTION_IF_NULL(acl_desc);

  // Create buf.
  auto buffer_maker = std::make_shared<AclTensorBufferMaker>(address->addr, address->size);
  auto acl_data = buffer_maker->Get();
  MS_EXCEPTION_IF_NULL(acl_data);

  return std::make_pair(acl_desc, acl_data);
}

std::pair<aclTensorDesc *, aclDataBuffer *> AclConverter::ConvertTensorToAclDesc(const tensor::TensorPtr &host_tensor,
                                                                                 const TensorParams &params,
                                                                                 const std::string &desc_name,
                                                                                 AclDumpString *dump_str) const {
  AclTensorDescMaker tensor;
  if (dump_str != nullptr) {
    dump_str->tensor_name = desc_name;
  }

  // Create desc.
  aclTensorDesc *acl_desc = nullptr;
  if (params.data_type == kMetaTypeNone) {
    acl_desc = tensor.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(ACL_DT_UNDEFINED, params.ori_shape, params.dev_shape, ACL_FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED,
                  dump_str);
  } else {
    auto acl_data_type = ConvertType(host_tensor->data_type());
    auto acl_ori_format = ConvertFormat(params.ori_format);
    acl_desc = tensor.Create(acl_data_type, params.ori_shape, acl_ori_format)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(acl_data_type, params.ori_shape, params.dev_shape, acl_ori_format, ACL_FORMAT_UNDEFINED, dump_str);
  }
  MS_EXCEPTION_IF_NULL(acl_desc);

  // Create buf.
  auto buffer_maker = std::make_shared<AclTensorBufferMaker>(host_tensor);
  auto acl_data = buffer_maker->Get();
  MS_EXCEPTION_IF_NULL(acl_data);

  return std::make_pair(acl_desc, acl_data);
}

template <typename T>
void AclConverter::AclRunnerAddAttr(const std::string &attrName, T value) {
  runner_.AddAttr(attrName, value);
  if (transform::AclHelper::IsPrintDebugString()) {
    std::stringstream ss;
    ss << "attr name: " << attrName << ", value: " << value;
    (void)attr_map_str_.emplace_back(ss.str());
    MS_LOG(DEBUG) << "set acl attr:" << attrName << " value:" << value;
  }
}

std::string AclConverter::DebugString() const {
  if (!transform::AclHelper::IsPrintDebugString()) {
    return "";
  }
  std::stringstream ss;
  ss << "[AclLaunchInfo]OpType:" << runner_.GetName() << std::endl;
  for (size_t i = 0; i < runner_.GetNumRealInputs(); ++i) {
    ss << "InputDesc[" << i << "]:";
    ss << AclTensorDescString(input_str_[i]) << std::endl;
  }
  for (size_t i = 0; i < attr_map_str_.size(); ++i) {
    ss << "Attr[" << i << "] " << attr_map_str_[i] << std::endl;
  }
  for (size_t i = 0; i < runner_.GetNumRealOutputs(); ++i) {
    ss << "OutputDesc[" << i << "]:";
    ss << AclTensorDescString(output_str_[i]) << std::endl;
  }
  return ss.str();
}

void AclConverter::ProcessRunnerSpecialInfo(const std::string &prim_name,
                                            const std::vector<TensorParams> &output_params) {
  auto opinfo = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(opinfo);
  auto op_type = opinfo->op_type();
  if (!AclAdapterManager::GetInstance().CheckAclAdapter(op_type)) {
    // Default fuzz compile.
    is_dynamic_ = true;
    // 910 default fp16 mode.
    precision_mode_ = ALLOW_FP32_TO_FP16;
    return;
  }
  auto info = AclAdapterManager::GetInstance().GetOpInfo(op_type);

  // Set need retrieve output shape flag.
  is_need_retrieve_output_shape_ = info.is_need_retrieve_output_shape();

  // Set dynamic or static compile mode.
  is_dynamic_ = info.is_dynamic();

  // Set acl precision mode
  precision_mode_ = info.precision_mode();
  if (precision_mode_ == FORCE_FP32 &&
      std::any_of(output_params.begin(), output_params.end(),
                  [](const TensorParams &param) { return param.data_type != kNumberTypeFloat32; })) {
    precision_mode_ = ALLOW_FP32_TO_FP16;
  }
}

void AclConverter::SetRunnerSpecialInfo() {
  if (is_dynamic_) {
    runner_.SetDynamicMode();
  } else {
    runner_.SetStaticMode();
  }
  runner_.SetPrecisionMode(precision_mode_);
}

void AclConverter::Run(void *stream_ptr) { runner_.Run(stream_ptr, is_need_retrieve_output_shape_); }

void AclConverter::Reset() {
  runner_.Reset();
  if (transform::AclHelper::IsPrintDebugString()) {
    input_str_.clear();
    output_str_.clear();
    attr_map_str_.clear();
  }
  num_folded_inputs_ = 0;
  is_dynamic_ = false;
  precision_mode_ = FORCE_FP32;
  is_need_retrieve_output_shape_ = false;
}
}  // namespace mindspore::transform
