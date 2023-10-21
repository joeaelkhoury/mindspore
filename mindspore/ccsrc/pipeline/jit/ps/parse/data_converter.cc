/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/parse/data_converter.h"
#include <utility>
#include "mindspore/core/ops/structure_ops.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/pipeline.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "ir/func_graph_cloner.h"
#include "ir/cell.h"
#include "utils/symbolic.h"
#include "utils/ms_context.h"
#include "include/common/fallback.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"

namespace mindspore {
namespace parse {
namespace {
struct PyDataToValueRegister {
  PyDataToValueRegister() noexcept {
    python_adapter::PyAdapterCallback::SetPyDataToValueHandler(data_converter::PyDataToValue);
  }
} callback_register;
}  // namespace
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;
using MetaTensor = mindspore::tensor::MetaTensor;
using MetaTensorPtr = mindspore::tensor::MetaTensorPtr;
using CSRTensor = mindspore::tensor::CSRTensor;
using CSRTensorPtr = mindspore::tensor::CSRTensorPtr;
using COOTensor = mindspore::tensor::COOTensor;
using COOTensorPtr = mindspore::tensor::COOTensorPtr;
using MapTensor = mindspore::tensor::MapTensor;
using MapTensorPtr = mindspore::tensor::MapTensorPtr;

using InstanceCheckFunc = std::function<bool(const py::object &)>;
using InstanceConvertFunc = std::function<ValuePtr(const py::object &, bool, const TypePtr &, const ValuePtrList &)>;
static constexpr int kBit8 = 8;
static constexpr int kBit16 = 16;
static constexpr int kBit32 = 32;
static constexpr int kBit64 = 64;

class DataConvertFunc {
 public:
  explicit DataConvertFunc(InstanceConvertFunc convert_func) : convert_func_(std::move(convert_func)) {}

  virtual ~DataConvertFunc() = default;

  virtual bool Matched(const py::object &obj) = 0;

  ValuePtr ConvertPyObject(const py::object &obj, bool use_sig, const TypePtr &dtype,
                           const ValuePtrList &args_value_list = {}) {
    if (convert_func_ == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "convert func is null";
    }
    return convert_func_(obj, use_sig, dtype, args_value_list);
  }

 private:
  InstanceConvertFunc convert_func_ = nullptr;
};

using DataConvertFuncPtr = std::shared_ptr<DataConvertFunc>;

using ArgsObjConvertFunc = std::function<ValuePtr(const py::object &)>;
using ArgsObjSigConvertFunc = std::function<ValuePtr(const py::object &, bool)>;
using ArgsObjTypeConvertFunc = std::function<ValuePtr(const py::object &, const TypePtr &)>;
using ArgsObjArgsValueConvertFunc = std::function<ValuePtr(const py::object &, const ValuePtrList &)>;

// Convert the data according to instance type
template <typename T>
class ByTypeDataConvertFunc : public DataConvertFunc {
 public:
  explicit ByTypeDataConvertFunc(const InstanceConvertFunc &convert_func)
      : DataConvertFunc(convert_func), check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ValuePtr &converted_type)
      : DataConvertFunc([converted_type](const py::object &, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return converted_type;
        }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return convert_func(obj);
        }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjSigConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool use_sig, const TypePtr &,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, use_sig); }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjTypeConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &dtype,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, dtype); }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConvertFunc(const ArgsObjArgsValueConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &,
                                       const ValuePtrList &args_value_list) -> ValuePtr {
          return convert_func(obj, args_value_list);
        }),
        check_func_(py::isinstance<T>) {}

  ~ByTypeDataConvertFunc() override = default;

  bool Matched(const py::object &obj) override { return check_func_ != nullptr ? check_func_(obj) : false; }

 private:
  InstanceCheckFunc check_func_ = nullptr;
};

// Convert the data according to object attribute.
class ByAttrDataConvertFunc : public DataConvertFunc {
 public:
  ByAttrDataConvertFunc(const std::string &attr_name, const ArgsObjConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return convert_func(obj);
        }),
        attr_name_(attr_name) {}

  ByAttrDataConvertFunc(const std::string &attr_name, const ArgsObjSigConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool use_sig, const TypePtr &,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, use_sig); }),
        attr_name_(attr_name) {}

  ~ByAttrDataConvertFunc() override = default;

  bool Matched(const py::object &obj) override { return py::hasattr(obj, attr_name_.c_str()); }

 private:
  std::string attr_name_;
};

// Convert the data according to match function.
class ByFuncDataConvertFunc : public DataConvertFunc {
 public:
  ByFuncDataConvertFunc(const InstanceCheckFunc &match_func, const ArgsObjConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool, const TypePtr &, const ValuePtrList &) -> ValuePtr {
          return convert_func(obj);
        }),
        match_func_(match_func) {}

  ByFuncDataConvertFunc(const InstanceCheckFunc &match_func, const ArgsObjSigConvertFunc &convert_func)
      : DataConvertFunc([convert_func](const py::object &obj, bool use_sig, const TypePtr &,
                                       const ValuePtrList &) -> ValuePtr { return convert_func(obj, use_sig); }),
        match_func_(match_func) {}

  ~ByFuncDataConvertFunc() override = default;

  bool Matched(const py::object &obj) override { return match_func_ != nullptr ? match_func_(obj) : false; }

 private:
  InstanceCheckFunc match_func_ = nullptr;
};

FuncGraphPtr ConvertToBpropCut(const py::object &obj) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_key = results[0];
  py::function bprop_func = py::getattr(obj, CUSTOM_BPROP_NAME);

  auto bprop_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;

  auto fake_bprop = std::make_shared<PrimitivePy>("bprop_cut");
  fake_bprop->AddBackwardHookFn(0, bprop_func);
  (void)fake_bprop->AddAttr(CUSTOM_BPROP_NAME, MakeValue(true));
  outputs.push_back(NewValueNode(fake_bprop));

  py::object code_obj = py::getattr(bprop_func, "__code__");
  // Three parameters self, out and dout need to be excluded
  constexpr auto kBpropExcludeParamNum = 3;
  size_t inputs_num = py::cast<int64_t>(py::getattr(code_obj, "co_argcount")) - kBpropExcludeParamNum;
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = bprop_graph->add_parameter();
    outputs.push_back(param);
  }
  auto p1 = bprop_graph->add_parameter();
  auto p2 = bprop_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  bprop_graph->set_output(bprop_graph->NewCNode(std::move(outputs)));
  data_converter::SetObjGraphValue(obj_key, bprop_graph);
  return bprop_graph;
}

namespace {
ValuePtr ConvertTuple(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python tuple";
  auto tuple = obj.cast<py::tuple>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < tuple.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(tuple[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueTuple>(value_list);
}

bool IsNamedTuple(const py::object &obj) { return py::hasattr(obj, "_fields") && py::isinstance<py::tuple>(obj); }

ValuePtr ConvertNamedTuple(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python NamedTuple";
  if (!py::hasattr(obj, "_asdict")) {
    return nullptr;
  }
  auto asdict_fn = obj.attr("_asdict");
  auto asdict_obj = asdict_fn();
  auto dict_values = asdict_obj.cast<py::dict>();
  std::vector<ValuePtr> keys;
  std::vector<ValuePtr> values;
  for (auto item : dict_values) {
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = ConvertData(py::cast<py::object>(item.first), &key, use_signature) &&
                   ConvertData(py::cast<py::object>(item.second), &value, use_signature);
    if (!success) {
      return nullptr;
    }
    MS_LOG(DEBUG) << key->ToString() << ", " << value->ToString();
    keys.push_back(key);
    values.push_back(value);
  }
  auto obj_name = obj.attr("__class__").attr("__name__");
  std::string type_name = py::str(obj_name).cast<std::string>();
  return std::make_shared<ValueNamedTuple>(type_name, keys, values);
}

ValuePtr ConvertStubTuple(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python tuple";
  auto tuple = obj.cast<py::tuple>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < tuple.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertStubData(tuple[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueTuple>(value_list);
}

ValuePtr ConvertList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python list";

  auto list = obj.cast<py::list>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(list[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueList>(value_list);
}

ValuePtr ConvertStubList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python list";

  auto list = obj.cast<py::list>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertStubData(list[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueList>(value_list);
}

ValuePtr ConvertCellList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting cell list";
  py::sequence list = obj;
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(list[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueTuple>(value_list);
}

ValuePtr ConvertDict(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python dict";
  auto dict_values = obj.cast<py::dict>();
  std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
  for (auto item : dict_values) {
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = ConvertData(py::cast<py::object>(item.first), &key, use_signature) &&
                   ConvertData(py::cast<py::object>(item.second), &value, use_signature);
    if (!success) {
      return nullptr;
    }
    (void)key_values.emplace_back(key, value);
  }
  return std::make_shared<ValueDictionary>(key_values);
}

ValuePtr ConvertModuleNameSpace(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting python module";
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object module_namespace = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MODULE_NAMESPACE, obj);
  auto converted = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_MODULE, module_namespace, obj);
  MS_LOG(DEBUG) << "name_space: " << converted->ToString();
  return converted;
}

ValuePtr ConvertMsClass(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting ms class";
  // Convert class instance decorated with jit_class.
  if (py::hasattr(obj, PYTHON_PARSE_METHOD)) {
    MS_LOG(DEBUG) << "Convert obj to func graph.";
    FuncGraphPtr func_graph = ConvertToFuncGraph(obj);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
    PyObjectWrapperPtr python_obj = std::make_shared<PyObjectWrapper>(obj, "graph python obj");
    func_graph->set_python_obj(python_obj);
    return func_graph;
  }
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object name = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MS_CLASS_NAME, obj);
  auto cls_name = py::cast<std::string>(name);
  return std::make_shared<MsClassObject>(obj, cls_name);
}

ValuePtr ConvertPrimitive(const py::object &obj, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting primitive object" << use_signature;

  // need check the primitive is class type or instance
  auto obj_type = data_converter::GetObjType(obj);
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    auto desc = py::cast<std::string>(python_adapter::CallPyObjMethod(obj, PYTHON_GET_OBJ_DESC, obj));
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
    return std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  }
  py::object adapter_obj = obj;
  if (py::hasattr(obj, "__setattr_flag__")) {
    if (py::hasattr(obj, "_clone")) {
      auto clone_fn = obj.attr("_clone");
      adapter_obj = clone_fn();
    }
  }
  auto prim_adapter = adapter_obj.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(prim_adapter);
  auto primitive = prim_adapter->attached_primitive();
  if (primitive == nullptr) {
    primitive = std::make_shared<PrimitivePy>(adapter_obj);
    prim_adapter->set_attached_primitive(primitive);
  }

  if (use_signature) {
    return std::make_shared<prim::DoSignaturePrimitive>(primitive->name(), primitive);
  }
  return primitive;
}

ValuePtr ConvertMetaFuncGraph(const py::object &obj, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting MetaFuncGraph object";
  auto meta = obj.cast<MetaFuncGraphPtr>();
  if (meta == nullptr) {
    MS_LOG(ERROR) << "Resolve MetaFuncGraph error, get ptr is null";
    return nullptr;
  }
  if (use_signature) {
    return std::make_shared<prim::DoSignaturePrimitive>(meta->name(), meta);
  }
  return meta;
}

ValuePtr ConvertFuncGraph(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting FuncGraph object";
  auto func_graph = obj.cast<FuncGraphPtr>();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Resolve FuncGraph error, get ptr is null";
    return nullptr;
  }
  func_graph->set_attr("is_load", MakeValue(true));
  return func_graph;
}

ValuePtr ConvertSlice(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting slice object";

  auto convert_func = [obj](const std::string &attr) -> ValuePtr {
    auto py_attr = py::getattr(obj, attr.c_str());
    if (py::isinstance<py::none>(py_attr)) {
      return kNone;
    }
    if (py::isinstance<py::int_>(py_attr)) {
      auto value = py::cast<int64_t>(py_attr);
      return MakeValue(value);
    }
    if (py::isinstance<Tensor>(py_attr)) {
      return py::cast<TensorPtr>(py_attr);
    }
    if (IsStubTensor(py_attr)) {
      return ConvertStubTensor(py_attr);
    }
    MS_LOG(EXCEPTION) << "Attribute '" << attr << "' of " << py::str(obj)
                      << " should be int or Tensor with Int type but got " << py::str(py_attr);
  };
  ValuePtr start = convert_func(kSliceStart);
  ValuePtr stop = convert_func(kSliceStop);
  ValuePtr step = convert_func(kSliceStep);
  return std::make_shared<ValueSlice>(start, stop, step);
}

ValuePtr ConvertCellObjToFuncGraph(const py::object &obj, const ValuePtrList &args_value_list) {
  FuncGraphPtr func_graph = ConvertToFuncGraph(obj, args_value_list);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse resolve function error.";
    return nullptr;
  }
  // if the cell object has specified bprop, it has user-defined bprop function parse and record it
  data_converter::SetFuncGraphByCellObj(func_graph, obj);
  return func_graph;
}

ValuePtr ConvertConstantNumpyNumber(const py::object &obj, ResolveType obj_type) {
  if (obj_type == RESOLVE_TYPE_NUMPY_INT_NUMBER) {
    MS_LOG(INFO) << "Convert constant numpy int64_t number:" << (std::string)py::str(obj);
    return MakeValue(py::cast<int64_t>(obj));
  }
  if (obj_type == RESOLVE_TYPE_NUMPY_FLOAT_NUMBER) {
    MS_LOG(INFO) << "Convert constant numpy float number::" << (std::string)py::str(obj);
    return MakeValue(py::cast<float>(obj));
  }
  if (obj_type == RESOLVE_TYPE_NUMPY_BOOL_NUMBER) {
    MS_LOG(INFO) << "Convert constant numpy bool_ number::" << (std::string)py::str(obj);
    return MakeValue(py::cast<bool>(obj));
  }

  MS_LOG(ERROR) << "Convert numpy number type is invalid, obj: " << py::str(obj);
  return nullptr;
}

void CheckJITForbiddenAPI(const py::object &obj) {
  auto module = python_adapter::GetPyModule(PYTHON_MOD_MODULE);
  py::object res = python_adapter::CallPyModFn(module, PYTHON_MOD_GET_MODULE_AND_NAME_INFO, obj);
  if (!py::isinstance<py::none>(res)) {
    auto obj_info = py::cast<py::list>(res);
    auto obj_module = py::cast<std::string>(obj_info[0]);
    auto obj_name = py::cast<std::string>(obj_info[1]);
    auto obj_type = py::cast<std::string>(obj_info[2]);
    std::ostringstream oss;
    oss << "Failed to compile in GRAPH_MODE because the " << obj_type << " '" << obj_module << "." << obj_name
        << "' is not supported in 'construct' or function with @jit decorator. "
        << "Try to use the " << obj_type << " '" << obj_module << "." << obj_name << "' externally "
        << "such as initialized in the method '__init__' before assigning"
        << ".\nFor more details, please refer to "
        << "https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html \n";
    // Check if the API is decoratored by @jit_forbidden_register.
    bool is_jit_forbidden_register = data_converter::IsJITForbiddenAPI(obj);
    if (is_jit_forbidden_register) {
      MS_LOG(EXCEPTION) << oss.str();
    }
    // Check if the API's module is in the JIT forbidden module set.
    bool is_jit_forbidden_module =
      py::cast<bool>(python_adapter::CallPyModFn(module, PYTHON_MOD_IS_JIT_FORBIDDEN_MODULE, obj_info[0]));
    if (is_jit_forbidden_module) {
      MS_LOG(EXCEPTION) << oss.str();
    }
  }
}

ValuePtr ConvertOtherObj(const py::object &obj, bool forbid_reuse = false) {
  auto obj_type = data_converter::GetObjType(obj);
  MS_LOG(DEBUG) << "Converting the object(" << ((std::string)py::str(obj)) << ") detail type: " << obj_type << " ";
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    // Check JIT forbidden API
    CheckJITForbiddenAPI(obj);
    MS_LOG(DEBUG) << "Resolve the class type, need create class instance.";
    std::string desc = py::str(obj);
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
    return std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  }
  if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD ||
      (obj_type == RESOLVE_TYPE_CLASS_INSTANCE && py::hasattr(obj, PYTHON_PARSE_METHOD))) {
    if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD) {
      // Check JIT forbidden API
      CheckJITForbiddenAPI(obj);
      const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
      if (allow_fallback_runtime) {
        // Check if the function is from a third-party library.
        py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
        bool is_third_party_function =
          python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_FROM_THIRD_PARTY_LIBRARY, obj).cast<bool>();
        if (is_third_party_function) {
          MS_LOG(DEBUG) << "Converting the function from third-party library: " << py::str(obj);
          return std::make_shared<InterpretedObject>(obj);
        }
      }
    }
    MS_LOG(DEBUG) << "Convert the obj to func graph, type is " << obj_type;
    FuncGraphPtr func_graph = ConvertToFuncGraph(obj, {}, PYTHON_MOD_GET_PARSE_METHOD, forbid_reuse);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
    return func_graph;
  }
  if (obj_type == RESOLVE_TYPE_CLASS_INSTANCE) {
    MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert class instance: " << py::str(obj);
  }
  // Start RESOLVE_TYPE_INVALID.
  if (obj_type == RESOLVE_TYPE_NUMPY_INT_NUMBER || obj_type == RESOLVE_TYPE_NUMPY_FLOAT_NUMBER ||
      obj_type == RESOLVE_TYPE_NUMPY_BOOL_NUMBER) {
    return ConvertConstantNumpyNumber(obj, obj_type);
  }
  auto res = std::make_shared<InterpretedObject>(obj);
  MS_EXCEPTION_IF_NULL(res);
  MS_LOG(DEBUG) << "Get interpreted object: " << res->ToString();
  return res;
}

template <typename T>
ValuePtr ConvertNumberWithType(const T &obj, const TypePtr &dtype) {
  ValuePtr data = nullptr;
  auto int_dypte = dyn_cast<Int>(dtype);
  if (int_dypte != nullptr) {
    switch (int_dypte->nbits()) {
      case kBit8:
        data = std::make_shared<Int8Imm>(obj);
        break;
      case kBit16:
        data = std::make_shared<Int16Imm>(obj);
        break;
      case kBit32:
        data = std::make_shared<Int32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<Int64Imm>(obj);
        break;
      default:
        data = std::make_shared<Int64Imm>(obj);
    }
    return data;
  }

  auto uint_dypte = dyn_cast<UInt>(dtype);
  if (uint_dypte != nullptr) {
    switch (uint_dypte->nbits()) {
      case kBit8:
        data = std::make_shared<UInt8Imm>(obj);
        break;
      case kBit16:
        data = std::make_shared<UInt16Imm>(obj);
        break;
      case kBit32:
        data = std::make_shared<UInt32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<UInt64Imm>(obj);
        break;
      default:
        data = std::make_shared<UInt32Imm>(obj);
    }
    return data;
  }

  auto float_dypte = dyn_cast<Float>(dtype);
  if (float_dypte != nullptr) {
    switch (float_dypte->nbits()) {
      case kBit32:
        data = std::make_shared<FP32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<FP64Imm>(obj);
        break;
      default:
        data = std::make_shared<FP32Imm>(obj);
    }
    return data;
  }
  return nullptr;
}

ValuePtr ConvertIntegerWithType(const py::object &obj, const TypePtr &dtype = nullptr) {
  auto obj_int64 = py::cast<int64_t>(obj);
  if (dtype == nullptr) {
    return std::make_shared<Int64Imm>(obj_int64);
  }
  return ConvertNumberWithType<int64_t>(obj_int64, dtype);
}

ValuePtr ConvertFloatWithType(const py::object &obj, const TypePtr &dtype = nullptr) {
  auto obj_float32 = py::cast<float>(obj);
  if (dtype == nullptr) {
    auto obj_double = py::cast<double>(obj);
    auto ret = std::make_shared<FP32Imm>(obj_float32);
    ret->set_prim_value(obj_double);
    return ret;
  }
  return ConvertNumberWithType<float>(obj_float32, dtype);
}

template <typename T, typename U>
ValuePtr PyCast(const py::object &obj) {
  return std::make_shared<T>(py::cast<U>(obj));
}

template <typename T>
ValuePtr ObjCast(const py::object &obj) {
  return obj.cast<T>();
}

static const std::vector<DataConvertFuncPtr> &GetDataConvertFuncs() {
  // Convert data by python object type.
  static const std::vector<DataConvertFuncPtr> data_convert_funcs{
    // AdapterTensor needs to be processed before Tensor because it inherits from Tensor.
    std::make_shared<ByFuncDataConvertFunc>(IsStubTensor, ConvertStubTensor),
    std::make_shared<ByFuncDataConvertFunc>(IsNamedTuple, ConvertNamedTuple),
    std::make_shared<ByTypeDataConvertFunc<Tensor>>(ObjCast<TensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<py::tuple>>(ConvertTuple),
    std::make_shared<ByTypeDataConvertFunc<py::list>>(ConvertList),
    std::make_shared<ByTypeDataConvertFunc<py::bool_>>(PyCast<BoolImm, bool>),
    std::make_shared<ByTypeDataConvertFunc<py::int_>>(ConvertIntegerWithType),
    std::make_shared<ByTypeDataConvertFunc<py::float_>>(ConvertFloatWithType),
    std::make_shared<ByTypeDataConvertFunc<py::str>>(PyCast<StringImm, string>),
    std::make_shared<ByTypeDataConvertFunc<py::none>>(kNone),
    std::make_shared<ByTypeDataConvertFunc<MetaTensor>>(ObjCast<MetaTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<CSRTensor>>(ObjCast<CSRTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<COOTensor>>(ObjCast<COOTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<MapTensor>>(ObjCast<MapTensorPtr>),
    std::make_shared<ByTypeDataConvertFunc<py::ellipsis>>(kEllipsis),
    std::make_shared<ByTypeDataConvertFunc<py::module>>(ConvertModuleNameSpace),
    std::make_shared<ByAttrDataConvertFunc>(PYTHON_MS_CLASS, ConvertMsClass),
    std::make_shared<ByTypeDataConvertFunc<Type>>(ObjCast<TypePtr>),
    std::make_shared<ByTypeDataConvertFunc<UMonad>>(ObjCast<UMonadPtr>),
    std::make_shared<ByTypeDataConvertFunc<IOMonad>>(ObjCast<IOMonadPtr>),
    std::make_shared<ByAttrDataConvertFunc>(PYTHON_CLASS_MEMBER_NAMESPACE,
                                            [](const py::object &obj) -> ValuePtr {
                                              auto res =
                                                std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, obj);
                                              MS_LOG(DEBUG) << "name_space: " << res->ToString();
                                              return res;
                                            }),
    std::make_shared<ByTypeDataConvertFunc<py::dict>>(ConvertDict),
    std::make_shared<ByAttrDataConvertFunc>(PYTHON_CELL_AS_DICT, ConvertDict),
    std::make_shared<ByTypeDataConvertFunc<py::slice>>(ConvertSlice),
    std::make_shared<ByAttrDataConvertFunc>(PYTHON_CELL_AS_LIST, ConvertCellList),
    std::make_shared<ByTypeDataConvertFunc<Cell>>(ConvertCellObjToFuncGraph),
    std::make_shared<ByAttrDataConvertFunc>(PYTHON_PRIMITIVE_FLAG, ConvertPrimitive),
    std::make_shared<ByTypeDataConvertFunc<MetaFuncGraph>>(ConvertMetaFuncGraph),
    std::make_shared<ByTypeDataConvertFunc<FuncGraph>>(ConvertFuncGraph),
  };
  return data_convert_funcs;
}

static const std::vector<DataConvertFuncPtr> &GetStubDataConvertFuncs() {
  // Convert data by python object type.
  static const std::vector<DataConvertFuncPtr> data_convert_funcs{
    std::make_shared<ByFuncDataConvertFunc>([](const py::object &obj) -> bool { return IsStubTensor(obj); },
                                            PyStubNodeCast),
    std::make_shared<ByTypeDataConvertFunc<py::tuple>>(ConvertStubTuple),
    std::make_shared<ByTypeDataConvertFunc<py::list>>(ConvertStubList),
  };
  return data_convert_funcs;
}
}  // namespace

bool ConvertData(const py::object &obj, ValuePtr *data, bool use_signature, const TypePtr &dtype, bool forbid_reuse) {
  // Check parameter valid
  if (data == nullptr) {
    MS_LOG(ERROR) << "The value pointer should not be null.";
    return false;
  }
  ValuePtr converted = nullptr;
  bool matched = false;
  const auto &converters = GetDataConvertFuncs();
  for (auto &converter : converters) {
    if (converter->Matched(obj)) {
      converted = converter->ConvertPyObject(obj, use_signature, dtype);
      matched = true;
      break;
    }
  }
  if (!matched) {
    converted = ConvertOtherObj(obj, forbid_reuse);
  }
  *data = converted;
  return converted != nullptr;
}

bool ConvertStubData(const py::object &obj, ValuePtr *data, bool use_signature, const TypePtr &dtype,
                     bool forbid_reuse) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "The value pointer should not be null.";
    return false;
  }
  ValuePtr converted = nullptr;
  const auto &convert_funcs = GetStubDataConvertFuncs();
  for (auto &convert_func : convert_funcs) {
    if (convert_func->Matched(obj)) {
      converted = convert_func->ConvertPyObject(obj, use_signature, dtype);
      *data = converted;
      return converted != nullptr;
    }
  }
  return ConvertData(obj, data, use_signature, dtype, forbid_reuse);
}

// Convert data to graph
FuncGraphPtr ConvertToFuncGraph(const py::object &obj, const ValuePtrList &args_value_list,
                                const std::string &python_mod_get_parse_method, bool forbid_reuse) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_id = results[0] + python_mod_get_parse_method;
  std::string obj_key = results[1];
  FuncGraphPtr func_graph = nullptr;
  ValuePtr value = nullptr;
  bool is_cache = data_converter::GetObjectValue(obj_id, &value);
  if (is_cache && value != nullptr && value->isa<FuncGraph>()) {
    MS_LOG(DEBUG) << "Get the cache data, obj: " << obj_id;
    func_graph = value->cast<FuncGraphPtr>();
    if (!func_graph->dropped()) {
      bool has_forbid_reuse_attr = py::hasattr(obj, PYTHON_FUNCTION_FORBID_REUSE);
      if (forbid_reuse || has_forbid_reuse_attr || pipeline::GetJitLevel() == "O0") {
        return BasicClone(func_graph);
      }
      return func_graph;
    }
  }

  func_graph = ParsePythonCode(obj, python_mod_get_parse_method, args_value_list);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse resolve function error.";
    return nullptr;
  }

  data_converter::CacheObjectValue(obj_id, func_graph);
  if (!obj_key.empty() && python_mod_get_parse_method == PYTHON_MOD_GET_PARSE_METHOD) {
    MS_LOG(DEBUG) << "Add graph: " << obj_key << ", func_graph: " << func_graph->ToString();
    data_converter::SetObjGraphValue(obj_key, func_graph);
  }

  PyObjectWrapperPtr python_obj = std::make_shared<PyObjectWrapper>(obj, "graph python obj");
  func_graph->set_python_obj(python_obj);

  return func_graph;
}

namespace data_converter {
static mindspore::HashMap<std::string, ValuePtr> object_map_;

static mindspore::OrderedMap<std::string, std::vector<FuncGraphPtr>> object_graphs_map_;

void SetObjGraphValue(const std::string &obj_key, const FuncGraphPtr &data) {
  object_graphs_map_[obj_key].push_back(data);
  MS_LOG(DEBUG) << "Set func graph size: " << object_graphs_map_.size();
}

const mindspore::OrderedMap<std::string, std::vector<FuncGraphPtr>> &GetObjGraphs() {
  MS_LOG(DEBUG) << "Obj graphs size: " << object_graphs_map_.size();
  return object_graphs_map_;
}

void CacheObjectValue(const std::string &obj_key, const ValuePtr &data) { object_map_[obj_key] = data; }

bool GetObjectValue(const std::string &obj_key, ValuePtr *const data) {
  if (object_map_.count(obj_key) != 0) {
    *data = object_map_[obj_key];
    return true;
  }
  return false;
}

std::vector<std::string> GetObjKey(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::tuple obj_tuple = python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_KEY, obj);
  if (obj_tuple.size() != 2) {
    MS_LOG(INTERNAL_EXCEPTION) << "The function of \'get_obj_key()\' must return 2 elements";
  }
  return {py::cast<std::string>(obj_tuple[0]), py::cast<std::string>(obj_tuple[1])};
}

// Get obj detail type
ResolveType GetObjType(const py::object &obj) {
  try {
    py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
    auto obj_type = ResolveType(python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_TYPE, obj).cast<int32_t>());
    return obj_type;
  } catch (const py::error_already_set &ex) {
    MS_LOG(ERROR) << "Meet a exception from Python when get the type of \'" << py::str(obj) << "\'.\n" << ex.what();
    std::rethrow_exception(std::current_exception());
  } catch (const py::type_error &ex) {
    MS_LOG(ERROR) << "Meet a exception when get the type of \'" << py::str(obj) << "\'.\n" << ex.what();
    std::rethrow_exception(std::current_exception());
  }
}

// Get class instance detail type.
ClassInstanceType GetClassInstanceType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto class_type =
    ClassInstanceType(python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_CLASS_INSTANCE_TYPE, obj).cast<int32_t>());
  return class_type;
}

// Check if the object is Cell instance.
bool IsCellInstance(const py::object &obj) {
  auto class_type = GetClassInstanceType(obj);
  return class_type == CLASS_INSTANCE_TYPE_CELL;
}

// Check if the object is Numpy Array instance.
bool IsNumpyArrayInstance(const py::object &obj) {
  auto class_type = GetClassInstanceType(obj);
  return class_type == CLASS_INSTANCE_TYPE_NUMPY_ARRAY;
}

// Check if the object is MsClass instance.
bool IsMsClassInstance(const py::object &obj) { return py::hasattr(obj, PYTHON_MS_CLASS); }

// Check if the object is jit forbidden api.
bool IsJITForbiddenAPI(const py::object &obj) { return py::hasattr(obj, PYTHON_JIT_FORBIDDEN); }

// Check if the object is class type.
bool IsClassType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  return python_adapter::CallPyModFn(mod, PYTHON_MOD_IS_CLASS_TYPE, obj).cast<bool>();
}

// Create the python class instance.
py::object CreatePythonObject(const py::object &type, const py::tuple &args_kwargs) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  // `args_kwargs` maybe a tuple(*args), tuple(**kwargs), or tuple(*args, **kwargs).
  return args_kwargs.empty() ? python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_INSTANCE, type)
                             : python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_INSTANCE, type, args_kwargs);
}

// Call the python script string.
py::object CallPythonScript(const py::object &script, const py::tuple &args_kwargs) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  // `args_kwargs` is a tuple(dict(global), dict(local)).
  return python_adapter::CallPyModFn(mod, PYTHON_MOD_EVAL_PY_SCRIPT, script, args_kwargs);
}

// Get the ids of python script string.
py::set GetPythonScriptIdAttrs(const py::object &script) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  return python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_SCRIPT_ID_ATTRS, script);
}

ValuePtr PyDataToValue(const py::object &obj) {
  py::object to_convert = obj;
  ValuePtr value = nullptr;
  (void)ConvertData(to_convert, &value);
  return value;
}

ValuePtr PyDataToStubNode(const py::object &obj) {
  py::object to_convert = obj;
  ValuePtr value = nullptr;
  (void)ConvertStubData(to_convert, &value);
  return value;
}

void SetFuncGraphByCellObj(const FuncGraphPtr &func_graph, const py::object &obj) {
  // if the cell object has specified bprop, it has user-defined bprop function parse and record it
  if (py::hasattr(obj, CUSTOM_BPROP_NAME)) {
    bool enable_bprop_debug = py::cast<bool>(py::getattr(obj, "bprop_debug"));
    FuncGraphPtr bprop_graph =
      enable_bprop_debug ? ConvertToBpropCut(obj) : ConvertToFuncGraph(obj, {}, PYTHON_MOD_GET_BPROP_METHOD);
    if (bprop_graph != nullptr) {
      (void)func_graph->transforms().emplace(CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph));
      (void)bprop_graph->transforms().emplace("primal", FuncGraphTransform(func_graph));
      func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
      func_graph->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
    }
  }
  if (py::hasattr(obj, STAGE_NAME)) {
    auto stage = py::cast<int>(py::getattr(obj, STAGE_NAME));
    func_graph->set_stage(stage);
  }
  if (py::hasattr(obj, SEGMENT_NAME)) {
    auto segment = py::cast<int>(py::getattr(obj, SEGMENT_NAME));
    func_graph->set_segment(segment);
  }
  auto cell = py::cast<CellPtr>(obj);
  if (cell != nullptr && cell->HasAttr(kAttrRandomOpSnapShot)) {
    auto value = cell->GetAttr(kAttrRandomOpSnapShot);
    MS_EXCEPTION_IF_NULL(value);
    func_graph->set_attr(kAttrRandomOpSnapShot, value);
  }
}

void ClearObjectCache() {
  object_map_.clear();
  object_graphs_map_.clear();
}
}  // namespace data_converter

ValuePtr DataConverter::ConvertData(const py::object &obj) {
  const auto &convert_funcs = GetDataConvertFuncs();
  for (auto &convert_func : convert_funcs) {
    if (convert_func->Matched(obj)) {
      return convert_func->ConvertPyObject(obj, use_signature_, dtype_, args_value_list_);
    }
  }
  return ConvertOtherObj(obj, forbid_reuse_);
}
}  // namespace parse
}  // namespace mindspore
