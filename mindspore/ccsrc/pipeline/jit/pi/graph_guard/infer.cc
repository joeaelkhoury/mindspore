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
#include "pipeline/jit/pi/graph_guard/infer.h"
#include <map>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/anf.h"
#include "utils/flags.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "frontend/operator/composite/composite.h"
#include "ir/cell.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/pi/pydef.h"

namespace mindspore {
namespace parse {
extern bool ConvertData(const py::object &obj, mindspore::ValuePtr *data, bool use_signature,
                        const mindspore::TypePtr &dtype, bool forbid_reuse);
}

namespace abstract {
extern mindspore::abstract::AbstractBasePtr ToAbstract(const mindspore::ValuePtr &value,
                                                       const mindspore::abstract::AnalysisContextPtr &context,
                                                       const mindspore::abstract::AnfNodeConfigPtr &conf);
extern std::optional<StandardPrimitiveImplReg> GetPrimitiveInferImpl(const PrimitivePtr &primitive);
}  // namespace abstract

namespace jit {
namespace graph {

static InferEnginePtr g_pInferEngine = nullptr;

InferEnginePtr InferEngine::GetInstance() {
  if (g_pInferEngine == nullptr) {
    g_pInferEngine = std::shared_ptr<InferEngine>(new InferEngine());
  }
  if (g_pInferEngine->Init()) {
    return g_pInferEngine;
  } else {
    return nullptr;
  }
}

InferEngine::InferEngine() {}

static PyObject *g_ms_module = nullptr;
static PyObject *g_ms_type = nullptr;
static PyObject *g_tensor_type = nullptr;

static bool InitMsModule() {
  if (g_ms_module == nullptr) {
    g_ms_module = PyImport_ImportModule("mindspore");
  }
  return g_ms_module != nullptr && g_ms_module != Py_None;
}

static bool InitMsType() {
  if (g_ms_type == nullptr) {
    g_ms_type = PyImport_ImportModule("mindspore.common.dtype");
  }
  return g_ms_type != nullptr && g_ms_type != Py_None;
}

static bool InitMsTensor() {
  if (g_tensor_type == nullptr && InitMsModule()) {
    g_tensor_type = PyObject_GetAttrString(g_ms_module, "Tensor");
  }
  return g_tensor_type != nullptr && g_tensor_type != Py_None && PyType_Check(g_tensor_type);
}

bool InferEngine::Init() {
  if (!bInit_) {
    bInit_ = InitMsModule() && InitMsType() && InitMsTensor();
  }
  return bInit_;
}

bool InferEngine::Deinit() {
  if (bInit_) {
    Py_XDECREF(g_tensor_type);
    Py_XDECREF(g_ms_type);
    Py_XDECREF(g_ms_module);
    bInit_ = false;
  }
  return bInit_;
}

static std::map<mindspore::TypeId, std::string> g_type2attr = {
  {mindspore::kNumberTypeBool, "bool_"},          {mindspore::kNumberTypeInt, "int_"},
  {mindspore::kNumberTypeInt4, "int_"},           {mindspore::kNumberTypeInt8, "int8"},
  {mindspore::kNumberTypeInt16, "int16"},         {mindspore::kNumberTypeInt32, "int32"},
  {mindspore::kNumberTypeInt64, "int64"},         {mindspore::kNumberTypeUInt, "uint"},
  {mindspore::kNumberTypeUInt8, "uint8"},         {mindspore::kNumberTypeUInt16, "uint16"},
  {mindspore::kNumberTypeUInt32, "uint32"},       {mindspore::kNumberTypeUInt64, "uint64"},
  {mindspore::kNumberTypeFloat, "float_"},        {mindspore::kNumberTypeFloat16, "float16"},
  {mindspore::kNumberTypeFloat32, "float32"},     {mindspore::kNumberTypeFloat64, "float64"},
  {mindspore::kNumberTypeDouble, "float64"},      {mindspore::kNumberTypeComplex, "complex128"},
  {mindspore::kNumberTypeComplex64, "complex64"}, {mindspore::kNumberTypeComplex128, "complex128"},
};

static py::object MakeObjectFromAbstract(const mindspore::abstract::BaseShapePtr &base_shape,
                                         const mindspore::TypePtr &type, bool *is_abstract);

static py::object CreateMetaTensor(const ShapeVector &shape, const mindspore::TypePtr &type) {
  mindspore::TypePtr dtype;
  if (type->isa<mindspore::TensorType>()) {
    dtype = type->cast<mindspore::TensorTypePtr>()->element();
  } else {
    dtype = type;
  }
  /**
   * NOTE: here create a lazy initialized tensor, avoid allocate data
   */
  auto tensor = std::make_shared<mindspore::tensor::Tensor>(dtype->type_id(), shape);
  py::object pytensor = py::reinterpret_borrow<py::object>(g_tensor_type);
  return pytensor(py::cast(tensor));
}

static py::object CreateMetaTensor(const mindspore::abstract::ShapePtr &shape, const mindspore::TypePtr &type) {
  return CreateMetaTensor(shape->shape(), type);
}

static py::object CreateScalar(const mindspore::TypePtr &type) {
  static std::map<mindspore::TypeId, py::object> ms_type2py_type_map = {
    {mindspore::kNumberTypeBool, py::bool_()},
    {mindspore::kNumberTypeInt, py::int_()},
    {mindspore::kNumberTypeInt4, py::int_()},
    {mindspore::kNumberTypeInt8, py::int_()},
    {mindspore::kNumberTypeInt16, py::int_()},
    {mindspore::kNumberTypeInt32, py::int_()},
    {mindspore::kNumberTypeInt64, py::int_()},
    {mindspore::kNumberTypeUInt, py::int_()},
    {mindspore::kNumberTypeUInt8, py::int_()},
    {mindspore::kNumberTypeUInt16, py::int_()},
    {mindspore::kNumberTypeUInt32, py::int_()},
    {mindspore::kNumberTypeUInt64, py::int_()},
    {mindspore::kNumberTypeFloat, py::float_()},
    {mindspore::kNumberTypeFloat16, py::float_()},
    {mindspore::kNumberTypeFloat32, py::float_()},
    {mindspore::kNumberTypeFloat64, py::float_()},
    {mindspore::kNumberTypeDouble, py::float_()},
    {mindspore::kNumberTypeComplex, py::reinterpret_steal<py::object>(PyComplex_FromDoubles(0.0, 0.0))},
    {mindspore::kNumberTypeComplex64, py::reinterpret_steal<py::object>(PyComplex_FromDoubles(0.0, 0.0))},
    {mindspore::kNumberTypeComplex128, py::reinterpret_steal<py::object>(PyComplex_FromDoubles(0.0, 0.0))},
  };
  auto it = ms_type2py_type_map.find(type->type_id());
  if (it != ms_type2py_type_map.cend()) {
    return it->second;
  } else {
    return py::cast<py::object>(nullptr);
  }
}

static py::object CreateTuple(const mindspore::abstract::BaseShapePtr &base_shape, const mindspore::TypePtr &type,
                              bool *is_abstract) {
  bool dynamic;
  mindspore::abstract::SequenceShapePtr shape_tuple;
  size_t elem_count = 0;
  auto type_tuple = type->cast_ptr<mindspore::Tuple>();
  if (base_shape->isa<mindspore::abstract::DynamicSequenceShape>()) {
    dynamic = true;
    elem_count = type_tuple->elements().size();
  } else {
    dynamic = false;
    shape_tuple = base_shape->cast<mindspore::abstract::TupleShapePtr>();
    elem_count = shape_tuple->size();
  }
  py::tuple tuple = py::tuple(elem_count);
  for (size_t it = 0; it < elem_count; ++it) {
    bool is_abstract_obj = false;
    auto tensor_it =
      MakeObjectFromAbstract(dynamic ? base_shape : (*shape_tuple)[it], type_tuple->elements()[it], &is_abstract_obj);
    Py_INCREF(tensor_it.ptr());
    PyTuple_SetItem(tuple.ptr(), it, tensor_it.ptr());
    *is_abstract |= is_abstract_obj;
  }
  return tuple;
}

static py::object CreateList(const mindspore::abstract::BaseShapePtr &base_shape, const mindspore::TypePtr &type,
                             bool *is_abstract) {
  bool dynamic;
  mindspore::abstract::SequenceShapePtr shape_list;
  size_t elem_count = 0;
  auto type_list = type->cast_ptr<mindspore::List>();
  if (base_shape->isa<mindspore::abstract::DynamicSequenceShape>()) {
    dynamic = true;
    elem_count = type_list->elements().size();
  } else {
    dynamic = false;
    shape_list = base_shape->cast<mindspore::abstract::ListShapePtr>();
    elem_count = shape_list->size();
  }
  py::list list = py::list(elem_count);
  for (size_t it = 0; it < elem_count; ++it) {
    bool is_abstract_obj = false;
    auto tensor_it =
      MakeObjectFromAbstract(dynamic ? base_shape : (*shape_list)[it], type_list->elements()[it], &is_abstract_obj);
    Py_INCREF(tensor_it.ptr());
    PyList_SetItem(list.ptr(), it, tensor_it.ptr());
    *is_abstract |= is_abstract_obj;
  }
  return list;
}

static py::object MakeObjectFromAbstract(const mindspore::abstract::BaseShapePtr &base_shape,
                                         const mindspore::TypePtr &type, bool *is_abstract) {
  *is_abstract = false;
  if (base_shape->isa<mindspore::abstract::Shape>()) {
    return CreateMetaTensor(base_shape->cast<mindspore::abstract::ShapePtr>(), type);
  } else if (base_shape->isa<mindspore::abstract::NoShape>() && type->isa<mindspore::Number>()) {
    *is_abstract = true;
    return CreateScalar(type);
  } else if (base_shape->isa<mindspore::abstract::TupleShape>() && type->isa<mindspore::Tuple>()) {
    return CreateTuple(base_shape, type, is_abstract);
  } else if (base_shape->isa<mindspore::abstract::ListShape>() && type->isa<mindspore::List>()) {
    return CreateList(base_shape, type, is_abstract);
  } else if (base_shape->isa<mindspore::abstract::NoShape>() && type->isa<mindspore::TypeNone>()) {
    // AbstractNone indicates there is no output for this CNode node.
    return py::cast<py::object>(Py_None);
  } else if (type->isa<mindspore::Monad>()) {
    // Return monad abstract if it is monad type.
    return py::cast<py::object>(nullptr);
  } else if (base_shape->isa<mindspore::abstract::DynamicSequenceShape>()) {
    *is_abstract = true;
    if (type->isa<mindspore::Tuple>()) {
      return CreateTuple(base_shape, type, is_abstract);
    } else if (type->isa<mindspore::List>()) {
      return CreateList(base_shape, type, is_abstract);
    } else if (type->isa<mindspore::TensorType>()) {
      return CreateMetaTensor({-2}, type);
    } else if (type->isa<mindspore::Number>()) {
      return CreateScalar(type);
    } else {
      MS_LOG(EXCEPTION) << "Evaluator return invalid shape " << base_shape->ToString() << " or type. "
                        << type->ToString();
      return py::cast<py::object>(nullptr);
    }
  } else {
    MS_LOG(EXCEPTION) << "Evaluator return invalid shape " << base_shape->ToString() << " or type. "
                      << type->ToString();
    return py::cast<py::object>(nullptr);
  }
}

static py::object MakeObjectFromPyObject(const py::object &shape_obj, const py::object &type_obj, bool *is_abstract) {
  *is_abstract = false;
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto res_vec = shape_obj.cast<ShapeVector>();
    auto res_dtype = type_obj.cast<TypePtr>();
    if (res_vec.empty() && (!res_dtype->isa<TensorType>())) {
      *is_abstract = true;
      return CreateScalar(res_dtype);
    }
    return CreateMetaTensor(res_vec, res_dtype);
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    auto typeid_tuple = type_obj.cast<py::tuple>();
    py::tuple ptr_list(typeid_tuple.size());
    for (size_t it = 0; !(*is_abstract) && it < typeid_tuple.size(); ++it) {
      py::object tmp =
        MakeObjectFromPyObject(shape_obj.cast<py::tuple>()[it], type_obj.cast<py::tuple>()[it], is_abstract);
      ptr_list[it] = tmp;
    }
    return ptr_list;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    auto typeid_list = type_obj.cast<py::list>();
    py::list ptr_list;
    for (size_t it = 0; !(*is_abstract) && it < typeid_list.size(); ++it) {
      py::object tmp =
        MakeObjectFromPyObject(shape_obj.cast<py::list>()[it], type_obj.cast<py::list>()[it], is_abstract);
      ptr_list.append(tmp);
    }
    return ptr_list;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    return py::cast<py::object>(Py_None);
  } else if (py::isinstance<Type>(type_obj) && type_obj.cast<Type *>()->isa<MonadType>()) {
    return py::cast<py::object>(nullptr);
  } else {
    MS_LOG(EXCEPTION) << "Python evaluator return invalid shape or type. " << py::str(type_obj);
  }
}

static bool HasTensor(py::object obj) {
  if (obj.ptr() == nullptr) {
    return false;
  }

  ReprRecursionScope scope(obj.ptr());
  if (scope.ReEnterOrError()) {
    return false;
  }
  if (py::isinstance<mindspore::tensor::MetaTensor>(obj)) {
    return true;
  } else if (py::isinstance<py::list>(obj)) {
    auto list_obj = py::cast<py::list>(obj);
    if (std::any_of(list_obj.begin(), list_obj.end(),
                    [](const auto &e) { return HasTensor(py::cast<py::object>(e)); })) {
      return true;
    }
  } else if (py::isinstance<py::tuple>(obj)) {
    auto tuple_obj = py::cast<py::tuple>(obj);
    if (std::any_of(tuple_obj.begin(), tuple_obj.end(),
                    [](const auto &e) { return HasTensor(py::cast<py::object>(e)); })) {
      return true;
    }
  } else if (py::isinstance<py::dict>(obj)) {
    auto dict_obj = py::cast<py::dict>(obj);
    if (std::any_of(dict_obj.begin(), dict_obj.end(), [](const auto &e) {
          return HasTensor(py::cast<py::object>(e.first)) || HasTensor(py::cast<py::object>(e.second));
        })) {
      return true;
    }
  }
  return false;
}

static AbstractBasePtrList ChangAbstractArgList(std::vector<PyObject *> args, bool *has_tensor, int *monad_count) {
  AbstractBasePtrList list;
  for (size_t i = 0; i < args.size(); ++i) {
    mindspore::ValuePtr converted = nullptr;
    py::object param_obj = py::reinterpret_borrow<py::object>(args[i]);
    if (IsStubTensor(param_obj)) {
      param_obj = python_adapter::CallPyObjMethod(param_obj, "stub_sync");
      args[i] = param_obj.ptr();
    } else if (py::isinstance<Monad>(param_obj)) {
      *monad_count = *monad_count + 1;
    }
    *has_tensor = HasTensor(param_obj);
    if (!mindspore::parse::ConvertData(param_obj, &converted, false, nullptr, false)) {
      MS_LOG(EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i << "]: " << py::str(param_obj);
      break;
    }
    auto arg = mindspore::abstract::ToAbstract(converted, nullptr, nullptr);
    list.push_back(arg);
  }
  return list;
}

// return new reference
PyObject *InferEngine::InferPrimitive(PyObject *primitive, const std::vector<PyObject *> &args, bool *is_abstract) {
  if (!Init()) {
    return nullptr;
  }
  int monad_count = 0;
  bool has_tensor = false;
  std::vector<PyObject *> arglist = args;
  AbstractBasePtrList list = ChangAbstractArgList(arglist, &has_tensor, &monad_count);
  py::object adapter_obj = py::reinterpret_borrow<py::object>(primitive);
  mindspore::PrimitivePyAdapterPtr prim_adapter = adapter_obj.cast<mindspore::PrimitivePyAdapterPtr>();
  mindspore::PrimitivePyPtr prim = prim_adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<mindspore::PrimitivePy>(adapter_obj);
    prim_adapter->set_attached_primitive(prim);
  }
  *is_abstract = false;
  auto eval_impl = mindspore::abstract::GetPrimitiveInferImpl(prim);
  if (eval_impl != std::nullopt && eval_impl->Get().get() != nullptr) {
    PyObject *special_type = InferSpecialPrimitive(primitive, arglist, prim);
    if (special_type == nullptr) {
      mindspore::abstract::BaseShapePtr shape = eval_impl->InferShape(prim, list);
      mindspore::TypePtr type = eval_impl->InferType(prim, list);
      auto pyObj = MakeObjectFromAbstract(shape, type, is_abstract);
      return pyObj.inc_ref().ptr();
    } else {
      return special_type;
    }
  } else if (prim->HasPyObj()) {
    if (py::hasattr(adapter_obj, PY_PRIM_METHOD_INFER)) {
      py::tuple py_vals(arglist.size() - monad_count);
      for (size_t i = 0; i < arglist.size() - monad_count; ++i) {
        py_vals[i] = py::reinterpret_borrow<py::object>(arglist[i]);
      }
      py::dict output = prim->RunInfer(py_vals);
      if (output[ATTR_VALUE].is_none()) {
        auto ret = MakeObjectFromPyObject(output[ATTR_SHAPE], output[ATTR_DTYPE], is_abstract);
        Py_INCREF(ret.ptr());
        return ret.ptr();
      } else {
        Py_INCREF(output[ATTR_VALUE].ptr());
        return output[ATTR_VALUE].ptr();
      }
    } else if (!has_tensor && py::hasattr(adapter_obj, PY_PRIM_METHOD_INFER_VALUE)) {
      // Tensor maybe uninitialized, avoid infer value and allocate data.
      // because tensor has no data when doing inference for type, infer_value will crash!
      py::tuple py_vals(arglist.size());
      for (size_t i = 0; i < arglist.size(); ++i) {
        py_vals[i] = py::reinterpret_borrow<py::object>(arglist[i]);
      }
      auto output = prim->RunInferValue(py_vals);
      Py_INCREF(output.ptr());
      return output.ptr();
    }
    return nullptr;
  }
  return nullptr;
}

PyObject *InferEngine::InferSpecialPrimitive(PyObject *primitive, const std::vector<PyObject *> &arglist,
                                             const PrimitivePyPtr &prim) {
  if ((prim->name() == "Shape" || prim->name() == "DType" || prim->name() == "Rank") && arglist.size() == 1) {
    PyObject *res = PyObject_CallOneArg(primitive, arglist[0]);
    if (res == nullptr) {
      MS_LOG(ERROR) << py::error_already_set().what();
      PyErr_Clear();
    }
    return res;
  } else if (prim->name() == "TileSize" && arglist.size() == 3) {
    py::tuple tuple(3);
    tuple[0] = py::cast<py::object>(arglist[0]);
    tuple[1] = py::cast<py::object>(arglist[1]);
    tuple[2] = py::cast<py::object>(arglist[2]);
    PyObject *t = PyObject_Call(primitive, tuple.ptr(), nullptr);
    if (PyErr_Occurred()) {
      PyObject *et, *ev, *tb;
      PyErr_Fetch(&et, &ev, &tb);
      MS_LOG(EXCEPTION) << "Shape infer [TileSize] failed " << std::string(py::str(ev));
      PyErr_Clear();
    }
    return t;
  } else {
    return nullptr;
  }
}

bool InferEngine::SupportInfer(PyObject *primitive) {
  if (!Init()) {
    return false;
  }
  py::object adapter_obj = py::reinterpret_borrow<py::object>(primitive);
  mindspore::PrimitivePyAdapterPtr prim_adapter = adapter_obj.cast<mindspore::PrimitivePyAdapterPtr>();
  mindspore::PrimitivePyPtr prim = prim_adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<mindspore::PrimitivePy>(adapter_obj);
    prim_adapter->set_attached_primitive(prim);
  }
  auto eval_impl = mindspore::abstract::GetPrimitiveInferImpl(prim);
  if (eval_impl != std::nullopt && eval_impl->Get().get() != nullptr) {
    return true;
  } else {
    return false;
  }
}

#define TYPE_CHECK(mod_name, type_name, check_sub_type)                             \
  py::object cls = Utils::GetModuleAttr(mod_name, type_name);                       \
  MS_EXCEPTION_IF_CHECK_FAIL(PyType_Check(cls.ptr()), "must be type");              \
  bool check_res = reinterpret_cast<PyObject *>(tp) == cls.ptr();                   \
  if (!check_res && (check_sub_type)) {                                             \
    check_res |= PyType_IsSubtype(tp, reinterpret_cast<PyTypeObject *>(cls.ptr())); \
  }                                                                                 \
  return check_res;

// sub-type check
template <>
bool IsGradOperationType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::prim::GradOperation, true>(tp);
}
template <>
bool IsVmapOperationType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::prim::VmapOperation, true>(tp);
}
template <>
bool IsShardType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::prim::Shard, true>(tp);
}
template <>
bool IsStubTensorType<true>(PyTypeObject *tp) {
  TYPE_CHECK("mindspore.common._stub_tensor", "StubTensor", true);
}
template <>
bool IsTensorType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::tensor::MetaTensor, true>(tp);
}
template <>
bool IsCellType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::Cell, true>(tp);
}
template <>
bool IsPrimitiveType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::PrimitivePyAdapter, true>(tp);
}
template <>
bool IsMetaFuncGraphType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::MetaFuncGraph, true>(tp);
}
template <>
bool IsMSDTypeType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::Type, true>(tp);
}
// exact type check
template <>
bool IsCellListType<false>(PyTypeObject *tp) {
  TYPE_CHECK("mindspore.nn", "CellList", false);
}

#undef TYPE_CHECK

bool CheckTensorDataInitialized(const py::object &py_tensor) {
  if (py::isinstance<mindspore::tensor::Tensor>(py_tensor)) {
    auto tensor = py_tensor.cast<mindspore::tensor::TensorPtr>();
    return tensor->data().const_data() != nullptr;
  }
  return false;
}

bool FindTensorName(const std::string &name) {
  const auto &meth = pipeline::GetMethodMap().find(kObjectTypeTensorType)->second;
  if (meth.find(name) != meth.end()) {
    return true;
  }
  const auto &attr = pipeline::GetAttrMap().find(kObjectTypeTensorType)->second;
  return attr.find(name) != attr.end();
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
