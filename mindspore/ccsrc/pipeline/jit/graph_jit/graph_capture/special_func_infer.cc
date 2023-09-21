
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
#include "pipeline/jit/graph_jit/graph_capture/special_func_infer.h"
#include <string>
#include <memory>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <vector>
#include "pipeline/jit/graph_jit/common.h"
#include "pipeline/jit/graph_jit/external.h"
#include "pipeline/jit/graph_jit/graph_capture/graph_build.h"
#include "pipeline/jit/graph_jit/graph_guard/infer.h"

namespace mindspore {
namespace jit {
namespace graph {
using CheckFunc = bool (*)(const py::object &);
using InferFunc = bool (*)(CallNode *);
struct SpecialAction {
  CheckFunc check;
  InferFunc infer;
};

extern AObject *InferFuncResult(const py::object &func, const std::vector<AObject *> &stack_args, int opcode,
                                const GraphJitConfig &conf, bool clear_guard);

extern AObject *InferFuncResult(const py::object &func, const py::object &args, const py::object &kwargs,
                                const GraphJitConfig &conf, bool clear_guard);

// ------------------------------builtins functions--------------------------------
static constexpr const char *kBuiltinNameIsinstance = "isinstance";  // call __instancecheck__
static constexpr const char *kBuiltinNameIssubclass = "issubclass";  // call __subclasscheck__
static constexpr const char *kBuiltinNameLen = "len";                // call __len__
static constexpr const char *kBuiltinNameAbs = "abs";                // call __abs__
static constexpr const char *kBuiltinNameAll = "all";                // for each value in the iterable. call __bool__
static constexpr const char *kBuiltinNameAny = "any";                // for each value in the iterable. call __bool__
static constexpr const char *kBuiltinNameHash = "hash";              // call __hash__
static constexpr const char *kBuiltinNameId = "id";                  // no side effects
static constexpr const char *kBuiltinNameOrd = "ord";                // convert char to int. no side effect
static constexpr const char *kBuiltinNameGlobals = "globals";        // global variables. no side effects
static constexpr const char *kBuiltinNameCallable = "callable";      // no side effects
static constexpr const char *kBuiltinNameGetattr = "getattr";        // call __getattr__, or __getattribute__
static constexpr const char *kBuiltinNameHasattr = "hasattr";        // call __getattr__, or __getattribute__
// ------------------------------builtins functions--------------------------------

// ------------------------------builtins method--------------------------------
// static constexpr const char *kBuiltinNameUpdate = "update";  // dict update
// static constexpr const char *kBuiltinNameAppend = "append";  // list update
// ------------------------------builtins method--------------------------------

// ------------------------------mindspore functions-------------------------------
static constexpr const char *kMindsporeNameGetCachePrim = "_get_cache_prim";
/**
 * NOTE: mindspore/ops/composite/base.py, after_grad decorated by '_warp_func'
 * code name is 'wrapper', not 'after_grad', it only called by pynative
 */
static constexpr const char *kMindsporeNameGradFunc = "after_grad";
static constexpr const char *kMindsporeNameJitFunc = "staging_specialize";  // mindspore.jit

static constexpr const char *kMindsporeNamePrimitive = "Primitive_";
static constexpr const char *kMindsporeNameMetaFuncGraph = "MetaFuncGraph_";
static constexpr const char *kMindsporeNameMsCell = "mindspore.nn.Cell";
static constexpr const char *kMindsporeNameTensorMethod = "Tensor.__method__";
/**
 * standard method
 * except_module = ['mindspore.ops.operations', 'mindspore.ops.functional', 'mindspore.ops.composite', \
 *                  'mindspore._extends.parse.standard_method']
 * any([True for x in except_module if func.__module__.startswith(x)])
 */
static constexpr const char *kMindsporeNameStandardFunction = "mindspore.P.C.F.standard";
/**
 * convert function map
 * refer to convert_object_map in mindspore._extends.parse.resources.py
 */
static constexpr const char *kMindsporeNameConvertMap = "mindspore._extends.parse.resources.convert_object_map";
// ------------------------------mindspore functions-------------------------------

static py::object GetGradClass() { return Utils::GetModuleAttr("mindspore._c_expression", "GradOperation_"); }

template <AObject::Type type>
bool SetCallResType(CallNode *call_node) {
  call_node->setVobj(AObject::MakeAObject(type));
  call_node->setSubGraph(nullptr);
  return false;
}

static bool check_ConvertMap(const py::object &func) {
  if (func.ptr() == nullptr || !PyFunction_Check(func.ptr())) {
    return false;
  }
  py::object tmp = Utils::GetModuleAttr("mindspore._extends.parse.resources", "convert_object_map");
  auto dict_obj = py::cast<py::dict>(tmp);
  if (dict_obj.contains(func)) {
    return true;
  } else {
    return false;
  }
}

static bool infer_ConvertMap(CallNode *call_node) {
  AObject *func_info = call_node->input(0)->getVobj();
  func_info->SetMsFlag(AObject::kMsFlagStandardFunc);
  py::object func = func_info->GetPyObject();
  py::object tmp = Utils::GetModuleAttr("mindspore._extends.parse.resources", "convert_object_map");
  auto dict_obj = py::cast<py::dict>(tmp);
  auto infer_obj = dict_obj[func];
  AObject *res = nullptr;
  call_node->setSubGraph(nullptr);
  SetCallResType<AObject::kTypeTensor>(call_node);
  if (PyFunction_Check(infer_obj.ptr())) {
    MS_LOG(DEBUG) << "infer function " << std::string(py::str(PyFunction_GET_CODE(infer_obj.ptr())));
    int op = call_node->getOpcode();
    const auto &conf = call_node->GetGraph()->Config();
    std::vector<AObject *> args;
    std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                   [](ValueNode *n) { return n->getVobj(); });
    res = InferFuncResult(func, {args.begin() + 1, args.end()}, op, conf, true);
  } else if (IsPrimitiveTypeOrSubType(Py_TYPE(infer_obj.ptr()))) {
    MS_LOG(DEBUG) << "infer primitive " << std::string(py::str(infer_obj));
    std::vector<PyObject *> list;
    bool infer_fail = false;
    for (size_t i = 1; !infer_fail && i < call_node->getInputs().size(); i++) {
      AObject *p = call_node->input(i)->getVobj();
      PyObject *o = p ? p->GetPyObject().ptr() : nullptr;
      list.push_back(o);
      infer_fail = o == nullptr;
    }
    if (infer_fail) {
      return false;
    }
    auto inst = mindspore::jit::graph::InferEngine::GetInstance();
    bool is_abstract = false;
    PyObject *ret = inst->InferPrimitive(infer_obj.ptr(), list, &is_abstract);
    if (ret == nullptr) {
      return false;
    }
    AObject::Type type = AObject::GetPyType(ret);
    res = is_abstract ? AObject::MakeAObject(type) : AObject::Convert(ret);
    Py_DECREF(ret);
  } else {
    return false;
  }
  if (res) {
    call_node->setVobj(res);
  }
  return false;
}

bool check__get_cache_prim(const py::object &f) {
  py::object tmp = Utils::GetModuleAttr("mindspore.ops._primitive_cache", kMindsporeNameGetCachePrim);
  return tmp.ptr() && tmp.ptr() == f.ptr();
}

bool infer__get_cache_prim(CallNode *n) {
  Graph *g = n->getSubGraph();
  n->setVobj(n->input(1)->getVobj());
  g->setRetVal(n->input(1));

  // extract operation
  auto &alloc = g->allocator();
  AbstractNodeList b = {nullptr, nullptr};
  b.push_back(alloc.NewInstrNode(ROT_TWO, 0));
  b.push_back(alloc.NewInstrNode(POP_TOP, 0));
  n->setExtraOper(reinterpret_cast<InstrNode *>(b.head()));
  return true;
}

static bool builtins_module_check(PyObject *m) {
  return m && PyModule_Check(m) && !strcmp(PyModule_GetName(m), "builtins");
}

bool check_builtin_cfunc(const py::object &f) {
  PyObject *func = f.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (!PyCFunction_Check(func)) {
    return false;
  }
  return builtins_module_check(reinterpret_cast<PyCFunctionObject *>(func)->m_self);
}

bool infer_builtin_len(CallNode *n) {
  auto g = n->getSubGraph();
  n->setSubGraph(nullptr);
  n->setVobj(AObject::MakeAObject(AObject::kTypeInt));
  AObject *arg = n->input(1)->getVobj();
  if (!arg) {
    return false;
  }
  Py_ssize_t siz = 0;
  switch (arg->GetType()) {
    case AObject::kTypeTuple:
    case AObject::kTypeList:
      siz = static_cast<AbstractTuple *>(arg)->size();
      break;
    case AObject::kTypeDict:
      siz = static_cast<AbstractDict *>(arg)->size();
      break;
    default:
      if (!arg->GetPyObject().ptr()) {
        return false;
      }
      siz = PyObject_Size(arg->GetPyObject().ptr());
      break;
  }
  if (siz < 0) {
    PyErr_Clear();
    return false;
  }
  AObject *res = AObject::Convert(py::int_(siz));
  n->setVobj(res);
  if (g->GuardValueNode(n)) {
    n->setSubGraph(g);
    auto &alloc = g->allocator();
    auto retVal = alloc.NewValueNode(res, LOAD_CONST, -1, {});
    retVal->setGraph(n->GetGraph());
    g->addInstr(retVal);
    g->addInstr(alloc.NewInstrNode(RETURN_VALUE, 0));
    g->setRetVal(retVal);
    n->setInlineReason(kInline);

    AbstractNodeList b = {nullptr, nullptr};
    b.push_back(n->GetGraph()->allocator().NewInstrNode(POP_TOP, 0));
    b.push_back(n->GetGraph()->allocator().NewInstrNode(POP_TOP, 0));
    n->setExtraOper(reinterpret_cast<InstrNode *>(b.head()));
    return true;
  }
  return false;
}

bool infer_builtin_getattr(CallNode *call_node) {
  call_node->setSubGraph(nullptr);
  ValueNode *load_name = call_node->input(2);
  PyObject *pyname = load_name->getVobj()->GetPyObject().ptr();
  if (!pyname || !PyUnicode_Check(pyname)) {
    // has a python exceptions, do nothing
    return false;
  }
  const char *name = PyUnicode_AsUTF8(pyname);
  AObject *attr = call_node->input(1)->get_attr(name);
  call_node->setVobj(attr);
  return false;
}

static inline bool InferBuiltinOneArg(CallNode *call_node, PyCFunction cpython_func) {
  auto &arg = call_node->input(1)->getVobj();
  if (arg && arg->GetPyObject().ptr() && arg->GetType() != AObject::kTypeAnyValue) {
    py::object res = py::reinterpret_steal<py::object>(cpython_func(nullptr, arg->GetPyObject().ptr()));
    call_node->setVobj(AObject::Convert(res));
    PyErr_Clear();
  }
  call_node->setSubGraph(nullptr);
  return false;
}

#define DECLARE_BUILTIN_CFUNCTION(func_name)                             \
  static PyCFunction cpython_func = nullptr;                             \
  if (!cpython_func) {                                                   \
    PyObject *p = PyDict_GetItemString(PyEval_GetBuiltins(), func_name); \
    MS_ASSERT(p &&PyCFunction_Check(p));                                 \
    cpython_func = PyCFunction_GET_FUNCTION(p);                          \
  }

#define DECLARE_INFER_BUILTIN_ONE_ARG(func_name) \
  [](CallNode *n) {                              \
    DECLARE_BUILTIN_CFUNCTION(func_name);        \
    return InferBuiltinOneArg(n, cpython_func);  \
  }

bool InferBuiltinGlobals(CallNode *call_node) {
  py::object global = call_node->GetGraph()->GetGlobals();
  AObject *res = AObject::Convert(global);
  const char *key = "globals()";
  call_node->GetGraph()->InstallToGlobal(key, global);

  auto g = call_node->getSubGraph();
  auto &alloc = g->allocator();
  auto retVal = alloc.NewValueNode(res, LOAD_GLOBAL, -1, {});
  retVal->setName(key);
  retVal->setGraph(call_node->GetGraph());
  g->addInstr(retVal);
  g->addInstr(alloc.NewInstrNode(RETURN_VALUE, 0));
  g->setRetVal(retVal);
  call_node->setInlineReason(kInline);

  AbstractNodeList b = {nullptr, nullptr};
  b.push_back(call_node->GetGraph()->allocator().NewInstrNode(POP_TOP, 0));
  call_node->setExtraOper(reinterpret_cast<InstrNode *>(b.head()));

  return true;
}

using InstanceSubclassCheckFunc = int (*)(PyObject *, PyObject *);
template <InstanceSubclassCheckFunc pyfunc>
bool InferBuiltinInstanceSubclassCheck(CallNode *call_node) {
  Graph *g = call_node->getSubGraph();
  call_node->setVobj(AObject::MakeAObject(AObject::kTypeBool));
  call_node->setSubGraph(nullptr);
  auto &arg1 = call_node->input(1)->getVobj();
  auto &arg2 = call_node->input(2)->getVobj();
  if (arg1 == nullptr || arg2 == nullptr || arg1->GetPyObject().ptr() == nullptr ||
      arg2->GetPyObject().ptr() == nullptr) {
    return false;
  }
  int stat = pyfunc(arg1->GetPyObject().ptr(), arg2->GetPyObject().ptr());
  if (stat < 0) {
    PyErr_Clear();
    return false;
  }
  AObject *res = AObject::Convert(py::bool_(stat));
  call_node->setVobj(res);
  if (g->GuardValueNode(call_node)) {
    call_node->setSubGraph(g);
    auto &alloc = g->allocator();
    auto retVal = alloc.NewValueNode(res, LOAD_CONST, -1, {});
    retVal->setGraph(call_node->GetGraph());
    g->addInstr(retVal);
    g->addInstr(alloc.NewInstrNode(RETURN_VALUE, 0));
    g->setRetVal(retVal);
    call_node->setInlineReason(kInline);

    AbstractNodeList b = {nullptr, nullptr};
    b.push_back(call_node->GetGraph()->allocator().NewInstrNode(POP_TOP, 0));
    b.push_back(call_node->GetGraph()->allocator().NewInstrNode(POP_TOP, 0));
    b.push_back(call_node->GetGraph()->allocator().NewInstrNode(POP_TOP, 0));
    call_node->setExtraOper(reinterpret_cast<InstrNode *>(b.head()));
    return true;
  }
  return false;
}

static bool support_infer_primitive(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  if (IsPrimitiveTypeOrSubType(Py_TYPE(obj))) {
    auto inst = mindspore::jit::graph::InferEngine::GetInstance();
    return inst->SupportInfer(obj);
  }
  return false;
}

static bool check_primitive(const py::object &func) {
  return AObject::GetPyType(func.ptr()) == AObject::kTypePrimitive;
}

bool infer_primitive(CallNode *call_node) {
  static const std::unordered_map<std::string, AObject::Type> not_ret_tensor_prim = {
    {"Prim[_get_grad_op]<constexpr_prim=True>", AObject::kTypeMetaFuncGraph},
    {"Prim[DType]", AObject::kTypeAnyValue},
    {"Prim[Partial]<side_effect_propagate=1>", AObject::kTypeAnyValue},
  };
  call_node->setVobj(AObject::MakeAObject(AObject::kTypeTensor));
  call_node->setSubGraph(nullptr);
  PyObject *prim = call_node->input(0)->getVobj()->GetPyObject().ptr();
  std::string prim_key = std::string(py::str(prim));
  if (prim_key == "Prim[_get_grad_op]<constexpr_prim=True>") {
    AbstractType *type = static_cast<AbstractType *>(AObject::Convert(GetGradClass()));
    AObject *res = type != nullptr ? type->BuildAbstractInstance({}, CALL_FUNCTION)
                                   : AObject::MakeAObject(AObject::kTypeMetaFuncGraph);
    call_node->setVobj(res);
    return false;
  }

  auto iter = not_ret_tensor_prim.find(prim_key);
  if (iter != not_ret_tensor_prim.end()) {
    call_node->setVobj(AObject::MakeAObject(iter->second));
  } else {
    call_node->setVobj(AObject::MakeAObject(AObject::kTypeTensor));
  }
  if (!support_infer_primitive(prim)) {
    return false;
  }

  std::vector<PyObject *> list;
  bool infer_fail = false;
  for (size_t i = 1; !infer_fail && i < call_node->getInputs().size(); i++) {
    AObject *p = call_node->input(i)->getVobj();
    PyObject *o = p ? p->GetPyObject().ptr() : nullptr;
    list.push_back(o);
    infer_fail = o == nullptr;
  }
  if (infer_fail) {
    return false;
  }

  auto inst = mindspore::jit::graph::InferEngine::GetInstance();
  bool is_abstract = false;
  PyObject *ret;
  try {
    ret = inst->InferPrimitive(prim, list, &is_abstract);
  } catch (std::exception &e) {
    MS_LOG(INFO) << "infer primitive failed. reason:";
    MS_LOG(INFO) << e.what();
    ret = nullptr;
  }
  if (ret == nullptr) {
    return false;
  }
  AObject::Type type = AObject::GetPyType(ret);
  AObject *type_info = is_abstract ? AObject::MakeAObject(type) : AObject::Convert(ret);
  call_node->setVobj(type_info);
  Py_DECREF(ret);
  return false;
}

bool InferGradOperation(CallNode *call_node, AObject::MindsporeFlag f) {
  call_node->setSubGraph(nullptr);
  AObject *grad_func = AObject::MakeAObject(AObject::kTypeFunction);
  grad_func->SetMsFlag(f);
  call_node->setVobj(grad_func);
  py::object func = GraphBuilder::FindPyFunc(call_node->input(1)->getVobj());
  if (func.ptr() == nullptr) {
    return false;
  }
  (void)graph_jit_should_compile(func, py::dict());
  auto jcr = getJitCompileResults(PyFunction_GET_CODE(func.ptr()));
  *jcr->conf = call_node->GetGraph()->Config();
  return false;
}

static bool check_MetaFunc_(const py::object &o) {
  PyTypeObject *tp = PyType_Check(o.ptr()) ? reinterpret_cast<PyTypeObject *>(o.ptr()) : Py_TYPE(o.ptr());
  return IsMetaFuncGraphTypeOrSubType(tp);
}

static bool infer_MetaFunc_(CallNode *call_node) {
  call_node->setSubGraph(nullptr);
  const auto &vo = call_node->input(0)->getVobj();
  if (vo->GetType() == AObject::kTypeType) {
    GraphBuilder::HandleCallClass(call_node);
    return false;
  }
  PyTypeObject *tp = vo->GetTypeObject();
  if (IsGradOperationTypeOrSubType(tp)) {
    // set grad flag
    return InferGradOperation(call_node, AObject::MindsporeFlag::kMsFlagGradFunc);
  } else if (IsVmapOperationTypeOrSubType(tp)) {
    // set vmap flag
    return InferGradOperation(call_node, AObject::MindsporeFlag::kMsFlagVmapFunc);
  } else if (IsShardTypeOrSubType(tp)) {
    // set shard flag
    return InferGradOperation(call_node, AObject::MindsporeFlag::kMsFlagShardFunc);
  }
  return false;
}

static bool check_TensorMethod(const py::object &func) {
  PyObject *src = func.ptr();
  if (src == nullptr) {
    return false;
  }
  if (PyMethod_Check(src)) {
    src = PyMethod_GET_FUNCTION(src);
  }
  if (!PyFunction_Check(src)) {
    return false;
  }
  py::object cls = Utils::GetModuleAttr("mindspore", "Tensor");
  PyObject *tar = PyObject_GetAttr(cls.ptr(), reinterpret_cast<PyFunctionObject *>(src)->func_name);
  bool is_tensor_method = tar && PyFunction_Check(tar) && PyFunction_GET_CODE(tar) == PyFunction_GET_CODE(src);
  Py_XDECREF(tar);
  PyErr_Clear();
  return is_tensor_method;
}

static bool infer_TensorMethod(CallNode *call_node) {
  AObject *callable = call_node->input(0)->getVobj();
  PyObject *func = callable->GetPyObject().ptr();
  // 'check_TensorMethod' ensure the func is function or boundmethod
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  PyObject *func_name = reinterpret_cast<PyFunctionObject *>(func)->func_name;
  py::object standard_func = FindTensorMethodMap(PyUnicode_AsUTF8(func_name));
  if (standard_func.ptr() == nullptr || !PyFunction_Check(standard_func.ptr())) {
    call_node->setVobj(AObject::MakeAObject(AObject::kTypeAnyValue));
    return false;
  }
  MS_LOG(DEBUG) << "find 'Tensor." << PyUnicode_AsUTF8(func_name) << "' map to 'standard_method."
                << std::string(py::str(standard_func.ptr())) << "'";
  // standard method result
  std::vector<AObject *> args;
  if (callable->GetType() == AObject::kTypeBoundMethod) {
    // replace method call with function call. must unpack self to func
    args.push_back(callable->GetAttr(GraphBuilder::ID___self__));
  }
  std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->getVobj(); });
  AObject *res = InferFuncResult(standard_func, args, call_node->getOpcode(), call_node->GetGraph()->Config(), true);
  if (res) {
    call_node->setVobj(res);
    call_node->setSubGraph(nullptr);
    return false;
  }
  return SetCallResType<AObject::kTypeTensor>(call_node);
}

/**
 * find first free variable in names from function
 */
static py::object FindClosure(const py::object &o, const std::vector<std::string> &names, TracePtr *trace) {
  PyObject *func = o.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (!PyFunction_Check(func)) {
    return py::object();
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func));
  PyObject *closure = PyFunction_GET_CLOSURE(func);
  Py_ssize_t i = PyTuple_GET_SIZE(co->co_freevars) - 1;
  bool find = false;
  for (; i >= 0 && !find; --i) {
    std::string name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_freevars, i));
    find = std::find(names.begin(), names.end(), name) != names.end();
  }
  if (find) {
    Py_ssize_t idx = i + 1;
    TracePtr attr = CreateOpTrace(closure, LOAD_ATTR, 0, {*trace}, "", "__closure__");
    PyObject *cell = PyTuple_GET_ITEM(closure, idx);
    TracePtr cc = CreateOpTrace(cell, BINARY_SUBSCR, 0, {attr, std::make_shared<ConstTrace>(py::int_(idx).ptr(), -1)});
    PyObject *content = PyCell_GET(cell);
    *trace = CreateOpTrace(content, LOAD_ATTR, 0, {cc}, "", "cell_contents");
    return py::cast<py::object>(content);
  }
  *trace = nullptr;
  return py::object();
}

/**
 * get decorated function from 'after_grad'
 * \param after_grad _Grad.__call__.<locals>.after_grad
 * \return decorated object
 */
static py::object GetGradDecorated(const py::object &after_grad, TracePtr *trace) {
  MS_ASSERT(PyFunction_Check(after_grad.ptr()));
  py::object decorated = FindClosure(after_grad, {"fn", "fn_"}, trace);
  MS_EXCEPTION_IF_CHECK_FAIL(decorated.ptr() != nullptr, "can't find decorated function 'fn' or 'fn_' from " +
                                                           std::string(py::str(after_grad.ptr())));
  if (!PyFunction_Check(decorated.ptr())) {
    return decorated;
  }
  std::string decorated_name = PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(decorated.ptr())->func_qualname);
  if (decorated_name == "_Grad.__call__.<locals>.aux_fn") {
    decorated = FindClosure(decorated, {"fn"}, trace);
    MS_EXCEPTION_IF_CHECK_FAIL(decorated.ptr() != nullptr, "can't find decorated function 'fn' from " + decorated_name);
  }
  return decorated;
}

static py::object DeleteGradSensArgs(const py::object &args, const py::object &kwargs) {
  // sens param specified in kwargs
  if (kwargs.ptr() != nullptr && PyDict_DelItemString(kwargs.ptr(), "sens_param") != -1) {
    return args;
  }
  PyErr_Clear();
  // sens param is the last position argument
  PyObject *new_arg = PyTuple_GetSlice(args.ptr(), 0, PyTuple_GET_SIZE(args.ptr()) - 1);
  return py::reinterpret_steal<py::object>(new_arg);
}

static AObject *InferGradFuncResult(const py::object &func, const py::object &args, const py::object &kwargs,
                                    const GraphJitConfig &conf, OptCodePtr *guard) {
  auto jcr = getJitCompileResults(func.ptr());
  MS_EXCEPTION_IF_CHECK_FAIL(jcr, "must be");
  if (jcr->conf == nullptr) {
    jcr->conf = new GraphJitConfig(conf);
  }

  OptOptionPtr opt = OptOption::CreateOptionByPoint(jcr);
  auto guard_size = jcr->codehub->GetOptTarget(opt).size();

  AObject *res = InferFuncResult(func, args, kwargs, conf, false);
  MS_EXCEPTION_IF_CHECK_FAIL(jcr->codehub->GetOptTarget(opt).size() - guard_size <= 1,
                             "Check, multiple guard generated for one graph");

  OptCodePtr guard_code;
  if (jcr->codehub->GetOptTarget(opt).size() - guard_size == 1) {
    guard_code = jcr->codehub->GetOptTarget(opt).back();
    jcr->codehub->DelOptTarget(opt, guard_code);
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(res == nullptr, "check infer result");
  }

  if (res == nullptr) {
    *guard = nullptr;
    return nullptr;
  }

  *guard = guard_code;
  return res;
}

/**
 * Use the function decorated by 'after_grad' and arguments of 'after_grad' when called to infer result.
 * If the function has no unsupported operation, merge the guard of inferred graph to caller graph.
 * else clear the mask of mindspore flag, avoid to capture this function call
 */
void HandleGradFuncCall(CallNode *call_node, AObject *decorated, bool sens_param, const TracePtr &trace) {
  const int except_flag = AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc;
  ValueNode *grad_func_node = call_node->input(0);
  std::vector<py::object> stack_args;
  std::vector<TracePtr> traces;
  py::object func;
  py::object args;
  py::object kwargs;

  // prepare parameters
  bool param_ready = decorated->GetPyObject().ptr() != nullptr;
  if (decorated->GetType() == AObject::kTypeCell) {
    traces.push_back(trace);
    param_ready = trace != nullptr;
  }
  for (size_t i = 1; param_ready && i < call_node->getInputs().size(); ++i) {
    AObject *tmp = call_node->input(i)->getVobj();
    stack_args.push_back(tmp != nullptr ? tmp->GetPyObject() : py::object());
    traces.push_back(call_node->GetGraph()->TraceValueNode(call_node->input(i)));
    param_ready = traces.back() != nullptr;
  }
  if (param_ready) {
    auto pair = Utils::PackCallStackArgs(stack_args, call_node->getOpcode());
    args = pair.first;
    kwargs = pair.second;
    param_ready = pair.first.ptr() != nullptr;
  }
  if (!param_ready) {
    call_node->setInlineReason(InlineReason::kInlineInfer_Fail);
    grad_func_node->getVobj()->ClearMsFlag(except_flag);
    return;
  }
  if (sens_param) {
    args = DeleteGradSensArgs(args, kwargs);
  }

  // get callable
  if (decorated->GetType() != AObject::kTypeCell) {
    MS_EXCEPTION_IF_CHECK_FAIL(decorated->GetType() == AObject::kTypeFunction, "check grad input");
    func = decorated->GetPyObject();
  } else {
    // here get bound method.
    func = decorated->GetAttr(GraphBuilder::ID_construct)->GetPyObject();
  }

  OptCodePtr guard;
  AObject *res = InferGradFuncResult(func, args, kwargs, call_node->GetGraph()->Config(), &guard);
  if (res == nullptr) {
    call_node->setInlineReason(InlineReason::kInlineInfer_Fail);
    grad_func_node->getVobj()->ClearMsFlag(except_flag);
    return;
  }

  auto current_guard = call_node->GetGraph()->GetGuard();
  current_guard->GetGuard()->AddTraceFromGuard(traces, guard->GetGuard());

  call_node->setInlineReason(InlineReason::kInlineGraphSupportedByMS);
  call_node->setVobj(res);
}

static void HandleGradFunc(CallNode *call_node, const py::object &after_grad, TracePtr *trace) {
  py::object decorated_func = GetGradDecorated(after_grad, trace);
  TracePtr ptr;
  py::object grad = FindClosure(after_grad, {"grad_", "self"}, &ptr);
  MS_EXCEPTION_IF_CHECK_FAIL(grad.ptr() != nullptr,
                             "can't find 'grad_' object from " + std::string(py::str(after_grad.ptr())));
  bool sens_param = grad.attr("sens_param").ptr() == Py_True;
  MS_LOG(DEBUG) << "infer function 'after_grad', has sens_param " << (sens_param ? "True" : "False");

  call_node->setSubGraph(nullptr);
  HandleGradFuncCall(call_node, AObject::Convert(decorated_func), sens_param, *trace);
}

static bool check_GradFunc(const py::object &f) {
  if (!PyFunction_Check(f.ptr())) {
    return false;
  }
  std::string decorated_name = PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(f.ptr())->func_qualname);
  return decorated_name == "_Grad.__call__.<locals>.after_grad" ||
         decorated_name == "GradOperation.__call__.<locals>.after_grad";
}

static bool infer_GradFunc(CallNode *call_node) {
  AObject *vo = call_node->input(0)->getVobj();
  vo->SetMsFlag(AObject::kMsFlagGradFunc);
  py::object after_grad = vo->GetPyObject();
  TracePtr trace = call_node->GetGraph()->TraceValueNode(call_node->input(0));
  if (trace == nullptr) {
    vo->ClearMsFlag(AObject::kMsFlagGradFunc);
    call_node->setSubGraph(nullptr);
    return false;
  }
  HandleGradFunc(call_node, after_grad, &trace);
  return false;
}

static bool check_JitFunc(const py::object &o) {
  static const char except_file[] = "mindspore/common/api.py";
  static const size_t except_size = sizeof(except_file) - 1;
  PyObject *func = o.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (!PyFunction_Check(func)) {
    return false;
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func));
  const char *file = PyUnicode_AsUTF8(co->co_filename);
  const size_t size = strlen(file);
  return size > except_size && !strncmp(file + (size - except_size), except_file, except_size);
}

static bool infer_JitFunc(CallNode *call_node) {
  AObject *func_info = call_node->input(0)->getVobj();
  TracePtr trace = call_node->GetGraph()->TraceValueNode(call_node->input(0));
  TracePtr ptr = trace;
  func_info->SetMsFlag(AObject::kMsFlagJitFunc);
  py::object func = func_info->GetPyObject();
  py::object decorated = FindClosure(func, {"func"}, &ptr);
  MS_EXCEPTION_IF_CHECK_FAIL(decorated.ptr() != nullptr, "can't find 'func' object from " + std::string(py::str(func)));
  if (check_GradFunc(decorated) && trace != nullptr) {
    func_info->SetMsFlag(AObject::kMsFlagGradFunc);
    HandleGradFunc(call_node, decorated, &ptr);
  }
  if (call_node->getVobj() != nullptr) {
    return false;
  }
  return SetCallResType<AObject::kTypeTensor>(call_node);
}

static bool check_StandardFunction(const py::object &func) {
  if (func.ptr() == nullptr || !PyFunction_Check(func.ptr())) {
    return false;
  }
  static const std::vector<std::string> except_module = {"mindspore.ops.", "mindspore._extends.parse.standard_method"};
  PyObject *func_module = reinterpret_cast<PyFunctionObject *>(func.ptr())->func_module;
  if (func_module == nullptr || !PyUnicode_Check(func_module)) {
    return false;
  }
  std::string module_name = PyUnicode_AsUTF8(func_module);
  auto iter = std::find_if(except_module.begin(), except_module.end(), [&module_name](const std::string &tar) {
    return module_name.size() > tar.size() && module_name.substr(0, tar.size()) == tar;
  });
  return iter != except_module.end();
}

static bool infer_StandardFunction(CallNode *call_node) {
  AObject *func_info = call_node->input(0)->getVobj();
  call_node->setSubGraph(nullptr);
  func_info->SetMsFlag(AObject::kMsFlagStandardFunc);
  py::object func = func_info->GetPyObject();
  MS_LOG(DEBUG) << "infer " << std::string(py::str(PyFunction_GET_CODE(func.ptr())));
  int op = call_node->getOpcode();
  const auto &conf = call_node->GetGraph()->Config();
  std::vector<AObject *> args;
  std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->getVobj(); });
  AObject *res = InferFuncResult(func, args, op, conf, true);
  call_node->setVobj(res);
  return false;
}

static bool check_MsCell(const py::object &cell) {
  PyTypeObject *tp = PyType_Check(cell.ptr()) ? reinterpret_cast<PyTypeObject *>(cell.ptr()) : Py_TYPE(cell.ptr());
  py::object tp_handle = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(tp));
  if (!IsCellTypeOrSubType(tp)) {
    return false;
  }
  py::object mod = tp_handle.attr("__module__");
  const char *current = PyUnicode_AsUTF8(mod.ptr());
  const char *except = "mindspore.";
  return !strncmp(current, except, strlen(except));
}

static bool infer_MsCell(CallNode *node) {
  node->setSubGraph(nullptr);
  // maybe need infer nn.cell shape and dtype
  return SetCallResType<AObject::kTypeTensor>(node);
}

// special function list
// special function that mindspore support and not inline,
// the return values or type can be infer
static const std::unordered_map<std::string, SpecialAction> kFuncWhiteListMap = {
  // fuzzy match
  {kMindsporeNamePrimitive, {check_primitive, infer_primitive}},
  {kMindsporeNameMetaFuncGraph, {check_MetaFunc_, infer_MetaFunc_}},
  {kMindsporeNameGradFunc, {check_GradFunc, infer_GradFunc}},
  {kMindsporeNameTensorMethod, {check_TensorMethod, infer_TensorMethod}},
  {kMindsporeNameStandardFunction, {check_StandardFunction, infer_StandardFunction}},
  {kMindsporeNameMsCell, {check_MsCell, infer_MsCell}},
  // name match
  {kMindsporeNameJitFunc, {check_JitFunc, infer_JitFunc}},
  {kMindsporeNameGetCachePrim, {check__get_cache_prim, infer__get_cache_prim}},
  {kBuiltinNameLen, {check_builtin_cfunc, infer_builtin_len}},
  {kBuiltinNameAbs, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameAbs)}},
  // NOTE: call __bool__ hook for each item
  {kBuiltinNameAll, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameAll)}},
  // NOTE: call __bool__ hook for each item
  {kBuiltinNameAny, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameAny)}},
  {kBuiltinNameHash, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameHash)}},
  {kBuiltinNameIsinstance, {check_builtin_cfunc, InferBuiltinInstanceSubclassCheck<PyObject_IsInstance>}},
  {kBuiltinNameIssubclass, {check_builtin_cfunc, InferBuiltinInstanceSubclassCheck<PyObject_IsSubclass>}},
  {kBuiltinNameId, {check_builtin_cfunc, SetCallResType<AObject::kTypeInt>}},
  {kBuiltinNameOrd, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameOrd)}},
  {kBuiltinNameGlobals, {check_builtin_cfunc, InferBuiltinGlobals}},
  {kBuiltinNameCallable, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameCallable)}},
  {kBuiltinNameGetattr, {check_builtin_cfunc, infer_builtin_getattr}},
  {kBuiltinNameHasattr, {check_builtin_cfunc, SetCallResType<AObject::kTypeBool>}},
  // object convert map
  {kMindsporeNameConvertMap, {check_ConvertMap, infer_ConvertMap}},
};

static const std::vector<std::pair<CheckFunc, std::string>> kFuncWhiteListFuzzyMatcher = {
  {check_MetaFunc_, kMindsporeNameMetaFuncGraph},
  {check_GradFunc, kMindsporeNameGradFunc},
  // guard these call by short traces
  // {check_TensorMethod, kMindsporeNameTensorMethod},
  // {check_StandardFunction, kMindsporeNameStandardFunction},
  // {check_MsCell, kMindsporeNameMsCell},
  {check_ConvertMap, kMindsporeNameConvertMap},
};

static const char *GetFuncName(const py::object &f) {
  PyObject *func = f.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (PyCFunction_Check(func)) {
    return reinterpret_cast<PyCFunctionObject *>(func)->m_ml->ml_name;
  }
  PyCodeObject *co = nullptr;
  if (PyFunction_Check(func)) {
    co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func));
  }
  if (co) {
    return PyUnicode_AsUTF8(co->co_name);
  }
  PyTypeObject *tp = PyType_Check(func) ? reinterpret_cast<PyTypeObject *>(func) : Py_TYPE(func);
  const char *res = strrchr(tp->tp_name, '.');
  return res ? res + 1 : tp->tp_name;
}

bool IsFuncInWhiteList(const py::object &f, std::string *special_func_key, bool bInferPrimitive) {
  if (f.ptr() == nullptr) {
    return false;
  }
  *special_func_key = GetFuncName(f);
  auto iter = kFuncWhiteListMap.find(*special_func_key);
  if (iter != kFuncWhiteListMap.end()) {
    return iter->second.check(f);
  }
  if (bInferPrimitive && check_primitive(f)) {
    *special_func_key = kMindsporeNamePrimitive;
    return true;
  }
  auto tar = std::find_if(kFuncWhiteListFuzzyMatcher.begin(), kFuncWhiteListFuzzyMatcher.end(),
                          [&f](const std::pair<CheckFunc, std::string> &i) { return i.first(f); });
  if (tar != kFuncWhiteListFuzzyMatcher.end()) {
    *special_func_key = tar->second;
    return true;
  }
  return false;
}

bool HandleFuncInWhiteList(const std::string &key, CallNode *n) {
  MS_LOG(DEBUG) << "specialize for " << key;
  return kFuncWhiteListMap.find(key)->second.infer(n);
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
