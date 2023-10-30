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
#include "pipeline/jit/pi_jit/common.h"
#include <algorithm>
#include <iomanip>
#include <list>
#include <map>
#include <string>
#include <vector>
#include "pipeline/jit/pi_jit/external.h"
#include "pipeline/jit/pi_jit/graph_capture/code_gen.h"
#include "pipeline/jit/pi_jit/graph_capture/graph_build.h"
#include "pipeline/jit/pi_jit/graph_capture/graph_analyzer.h"
#include "pipeline/jit/pi_jit/graph_compiler/compiler.h"
#include "pipeline/jit/pi_jit/graph_compiler/cg/byte_code_generator.h"
#include "pipeline/jit/pi_jit/graph_compiler/inliner/func_inliner.h"
#include "pipeline/jit/pi_jit/graph_compiler/parser/byte_code_parser.h"
#include "pipeline/jit/pi_jit/graph_compiler/utils.h"
#include "pipeline/jit/pi_jit/utils/utils.h"
#include "pipeline/jit/ps/pipeline.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/jit/pi_jit/ms_adapter/infer.h"

namespace mindspore {
namespace jit {
namespace graph {
static Py_tss_t *tss = NULL;

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard);
void AddGuardForParam(const PyFrameObject *f, OptGuardPtr guard);
static void AddGradFlagForParam(bool grad_flag, OptGuardPtr guard);
static void CollectTraceBack(JitCompileResults *c, const py::object &new_func, bool is_graph_mode);
static void BackupLocals(PyFrameObject *f, std::vector<PyObject *> *locals);
static void RestoreLocals(PyFrameObject *f, std::vector<PyObject *> *locals);

// jit compiler initialize
static void ensureInitialize() {
  static bool init = false;
  if (init) {
    return;
  }
  init = true;
  if (tss == NULL) {
    tss = PyThread_tss_alloc();
    PyThread_tss_create(tss);
  }
}

void Tracebackes::PushInlineInfo(InlineInfo info) {
  const auto &it = inline_infos_.find(info.root_name_);
  if (it != inline_infos_.cend()) {
    it->second.push_back(info);
  } else {
    std::list<InlineInfo> inlines;
    inlines.push_back(info);
    inline_infos_.emplace(info.root_name_, inlines);
  }
}

static void PrintLabel(std::stringstream &os, const std::string &str, int distance = 30) {
  os << std::left << std::setw(distance) << str << ": ";
}

std::string Tracebackes::Dump(bool is_all) const {
  std::stringstream os;
  std::string cur_name = tbs_.empty() ? "" : tbs_.back().func_name_;
  if (is_all) {
    os << "*** Dump Traceback on [" << raw_func_info_name_ << "] ***\n";
  } else {
    os << "*** Dump ByteCode After Traceback on [" << cur_name << "] ***\n";
  }
  if (tbs_.empty()) {
    return os.str();
  }
  std::list<Tracebacke> candidates;
  if (is_all) {
    candidates = tbs_;
  } else {
    // last one traceback
    candidates.emplace_back(tbs_.back());
  }
  // dump traceback list head
  int name_length = FindMaxNameLength(candidates);
  os << std::left << std::setw(name_length) << "func_name:"
     << "  -->  " << std::left << std::setw(name_length) << "changed_func:" << std::left << std::setw(10)
     << "run_mode:" << std::left << std::setw(30) << "stop_trace:" << std::left << std::setw(10)
     << "code_size:" << std::endl;
  os << "--------------------------------------------------------------------------------------\n";
  // dump traceback list content
  for (const auto &tb : candidates) {
    os << std::left << std::setw(name_length) << tb.func_name_ << "  -->  ";
    os << std::left << std::setw(name_length) << tb.changed_func_;
    if (tb.is_graph_mode_) {
      os << std::left << std::setw(10) << "[GRAPH]";
    } else {
      os << std::left << std::setw(10) << "PYNATIVE";
    }
    // dump stop trace reason
    auto it_trace = stop_trace_res_.find(tb.func_name_);
    if (it_trace != stop_trace_res_.cend()) {
      os << std::left << std::setw(30) << GetStopTraceReasonDesc(it_trace->second);
    } else {
      os << std::left << std::setw(30) << "unknown";
    }
    os << std::left << std::setw(10) << tb.code_size_ << " =====>\n";
    // dump inline info
    DumpInlineInfo(os, tb.func_name_);
  }
  os << "\n\n";
  if (is_all) {
    os << DumpSummary();
  }
  return os.str();
}

void Tracebackes::DumpInlineInfo(std::stringstream &os, const std::string &func_name) const {
  const auto &it = inline_infos_.find(func_name);
  if (it == inline_infos_.cend()) {
    return;
  }
  for (const auto &info : it->second) {
    std::string space((info.depth + 1) * 2, ' ');
    os << space << "| inline_info:" << GetInlineReasonDesc(info.res) << " line:" << info.line;
    if (!info.inline_name_.empty()) {
      os << " func_name:" << info.inline_name_;
    }
    if (info.res == InlineReason::kInline || info.res == InlineReason::kInlinePartial) {
      os << " code_size:" << info.code_size_;
    }
    os << "\n";
  }
}

std::string Tracebackes::DumpSummary() const {
  std::stringstream os;
  if (tbs_.empty()) {
    return os.str();
  }
  os << "*** Dump Summary on [" << raw_func_info_name_ << "] ***\n";
  PrintLabel(os, "traceback_num");
  os << tbs_.size() << "\n";

  std::array<int, kStopTrace_Reason_Count> stop_trace_reason_array{0};
  std::array<int, kInline_Reason_Count> inline_reason_array{0};
  int graph_mode_num = 0;
  int raw_code_size = raw_code_size_;
  int pynative_code_size = 0;
  int graph_mode_code_size = 0;
  for (const auto &tb : tbs_) {
    if (tb.is_graph_mode_) {
      graph_mode_num++;
      graph_mode_code_size += tb.code_size_;
    } else {
      pynative_code_size += tb.code_size_;
    }
    auto it_trace = stop_trace_res_.find(tb.func_name_);
    if (it_trace != stop_trace_res_.cend()) {
      // count stop trace reason
      stop_trace_reason_array[it_trace->second]++;
    }
    const auto &it_inline = inline_infos_.find(tb.func_name_);
    if (it_inline == inline_infos_.cend()) {
      continue;
    }
    for (const auto &info : it_inline->second) {
      // count inline reason
      inline_reason_array[info.res]++;
      if (info.res == InlineReason::kInline || info.res == InlineReason::kInlinePartial) {
        raw_code_size += info.code_size_;
      }
    }
  }
  PrintLabel(os, "graph_mode_num");
  os << graph_mode_num << "\n";
  PrintLabel(os, "raw_code_size(+ inline)");
  os << raw_code_size << "\n";
  PrintLabel(os, "pynative_code_size");
  os << pynative_code_size << "\n";
  PrintLabel(os, "graph_mode_code_size");
  os << graph_mode_code_size << "\n";
  os << "----------stop_trace_reason----------\n";
  for (size_t i = 0; i < stop_trace_reason_array.size(); ++i) {
    PrintLabel(os, GetStopTraceReasonDesc(static_cast<StopTraceReason>(i)));
    os << stop_trace_reason_array[i] << "\n";
  }
  os << "----------inline_reason----------\n";
  for (size_t i = 0; i < inline_reason_array.size(); ++i) {
    PrintLabel(os, GetInlineReasonDesc(static_cast<InlineReason>(i)));
    os << inline_reason_array[i] << "\n";
  }
  os << "\n\n";
  return os.str();
}

int Tracebackes::FindMaxNameLength(const std::list<Tracebacke> &tbs) const {
  int max_length = 15;
  for (const auto &tb : tbs) {
    int len1 = tb.func_name_.length();
    int len2 = tb.changed_func_.length();
    max_length = std::max(max_length, std::max(len1, len2)) + 2;
  }
  max_length = std::min(max_length, 35);
  return max_length;
}

static void freeJitCompileResults(void *jitCompileResults) {
  // maybe nullptr if other module use _PyEval_RequestCodeExtraIndex
  if (jitCompileResults == nullptr) {
    return;
  }
  // called after code object freed
  JitCompileResults *c = reinterpret_cast<JitCompileResults *>(jitCompileResults);
  Py_CLEAR(c->compiled.callable);
  c->codehub.reset();
  MS_LOG(DEBUG) << __FUNCTION__ << " " << c;
  if (c->sub_routine) {
    delete c;
    return;
  }
  if (c->tbs != nullptr) {
    delete c->tbs;
    c->tbs = nullptr;
  }
  if (c->conf != nullptr) {
    delete c->conf;
    c->conf = nullptr;
  }
  delete c;
}

static JitCompileResults *allocJitCompileResults() {
  JitCompileResults *c = new JitCompileResults();
  c->codehub = std::make_shared<OptCodeHub>();
  c->sub_routine = false;
  c->ms_mode_ = false;
  return c;
}

JitCompileResults *getJitCompileResults(PyObject *code, bool alloc) {
  if (PyMethod_Check(code)) {
    code = PyMethod_GET_FUNCTION(code);
  }
  if (PyFunction_Check(code)) {
    code = PyFunction_GET_CODE(code);
  }
  if (!PyCode_Check(code)) {
    return NULL;
  }
  ensureInitialize();
  Py_ssize_t index = (Py_ssize_t)PyThread_tss_get(tss);
  if (index == 0) {
    index = _PyEval_RequestCodeExtraIndex(freeJitCompileResults);
    if (index == -1) {
      return NULL;
    }
    // ensure index is not 0
    PyThread_tss_set(tss, reinterpret_cast<void *>(index + 1));
  } else {
    index = index - 1;
  }

  JitCompileResults *c = NULL;
  if (!_PyCode_GetExtra(code, index, reinterpret_cast<void **>(&c))) {
    if (c != NULL) {
      return c;
    }
    if (!alloc) {
      return NULL;
    }
    c = allocJitCompileResults();
    if (c == NULL) {
      return NULL;
    }
    if (!_PyCode_SetExtra(code, index, c)) {
      MS_LOG(DEBUG) << "allocJitCompileResults " << c << " for " << std::string(py::str(code));
      return c;
    }
    freeJitCompileResults(c);
  }
  PyErr_Clear();
  return NULL;
}

static void BackupItem(PyObject *src, PyObject **dst) {
  if (src != nullptr) {
    Py_INCREF(src);
    *dst = src;
  } else {
    *dst = nullptr;
  }
}

static void BackupLocals(PyFrameObject *f, std::vector<PyObject *> *locals) {
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  argc += (f->f_code->co_flags & CO_VARARGS) ? 1 : 0;
  argc += (f->f_code->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  int total =
    f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars) + PyTuple_GET_SIZE(f->f_code->co_freevars);
  locals->resize(total, nullptr);

  // deal arguments
  for (int i = 0; i < argc; i++) {
    BackupItem(f->f_localsplus[i], &((*locals)[i]));
  }

  // deal cell
  int cell_size = PyTuple_GET_SIZE(f->f_code->co_cellvars);
  for (int i = 0; i < cell_size; i++) {
    BackupItem(f->f_localsplus[f->f_code->co_nlocals + i], &((*locals)[f->f_code->co_nlocals + i]));
  }

  // deal free
  for (int i = 0; i < PyTuple_GET_SIZE(f->f_code->co_freevars); ++i) {
    BackupItem(f->f_localsplus[f->f_code->co_nlocals + cell_size + i],
               &((*locals)[f->f_code->co_nlocals + cell_size + i]));
  }
}

static void RestoreItem(PyObject *src, PyObject **dst) {
  PyObject *cur = *dst;
  *dst = src;
  if (cur != nullptr) {
    Py_DECREF(cur);
  }
}

static void RestoreLocals(PyFrameObject *f, std::vector<PyObject *> *locals) {
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  argc += (f->f_code->co_flags & CO_VARARGS) ? 1 : 0;
  argc += (f->f_code->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  int total =
    f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars) + PyTuple_GET_SIZE(f->f_code->co_freevars);
  locals->resize(total, nullptr);

  // deal arguments
  for (int i = 0; i < argc; i++) {
    RestoreItem((*locals)[i], &(f->f_localsplus[i]));
  }

  // deal cell
  int cell_size = PyTuple_GET_SIZE(f->f_code->co_cellvars);
  for (int i = 0; i < cell_size; i++) {
    RestoreItem((*locals)[f->f_code->co_nlocals + i], &(f->f_localsplus[f->f_code->co_nlocals + i]));
  }

  // deal free
  for (int i = 0; i < PyTuple_GET_SIZE(f->f_code->co_freevars); ++i) {
    RestoreItem((*locals)[f->f_code->co_nlocals + cell_size + i],
                &(f->f_localsplus[f->f_code->co_nlocals + cell_size + i]));
  }
}

static PyFrameObject *RebuildFrame(PyThreadState *tstate, PyCodeObject *co, const PyFrameObject *f, PyObject *closure,
                                   PyObject *globals) {
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  MS_ASSERT(co != nullptr && argc == co->co_argcount + co->co_kwonlyargcount);
  MS_ASSERT((f->f_code->co_flags & CO_VARARGS) == (co->co_flags & CO_VARARGS));
  MS_ASSERT((f->f_code->co_flags & CO_VARKEYWORDS) == (co->co_flags & CO_VARKEYWORDS));
  argc += (f->f_code->co_flags & CO_VARARGS) ? 1 : 0;
  argc += (f->f_code->co_flags & CO_VARKEYWORDS) ? 1 : 0;

  PyFrameObject *frame = PyFrame_New(tstate, co, globals, NULL);
  // copy arguments
  for (int i = 0; i < argc; i++) {
    Py_XINCREF(f->f_localsplus[i]);
    frame->f_localsplus[i] = f->f_localsplus[i];
  }
  // restore arguments from cell
  std::vector<PyObject *> cells_content(f->f_code->co_nlocals, nullptr);
  for (int i = 0; f->f_code->co_cell2arg != NULL && i < PyTuple_GET_SIZE(f->f_code->co_cellvars); ++i) {
    Py_ssize_t argi = f->f_code->co_cell2arg[i];
    if (argi != CO_CELL_NOT_AN_ARG) {
      PyObject *cell = f->f_localsplus[f->f_code->co_nlocals + i];
      cells_content[argi] = PyCell_GET(cell);
    }
  }
  // new cell
  for (int i = 0; i < PyTuple_GET_SIZE(co->co_cellvars); ++i) {
    PyObject *cell;
    if (co->co_cell2arg != NULL && co->co_cell2arg[i] != CO_CELL_NOT_AN_ARG) {
      Py_ssize_t argi = co->co_cell2arg[i];
      MS_EXCEPTION_IF_CHECK_FAIL(cells_content[argi], "Unbound local exception");
      cell = PyCell_New(cells_content[argi]);
    } else {
      cell = PyCell_New(NULL);
    }
    frame->f_localsplus[co->co_nlocals + i] = cell;
  }
  // assert expression
  if (closure != nullptr) {
    MS_ASSERT(PyTuple_Check(closure) && PyTuple_GET_SIZE(co->co_freevars) == PyTuple_GET_SIZE(closure));
  } else {
    MS_ASSERT(PyTuple_GET_SIZE(co->co_freevars) == PyTuple_GET_SIZE(f->f_code->co_freevars));
  }
  // copy closure
  for (int i = 0; i < PyTuple_GET_SIZE(co->co_freevars); ++i) {
    int a = f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars) + i;
    int b = co->co_nlocals + PyTuple_GET_SIZE(co->co_cellvars) + i;
    auto o = closure ? PyTuple_GET_ITEM(closure, i) : f->f_localsplus[a];
    Py_XINCREF(o);
    frame->f_localsplus[b] = o;
  }
  return frame;
}

static PyObject *GetClosure(PyFrameObject *f) {
  int nfrees = PyTuple_GET_SIZE(f->f_code->co_freevars);
  if (!nfrees) {
    return nullptr;
  }
  PyObject *closure = PyTuple_New(nfrees);
  int idx = f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars);
  for (int i = 0; i < nfrees; ++i) {
    PyObject *o = f->f_localsplus[idx + i];
    Py_INCREF(o);
    PyTuple_SET_ITEM(closure, i, o);
  }
  return closure;
}

static PyFrameObject *PrepareCallCompiledCallable(PyThreadState *tstate, const PyFrameObject *f,
                                                  const JitCompileResults *c) {
  PyCodeObject *co = nullptr;
  PyObject *closure = nullptr;
  PyObject *globals = f->f_globals;
  if (PyCode_Check(c->compiled.callable)) {
    co = reinterpret_cast<PyCodeObject *>(c->compiled.callable);
  } else if (PyFunction_Check(c->compiled.callable)) {
    globals = PyFunction_GET_GLOBALS(c->compiled.callable);
    co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(c->compiled.callable));
    closure = PyFunction_GET_CLOSURE(c->compiled.callable);
  } else {
    MS_LOG(EXCEPTION) << "unknown compile results error";
  }
  return RebuildFrame(tstate, co, f, closure, globals);
}

static void GuardForFrame(PyFrameObject *frame, const OptCodePtr &oc, const GraphJitConfig &conf) {
  const char *code_name = PyUnicode_AsUTF8(frame->f_code->co_name);
  AddConfigToGuard(conf, oc->GetGuard());
  AddGuardForParam(frame, oc->GetGuard());
  AddGradFlagForParam(pynative::PyNativeExecutor::GetInstance()->grad_flag(), oc->GetGuard());
  if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
    GRAPH_JIT_LOG_F("Guard on %s by %s!\n", code_name, oc->GetGuard()->GetDescript().c_str());
    return;
  }
  MS_LOG(DEBUG) << "Guard on " << code_name << " by " << oc->GetGuard()->GetDescript() << "!" << std::endl;
}

static void HandleBreakGraph(JitCompileResults *jcr, const OptCodePtr &oc) {
  const auto &conf = *jcr->conf;
  PyObject *new_func = jcr->compiled.callable;
  MS_EXCEPTION_IF_CHECK_FAIL(Py_REFCNT(new_func) == 1 && jcr->stat == JitCompileResults::GRAPH_CALLABLE,
                             "only call this just after graph captured");

  CollectTraceBack(jcr, py::cast<py::object>(new_func), false);
  if (conf.GetBoolConfig(GraphJitConfig::kEnableGuard)) {
    oc->SetPythonCallable(new_func);
    GuardForFrame(jcr->f, oc, conf);
  }
}

static void ValidateCompiledResults(JitCompileResults *c) {
  if (c->stat != JitCompileResults::GRAPH_CALLABLE) {
    return;
  }
  PyObject *func = c->compiled.callable;
  bool valid_res;
  if (c->compiled.cFunc) {
    valid_res = true;
  } else {
    valid_res = PyCallable_Check(func) || (func && PyCode_Check(func));
  }
  MS_EXCEPTION_IF_CHECK_FAIL(valid_res, "check compiled result");
}

static py::object MakeFunctionFromCodeGen(CodeGenerator *cg, PyObject *builtins, PyObject *default_globals) {
  PyObject *new_code = reinterpret_cast<PyObject *>(cg->MakeCodeObj());
  PyDict_Merge(default_globals, cg->GetGlobals().ptr(), 0);
  PyObject *new_func = PyFunction_New(new_code, default_globals);
  Py_DECREF(new_code);
  Py_XSETREF(PyFunction_GET_CLOSURE(new_func), cg->GetClosure().inc_ref().ptr());
  return py::reinterpret_steal<py::object>(new_func);
}

// preprocess before compile, split bytecode to sub-function
static void GraphCapture(JitCompileResults *jcr) {
  size_t guard_ocs_size = jcr->codehub->GetOptTarget(OptOption::CreateOptionByPoint(jcr)).size();
  GraphJitConfig &conf = *jcr->conf;
  OptCodePtr oc;

  GraphBuilder g(jcr->f);
  (void)g.BuildGraph();
  oc = g.GetGraph()->GetGuard();

  GraphAnalyzer analyzer(g.GetGraph());
  analyzer.Analyze();

  // dump DFG
  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    g.DumpDFG();
  }

  CodeGenerator cg(g.GetGraph(), analyzer.GetCaptureInfo());
  bool is_graph_break = cg.TryToBreakGraphIfParameterUnsupported();
  cg.CutoffBytecodesIfGraphBreak();

  py::object new_func = MakeFunctionFromCodeGen(&cg, jcr->f->f_builtins, jcr->f->f_globals);
  Py_XSETREF(jcr->compiled.callable, new_func.inc_ref().ptr());

  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    Utils::DisFuncObject(jcr->compiled.callable);
    GRAPH_JIT_LOG_F("\n\n");
  }
  jcr->stat = JitCompileResults::GRAPH_CALLABLE;

  // collect stop trace reason to traceback
  jcr->tbs->PushStopTraceRes(g.GetGraph()->GetCodeName(), g.GetGraph()->GetStopTraceReason());
  if (conf.GetBoolConfig(GraphJitConfig::kEnableGuard)) {
    OptCodeSet ocs = jcr->codehub->GetOptTarget(OptOption::CreateOptionByPoint(jcr));
    MS_EXCEPTION_IF_CHECK_FAIL(guard_ocs_size < ocs.size() && ocs[guard_ocs_size] == oc, "multiply guard oc generated");
  }
  AObject::aobject_mem_pool_.Clear(__FILE__, __LINE__);

  if (is_graph_break || (cg.IsCodeChanged() && g.GetGraph()->GetStopTraceAt())) {
    // break graph interpret
  } else if (conf.GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
    // config interpret
  } else {
    jcr->stat = JitCompileResults::GRAPH_CAPTURED;
    return;
  }
}

static void CollectTraceBack(JitCompileResults *c, const py::object &new_func, bool is_graph_mode) {
  auto *code = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(new_func.ptr()));
  std::string name = Utils::GetPyName(c->f->f_code->co_name);
  std::string changed_name = Utils::GetPyName(code->co_name);
  int code_size = (PyBytes_GET_SIZE(code->co_code)) / sizeof(_Py_CODEUNIT);
  c->tbs->PushTbs({name, changed_name, code_size, is_graph_mode});
}

std::string GetFuncGraphPhase(const PyFrameObject &frame, const OptCodePtr &oc) {
  std::string phase = py::cast<std::string>(frame.f_code->co_filename) + "_" +
                      std::to_string(frame.f_code->co_firstlineno) + "_" + py::cast<std::string>(frame.f_code->co_name);
  if (oc != nullptr) {
    phase += oc->GetGuard()->GetDescript();
  } else {
    for (int i = 0; i < frame.f_code->co_argcount; i++) {
      PyObject *obj = PyTuple_GET_ITEM(frame.f_code->co_varnames, i);
      py::object para = py::cast<py::object>(PyDict_GetItem(frame.f_locals, obj));
      auto node = GraphUtils::ConvertPythonObjectToAnfNode(para);
      phase += "_" + node->abstract()->ToString();
    }
  }
  phase += ".pi_jit";
  return phase;
}

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard) {
  std::map<std::string, bool> cfg;
  cfg[kSpecializeScalar] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeScalar);
  cfg[kSpecializeContainer] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeContainer);
  cfg[kSpecializeTensor] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeTensor);
  guard->UpdateConfig(cfg);
}

void AddGuardForParam(const PyFrameObject *f, OptGuardPtr guard) {
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  PyTupleObject *vargs = NULL;
  PyDictObject *kwargs = NULL;
  if (f->f_code->co_flags & CO_VARARGS) {
    vargs = _PyTuple_CAST(f->f_localsplus[argc]);
  }
  if (f->f_code->co_flags & CO_VARKEYWORDS) {
    kwargs = reinterpret_cast<PyDictObject *>(f->f_localsplus[argc + (vargs ? 1 : 0)]);
  }
  for (int i = 0; i < argc; ++i) {
    RootTracePtr ptr = std::make_shared<RootTrace>(f->f_localsplus[i], mindspore::jit::graph::TraceType::Param, i);
    guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
  }
  if (vargs != NULL) {
    RootTracePtr ptr =
      std::make_shared<RootTrace>(f->f_localsplus[argc], mindspore::jit::graph::TraceType::Param, argc);
    guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
  }
  if (kwargs != NULL) {
    RootTracePtr ptr = std::make_shared<RootTrace>(f->f_localsplus[argc + (vargs ? 1 : 0)],
                                                   mindspore::jit::graph::TraceType::Param, argc + (vargs ? 1 : 0));
    guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
  }
  for (int i = 0; f->f_code->co_cell2arg && i < PyTuple_GET_SIZE(f->f_code->co_cellvars); ++i) {
    Py_ssize_t arg = f->f_code->co_cell2arg[i];
    if (arg != CO_CELL_NOT_AN_ARG) {
      RootTracePtr ptr = std::make_shared<RootTrace>(
        f->f_localsplus[f->f_code->co_nlocals + i], mindspore::jit::graph::TraceType::Param, f->f_code->co_nlocals + i);
      guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
    }
  }
}

static void AddGradFlagForParam(bool grad_flag, OptGuardPtr guard) {
  CustomizedTracePtr ptr = std::make_shared<CustomizedTrace>(
    grad_flag ? Py_True : Py_False,
    [](PTraceContext context) -> PyObject * {
      auto pynative_exec = pynative::PyNativeExecutor::GetInstance();
      PyObject *ret = pynative_exec->grad_flag() ? Py_True : Py_False;
      Py_INCREF(ret);
      return ret;
    },
    [grad_flag]() -> std::string {
      return std::string("{PyNativeExecutor::GetInstance()->grad_flag == ") + std::to_string(grad_flag) +
             std::string("}(type:") + std::to_string(TraceType::Customized) + std::string(")");
    });
  guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GEqual, true);
}

static void CallGraphCompiler(JitCompileResults *jcr, PyFunctionObject *func, PyFrameObject *frame,
                              const std::string &phase) {
  MS_LOG(DEBUG) << "Phase is " << phase << "!";
  CallableGraph callable = mindspore::jit::graph::Compiler::Compile(*func, *frame, phase);
  jcr->compiled.cFunc = callable;
  jcr->stat = JitCompileResults::GRAPH_CALLABLE;
  CollectTraceBack(jcr, py::cast<py::object>(reinterpret_cast<PyObject *>(func)), true);
}

static void GuardForGraph(JitCompileResults *c, const std::string &graph_phase, const OptCodePtr &oc) {
  auto graph_executor = mindspore::pipeline::GraphExecutorPy::GetInstance();
  ReleaseFunc rFunc = nullptr;
  if (c->conf->GetBoolConfig(GraphJitConfig::kAutoCleanCache)) {
    rFunc = [graph_executor, graph_phase]() {
      if (graph_executor->HasCompiled(graph_phase)) {
        py::str p(graph_phase);
        py::set s;
        s.add(graph_phase);
        py::object o = py::none();
        graph_executor->DelNetRes(o, s);
        MS_LOG(DEBUG) << "To release " << graph_phase;
      }
    };
  }
  if (c->compiled.cFunc != nullptr || c->compiled.callable != nullptr) {
    if (c->compiled.cFunc != nullptr) {
      oc->SetNativeFunc(c->compiled.cFunc, rFunc);
      oc->SetPhase(graph_phase);
    }
    if (c->compiled.callable != nullptr) {
      oc->SetPythonCallable(c->compiled.callable);
    }
  } else {
    c->codehub->DelOptTarget(oc->GetOption(), oc);
  }
}

static void GraphCompile(JitCompileResults *jcr, OptCodePtr oc, py::object func_handle, py::object frame_handle) {
  jcr->stat = JitCompileResults::GRAPH_BUILDING;
  const auto &conf = *jcr->conf;
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(func_handle.ptr());
  PyFrameObject *frame = reinterpret_cast<PyFrameObject *>(frame_handle.ptr());
  PyFrame_FastToLocals(frame);

  // restore function object from frame
  if (func == nullptr) {
    PyObject *new_func = PyFunction_New(reinterpret_cast<PyObject *>(frame->f_code), frame->f_globals);
    Py_XSETREF(PyFunction_GET_CLOSURE(new_func), GetClosure(frame));
    func_handle = py::reinterpret_steal<py::object>(new_func);
    func = reinterpret_cast<PyFunctionObject *>(new_func);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(PyFunction_Check(func_handle.ptr()), "must be function");
  // restore some information from original function if it's top function
  if (jcr->func) {
    REPLACE_PY_MEMBER(func->func_module, PyFunction_GET_MODULE(jcr->func));
    REPLACE_PY_MEMBER(func->func_defaults, PyFunction_GET_DEFAULTS(jcr->func));
    REPLACE_PY_MEMBER(func->func_kwdefaults, PyFunction_GET_KW_DEFAULTS(jcr->func));
    REPLACE_PY_MEMBER(func->func_annotations, PyFunction_GET_ANNOTATIONS(jcr->func));
  }

  if (conf.GetBoolConfig(GraphJitConfig::kEnableGuard)) {
    if (oc == nullptr) {
      OptOptionPtr opt = OptOption::CreateOptionByPoint(jcr);
      oc = jcr->codehub->AddOptTarget(opt);
      MS_EXCEPTION_IF_CHECK_FAIL(oc != nullptr, "Fail to add optimized code!");
    }
    GuardForFrame(frame, oc, conf);
  }
  std::string phase = GetFuncGraphPhase(*frame, oc);
  CallGraphCompiler(jcr, func, frame, phase);

  if (jcr->conf->GetBoolConfig(GraphJitConfig::kEnableGuard)) {
    GuardForGraph(jcr, phase, oc);
  }
}

extern bool UnsupportedCodeTypeCheck(PyCodeObject *co);
static bool JitCompile(PyThreadState *tstate, JitCompileResults *c) {
  if (UnsupportedCodeTypeCheck(c->f->f_code)) {
    return false;
  }

  const char *origin_code_name = PyUnicode_AsUTF8(c->f->f_code->co_name);
  MS_LOG(DEBUG) << "---start compile " << origin_code_name << "---";

  OptCodePtr oc;
  py::object func = py::reinterpret_borrow<py::object>(c->func);
  py::object frame = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(c->f));
  if (c->stat == JitCompileResults::GRAPH_CANDIDATE) {
    size_t guard_ocs_size = c->codehub->GetOptTarget(OptOption::CreateOptionByPoint(c)).size();
    std::vector<PyObject *> locals;
    BackupLocals(c->f, &locals);
    GraphCapture(c);
    RestoreLocals(c->f, &locals);
    if (c->conf->GetBoolConfig(GraphJitConfig::kEnableGuard)) {
      OptCodeSet ocs = c->codehub->GetOptTarget(OptOption::CreateOptionByPoint(c));
      oc = ocs[guard_ocs_size];
    }
    if (c->stat == JitCompileResults::GRAPH_CAPTURED) {
      PyFrameObject *f = PrepareCallCompiledCallable(tstate, c->f, c);
      frame = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(f));
      func = py::reinterpret_borrow<py::object>(c->compiled.callable);
    } else {
      HandleBreakGraph(c, oc);
    }
  }
  if (c->stat == JitCompileResults::GRAPH_CAPTURED) {
    auto f = reinterpret_cast<PyFrameObject *>(frame.ptr());
    std::vector<PyObject *> locals;
    BackupLocals(f, &locals);
    GraphCompile(c, oc, func, frame);
    RestoreLocals(f, &locals);
  }
  MS_LOG(DEBUG) << "---compile " << origin_code_name << " successful---";
  if (c->conf->GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    GRAPH_JIT_LOG_F("%s\n", c->tbs->Dump().c_str());
  }
  MS_EXCEPTION_IF_CHECK_FAIL(c->stat == JitCompileResults::GRAPH_CALLABLE, "unknown compile state");
  return true;
}

static std::vector<py::object> PackArgs(const PyFrameObject *frame) {
  const Py_ssize_t argc = frame->f_code->co_argcount + frame->f_code->co_kwonlyargcount;
  bool has_varg = frame->f_code->co_flags & CO_VARARGS;
  py::list args(argc);
  py::object vargs;
  py::object kwvargs;
  for (Py_ssize_t i = 0; i < argc; ++i) {
    args[i] = py::reinterpret_borrow<py::object>(frame->f_localsplus[i]);
  }
  if (has_varg) {
    vargs = py::reinterpret_borrow<py::object>(frame->f_localsplus[argc]);
  }
  if (frame->f_code->co_flags & CO_VARKEYWORDS) {
    kwvargs = py::reinterpret_borrow<py::object>(frame->f_localsplus[argc + has_varg]);
  }

  const Py_ssize_t ncells = PyTuple_GET_SIZE(frame->f_code->co_cellvars);
  for (Py_ssize_t i = 0; frame->f_code->co_cell2arg && i < ncells; ++i) {
    Py_ssize_t argi = frame->f_code->co_cell2arg[i];
    if (argi != CO_CELL_NOT_AN_ARG) {
      PyObject *cell = frame->f_localsplus[frame->f_code->co_nlocals + i];
      args[argi] = py::reinterpret_borrow<py::object>(PyCell_GET(cell));
    }
  }
  return {args, vargs, kwvargs};
}

static py::object CallGraph(const JitCompileResults *c, const py::object &args, const py::object &kwvargs) {
  PyObject *res = c->compiled.cFunc(args.ptr(), kwvargs.ptr());
  if (res == NULL && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError, "compiled graph execute failed");
  }
  return py::reinterpret_steal<py::object>(res);
}

static py::object CallCompiledCallable(PyThreadState *tstate, const PyFrameObject *f, const JitCompileResults *c) {
  PyObject *res;
  PyFrameObject *new_f = PrepareCallCompiledCallable(tstate, f, c);
  res = _PyEval_EvalFrameDefault(tstate, new_f, 0);
  if (res == NULL && !PyErr_Occurred()) {
    PyErr_Format(PyExc_RuntimeError, "compiled function failed with unknown error, error bci %d", new_f->f_lasti);
  }
  Py_DECREF(new_f);
  return py::reinterpret_steal<py::object>(res);
}

static py::object CallCompiledResults(PyThreadState *tstate, const PyFrameObject *f, const JitCompileResults *c) {
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    return py::none();
  }

  std::vector<py::object> packed_args = PackArgs(f);
  if (packed_args[1].ptr() != nullptr) {
    PyList_Append(packed_args[0].ptr(), packed_args[1].ptr());
  }
  py::object args = py::reinterpret_steal<py::object>(PyList_AsTuple(packed_args[0].ptr()));
  py::object kwvargs = packed_args[2];
  py::object res = c->compiled.cFunc ? CallGraph(c, args, kwvargs) : CallCompiledCallable(tstate, f, c);

  // dump traceback
  if (c->conf->GetBoolConfig(GraphJitConfig::kPrintTraceback) && c->func != nullptr) {
    // dump all traceback for the root function
    GRAPH_JIT_LOG_F("%s\n", c->tbs->Dump(true).c_str());
  }
  if (!PyErr_Occurred() && c->func != nullptr && c->tbs != nullptr) {
    c->tbs->Clear();
  }
  return res;
}

static bool CheckGuard(JitCompileResults *c, PyFrameObject *f) {
  bool is_guard = c->conf->GetBoolConfig(GraphJitConfig::kEnableGuard);
  is_guard &= !c->sub_routine || c->conf->GetBoolConfig(GraphJitConfig::kGuardSubRoutine);
  if (!is_guard) {
    ValidateCompiledResults(c);
    return true;
  }
  bool valid_res = false;
  OptOptionPtr opt = OptOption::CreateOptionByPoint(c);
  for (auto &oc : c->codehub->GetOptTarget(opt)) {
    OptGuardPtr guard = oc->GetGuard();
    bool print_guard = c->conf->GetBoolConfig(GraphJitConfig::kPrintGuard);
    if (guard != nullptr && guard->Check(f, print_guard)) {
      c->compiled.cFunc = oc->GetNativeFunc();
      PyObject *new_ref = oc->GetPythonCallable();
      Py_XINCREF(new_ref);
      Py_XSETREF(c->compiled.callable, new_ref);
      valid_res = true;
      break;
    }
  }
  ValidateCompiledResults(c);
  MS_LOG(DEBUG) << __FUNCTION__ << (valid_res ? " success !" : " failed !");
  return valid_res;
}

static bool JitCompileWithTry(PyThreadState *tstate, JitCompileResults *c) {
  bool compiled = false;
  try {
    compiled = JitCompile(tstate, c);
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  if (PyErr_Occurred()) {
    compiled = false;
  }
  if (!compiled) {
    MS_LOG(ERROR) << "compiled failed with" << py::error_already_set().what() << " at "
                  << std::string(py::str(reinterpret_cast<PyObject *>(c->f->f_code)));
    c->stat = JitCompileResults::NEVER_COMPILE;
    Py_CLEAR(c->compiled.callable);
    c->compiled.cFunc = nullptr;
    PyErr_Clear();
  }
  return compiled;
}

static py::object CodeHook(PyThreadState *tstate, JitCompileResults *c) {
  PyFrameObject *frame = c->f;
  bool just_compiled = false;
  bool compiled = false;
  switch (c->stat) {
    case JitCompileResults::NEVER_COMPILE:
      break;
    case JitCompileResults::GRAPH_CAPTURED:
      if (c->conf->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
        break;
      }
    /* fallthrough */
    case JitCompileResults::GRAPH_CANDIDATE:
      if (c->conf->GetBoolConfig(GraphJitConfig::kCompileWithoutCapture)) {
        c->stat = JitCompileResults::GRAPH_CAPTURED;
      }
      if (c->conf->GetBoolConfig(GraphJitConfig::kCompileWithTry)) {
        compiled = JitCompileWithTry(tstate, c);
      } else {
        compiled = JitCompile(tstate, c);
      }
      if (!compiled) {
        c->stat = JitCompileResults::NEVER_COMPILE;
        break;
      }
      just_compiled = true;
    /* fallthrough */
    case JitCompileResults::GRAPH_CALLABLE: {
      std::vector<PyObject *> locals;
      BackupLocals(frame, &locals);
      bool check_guard = CheckGuard(c, frame);
      RestoreLocals(frame, &locals);
      if (check_guard) {
        return CallCompiledResults(tstate, frame, c);
      }
      if (!just_compiled) {
        c->stat = JitCompileResults::GRAPH_CANDIDATE;
        return CodeHook(tstate, c);
      }
    }
    /* fallthrough */
    case JitCompileResults::GRAPH_BUILDING:
      MS_LOG(EXCEPTION) << "shouldn't reach here";
      break;
  }
  PyObject *res = _PyEval_EvalFrameDefault(tstate, frame, 0);
  return py::reinterpret_steal<py::object>(res);
}

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)
PyObject *EvalFrame(PyFrameObject *f, int exc) {
  PyThreadState *tstate = PyThreadState_Get();

#else
PyObject *EvalFrame(PyThreadState *tstate, PyFrameObject *f, int exc) {
#endif

  if (kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kCapturedMSadapterForward)) {
    SpecializeForMSAdapterModule(f);
  }

  PyObject *code = reinterpret_cast<PyObject *>(f->f_code);
  JitCompileResults *c = getJitCompileResults(code, false);
  if (c == nullptr || exc) {
    return _PyEval_EvalFrameDefault(tstate, f, exc);
  }
  c->f = f;
  py::object res;
  try {
    res = CodeHook(tstate, c);
  } catch (py::error_already_set &e) {
    MS_LOG(ERROR) << "execute failed with" << e.what() << " at "
                  << std::string(py::str(reinterpret_cast<PyObject *>(f->f_code)));

    e.restore();
  }
  if (PyErr_Occurred()) {
    res = py::object();
  }
  c->f = nullptr;
  return res.inc_ref().ptr();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

namespace mindspore {

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 9 || PY_MINOR_VERSION == 7)

py::bool_ pi_jit_enable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  if (prev != _PyEval_EvalFrameDefault) {
    return false;
  }
  mindspore::jit::graph::ensureInitialize();
  _PyInterpreterState_SetEvalFrameFunc(inter, mindspore::jit::graph::EvalFrame);
  return true;
}

py::bool_ pi_jit_disable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  if (prev != mindspore::jit::graph::EvalFrame) {
    return false;
  }
  _PyInterpreterState_SetEvalFrameFunc(inter, _PyEval_EvalFrameDefault);
  return true;
}

// bellowing code is used for debugging code generate, and will be remove soon
py::object test_graph_ir_code_gen(const py::object &func) {
  mindspore::jit::graph::Utils::DisFuncObject(func.ptr());
  auto byteCodeParser = std::make_shared<mindspore::jit::graph::ByteCodeParser>(func);
  mindspore::jit::graph::ir::FunctionNodePtr func_node = byteCodeParser->Parse();
  auto inliner = std::make_shared<mindspore::jit::graph::FuncInliner>(func_node);
  inliner->Run();
  auto func_obj = mindspore::jit::graph::ByteCodeGenerator::GenFunction(func_node);
  mindspore::jit::graph::Utils::DisFuncObject(func_obj.ptr());
  func_obj.inc_ref();
  return func_obj;
}

py::bool_ pi_jit_should_compile(const py::object &funcHandle, const py::object &tag) {
  PyObject *func = funcHandle.ptr();
  PyObject *code = NULL;
  if (PyFunction_Check(func)) {
    code = PyFunction_GET_CODE(func);
  } else if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
    code = PyFunction_GET_CODE(func);
  } else if (PyCode_Check(func)) {
    code = func;
  } else {
    return false;
  }
  mindspore::jit::graph::JitCompileResults *c = mindspore::jit::graph::getJitCompileResults(code);
  if (c == NULL) {
    return false;
  }
  c->stat = mindspore::jit::graph::JitCompileResults::GRAPH_CANDIDATE;
  c->func = PyCode_Check(func) ? nullptr : func;
  std::string raw_func_name = "";
  if (func != nullptr && PyFunction_Check(func)) {
    raw_func_name = mindspore::jit::graph::Utils::GetPyName(reinterpret_cast<PyFunctionObject *>(func)->func_qualname);
  }
  std::string raw_func_info_name = "";
  int raw_code_size = 0;
  if (code) {
    raw_func_info_name = py::str(code).cast<std::string>();
    raw_code_size = (PyBytes_GET_SIZE(reinterpret_cast<PyCodeObject *>(code)->co_code)) / sizeof(_Py_CODEUNIT);
  }
  if (c->conf != nullptr) {
    delete c->conf;
  }
  if (c->tbs != nullptr) {
    delete c->tbs;
  }
  c->conf = new mindspore::jit::graph::GraphJitConfig(tag);
  c->tbs = new mindspore::jit::graph::Tracebackes(raw_func_name, raw_func_info_name, raw_code_size);
  c->sub_routine = false;
  if (PyFunction_Check(func)) {
    const char *module_name = PyUnicode_AsUTF8(PyFunction_GET_MODULE(func));
    const char *s = strchr(module_name, '.');
    std::string top_module = s ? std::string(module_name, s - module_name) : module_name;
    mindspore::jit::graph::kPIJitConfigDefault.AddAllowedInlineModules(top_module);
  }
  return true;
}
#else

py::bool_ pi_jit_enable() { return py::bool_(false); }
py::bool_ pi_jit_disable() { return py::bool_(false); }
py::bool_ pi_jit_should_compile(const py::object &func, const py::object &tag) {
  MS_LOG(WARNING) << "GraphJit not support this python version " << PY_MAJOR_VERSION << '.' << PY_MINOR_VERSION
                  << " only support on python3.9 or python3.7";
  return py::bool_(false);
}
// bellowing code is used for debugging code generate, and will be remove soon
py::object test_graph_ir_code_gen(const py::object &func) { return py::bool_(false); }

#endif

}  // namespace mindspore
