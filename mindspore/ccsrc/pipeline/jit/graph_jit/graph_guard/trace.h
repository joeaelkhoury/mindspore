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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_TRACE_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_TRACE_H

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "pipeline/jit/graph_jit/pydef.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace jit {
namespace graph {

typedef enum _TraceType {
  Unknown = 0,
  Global,
  BuiltIn,
  Local,
  Param,
  Const,
  Item,
  Attr,
  Type,
  Operation,
  Customized,
} TraceType;

typedef struct _TraceContext {
  PyObject *f_globals;
  PyObject *f_builtins;
  PyObject *f_locals;
  PyObject **f_localsplus;
  PyCodeObject *f_code;
} TraceContext, *PTraceContext;

class Trace {
 public:
  Trace(PyObject *obj, std::shared_ptr<Trace> origin);
  virtual ~Trace();
  virtual std::shared_ptr<Trace> GetOrigin();
  /// \brief Get the borrow reference for the object and call Py_INCREF/Py_DECREF by yourself.
  /// \param[out] borrow reference for PyObject
  virtual PyObject *GetObject();
  virtual TraceType GetTraceType();
  virtual TraceType GetOriginType();
  virtual void Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src);
  virtual bool operator==(const Trace &trace);
  /// \brief Get the reference for the object by Py_INCREF and call Py_DECREF by yourself.
  /// \param[out] borrow reference for PyObject
  virtual PyObject *Retrieve(PTraceContext context) = 0;
  virtual std::string ToString() = 0;

 protected:
  PyObject *obj_;
  std::shared_ptr<Trace> origin_;
  TraceType originType_;
  TraceType curType_;
};
using TracePtr = std::shared_ptr<Trace>;
using TraceVector = std::vector<TracePtr>;

class RootTrace : public Trace {
 public:
  RootTrace(PyObject *obj, TraceType tt, int index = -1, std::string name = "", std::string module_name = "");
  virtual ~RootTrace() = default;
  virtual PyObject *Retrieve(PTraceContext context);
  virtual std::string ToString();
  virtual void GetParam(int *index, std::string *name, std::string *module_name);
  virtual bool operator==(const Trace &trace);

 protected:
  int idx_;
  std::string name_;
  std::string module_name_;
};
using RootTracePtr = std::shared_ptr<RootTrace>;

class ItemTrace : public Trace {
 public:
  ItemTrace(PyObject *obj, TracePtr origin, TracePtr item);
  virtual ~ItemTrace() = default;
  virtual TracePtr GetItem();
  virtual void Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src);
  virtual PyObject *Retrieve(PTraceContext context);
  virtual std::string ToString();
  virtual bool operator==(const Trace &trace);

 protected:
  TracePtr item_;
};
using ItemTracePtr = std::shared_ptr<ItemTrace>;

class AttrTrace : public Trace {
 public:
  AttrTrace(PyObject *obj, TracePtr origin, std::string attr);
  virtual ~AttrTrace() = default;
  virtual std::string GetAttribute();
  virtual PyObject *Retrieve(PTraceContext context);
  virtual std::string ToString();
  virtual bool operator==(const Trace &trace);

 protected:
  std::string attr_;
};
using AttrTracePtr = std::shared_ptr<AttrTrace>;

class ConstTrace : public Trace {
 public:
  ConstTrace(PyObject *obj, int index);
  virtual ~ConstTrace() = default;
  virtual int GetIndex();
  virtual PyObject *Retrieve(PTraceContext context);
  virtual std::string ToString();
  virtual bool operator==(const Trace &trace);

 protected:
  int index_;
};
using ConstTracePtr = std::shared_ptr<ConstTrace>;

class TypeTrace : public Trace {
 public:
  TypeTrace(PyObject *obj, TracePtr origin);
  virtual ~TypeTrace() = default;
  virtual PyTypeObject *GetType();
  virtual PyObject *Retrieve(PTraceContext context);
  virtual std::string ToString();
  virtual bool operator==(const Trace &trace);

 protected:
  PyTypeObject *pType_;
};
using TypeTracePtr = std::shared_ptr<TypeTrace>;

class OpTrace : public Trace {
 public:
  OpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string name = "");
  virtual ~OpTrace() = default;
  virtual void Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src);
  virtual PyObject *Retrieve(PTraceContext context);
  virtual std::string ToString();
  virtual bool operator==(const Trace &trace);

 protected:
  int opcode_;
  int opargs_;
  TraceVector params_;
  std::string name_;
};
using OpTracePtr = std::shared_ptr<OpTrace>;
TracePtr CreateOpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string module_name = "",
                       std::string name = "", bool print = false);

/// \brief retrieve the PyObject with ref count plus 1 which will be minus outside
typedef std::function<PyObject *(PTraceContext context)> RetrieveFunc;
typedef std::function<std::string()> ToStringFunc;
class CustomizedTrace : public Trace {
 public:
  CustomizedTrace(PyObject *obj, RetrieveFunc rfunc, ToStringFunc sfunc);
  virtual ~CustomizedTrace() = default;
  virtual PyObject *Retrieve(PTraceContext context);
  virtual std::string ToString();

 protected:
  RetrieveFunc retrieve_;
  ToStringFunc tostring_;
};
using CustomizedTracePtr = std::shared_ptr<CustomizedTrace>;

/// \brief Get the reference for the object by Py_INCREF and call Py_DECREF by yourself.
PyObject *GetObjectFromTrace(PyFrameObject *frame, TracePtr trace);
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_TRACE_H
