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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_CACHE_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_CACHE_H

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <Python.h>
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi_jit/graph_guard/guard.h"

namespace mindspore {
namespace jit {
namespace graph {
using NativeFunc = std::function<PyObject *(PyObject *, PyObject *)>;
using ReleaseFunc = std::function<void()>;

/// \brief OptOption is the compilation option for the code
class OptOption : public std::enable_shared_from_this<OptOption> {
 public:
  /// \brief no support for default construction and you can extend the option class to support more feature
  OptOption() = delete;
  virtual ~OptOption() = default;
  /// \brief support create option by PyCodeObject
  static std::shared_ptr<OptOption> CreateOptionByCode(PyCodeObject *code);
  static std::shared_ptr<OptOption> CreateOptionByPoint(void *ptr);
  bool operator==(const OptOption &obj) const;

 protected:
  explicit OptOption(PyCodeObject *code);
  explicit OptOption(void *ptr);
  void *target_;
};
using OptOptionPtr = std::shared_ptr<OptOption>;

/// \brief optimized code with native function graph and guard based on the compilation option
class OptCode : public std::enable_shared_from_this<OptCode> {
 public:
  OptCode();
  virtual ~OptCode();
  virtual void SetPhase(std::string phase);
  virtual void SetNativeFunc(NativeFunc cFunc, ReleaseFunc rFunc = nullptr);
  virtual NativeFunc GetNativeFunc();
  virtual void SetPythonCallable(PyObject *pFunc);
  virtual PyObject *GetPythonCallable();
  virtual void SetGuard(OptGuardPtr guard);
  virtual OptGuardPtr GetGuard();
  virtual void SetOption(OptOptionPtr option);
  virtual OptOptionPtr GetOption();

 protected:
  std::string phase_;
  NativeFunc cFunc_;
  ReleaseFunc rFunc_;
  PyObject *pFunc_;
  OptGuardPtr guard_;
  OptOptionPtr option_;
};
using OptCodePtr = std::shared_ptr<OptCode>;
using OptCodeSet = std::vector<OptCodePtr>;

/// \brief hub for optimized code based on compilation option
class OptCodeHub : public std::enable_shared_from_this<OptCodeHub> {
 public:
  OptCodeHub() = default;
  virtual ~OptCodeHub() = default;
  virtual OptCodePtr AddOptTarget(OptOptionPtr option);
  virtual OptCodeSet GetOptTarget(OptOptionPtr option);
  virtual void DelOptTarget(OptOptionPtr option, OptCodePtr code);

 protected:
  std::map<OptOptionPtr, OptCodeSet> codeMap_;
};

using OptCodeHubPtr = std::shared_ptr<OptCodeHub>;
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_CACHE_H
