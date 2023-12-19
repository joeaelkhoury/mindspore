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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_execute.h"
#include "kernel/pyboost/op_runner.h"
#include "kernel/pyboost/op_register.h"
#include "pipeline/pynative/forward/forward_task.h"

#ifndef MS_UNLIKELY
#ifdef _MSC_VER
#define MS_UNLIKELY(x) (x)
#define MS_LIKELY(x) (x)
#else
#define MS_LIKELY(x) __builtin_expect(!!(x), 1)
#define MS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#endif
namespace mindspore {
namespace pynative {
class PyNativeExecutor;
namespace PyNativeAlgo {
// Common function
struct Common {
  static AbstractBasePtr SetAbstractValueToAnyValue(const AbstractBasePtr &abs);
  static AnfNodePtr ConvertValueSequenceToMakeTuple(const ValueNodePtr &node, const FuncGraphPtr &func_graph);
  static std::string GetIdByValue(const ValuePtr &v);
  static std::string GetCellId(const std::string &obj_id, const std::vector<std::string> &input_arg_id_vec,
                               const std::vector<ValuePtr> &input_arg_value_vec);
  static void SplitString(const std::string &str, std::vector<std::string> *id_vec);
  static bool ValueHasDynamicShape(const ValuePtr &value);
  static bool IsTensor(const ValuePtr &v, bool include_sequence = false);
  static bool IsControlFlowGraph(const FuncGraphPtr &func_graph);
  static ValuePtr FilterSensValues(const ValuePtr &value);
  static tensor::TensorPtr GetTensorFromParam(const AnfNodePtr &param_node);
  static void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph);
  static TypeId GetTypeFromAbstract(const abstract::AbstractBasePtr &abs);
  static ShapeVector GetShapeFromAbstract(const abstract::AbstractBasePtr &abs);
  static ValuePtr CreatOutputTensorValueByAbstract(const abstract::AbstractBasePtr &abs);
  static void ReplaceCNodeWithValueNode(const FuncGraphPtr &bprop_graph);
  static const std::shared_ptr<PyNativeExecutor> &GetPyNativeExecutor();
  static void StubNodeToValue(const FrontendOpRunInfoPtr &op_run_info);
  static TensorPtr StubNodeToTensor(const ValuePtr &value);
  static std::optional<tensor::TensorPtr> StubNodeToTensorOptional(const std::optional<ValuePtr> &value);
  static ValueTuplePtr StubNodeToValueTuple(const ValuePtr &v);
  static void GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                  const std::string &device_target, bool is_dynamic_shape,
                                  mindspore::HashSet<size_t> *input_to_attr_index);
  static ValueNodePtr CreateValueNodeByValue(const ValuePtr &v, const abstract::AbstractBasePtr &abs = nullptr);
  static ValuePtr CreateFakeValueWithoutDeviceAddress(const ValuePtr &value);
  static tensor::TensorPtr CreateFakeTensorWithoutDeviceAddress(const tensor::TensorPtr &tensor);
  static inline bool IsParam(TensorGradType grad_type) {
    return grad_type == TensorGradType::kParameter || grad_type == TensorGradType::kInput;
  }
  static inline bool IsConstant(TensorGradType grad_type) { return grad_type == TensorGradType::kConstant; }
  static TensorGradType SetValueGradInfo(const ValuePtr &value, const TopCellInfoPtr &top_cell,
                                         TensorGradType grad_type);
  static TensorGradType SetTensorGradInfo(const tensor::TensorPtr &tensor, const TopCellInfoPtr &top_cell);
  static void SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph,
                                          const TopCellInfoPtr &top_cell);
  static void ProcessTupleParam(const FuncGraphPtr &bprop_graph, size_t position);
  static void FreeFuncGraphForwardNodes(const FuncGraphPtr &func_graph);
};

// Parser python
struct PyParser {
  static std::string GetIdByPyObj(const py::object &obj);
  static std::pair<std::vector<std::string>, std::vector<ValuePtr>> GetArgsIdAndValue(const py::args &args);
  static void SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg);
  static void ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs,
                                      bool stub = false);
  static void PrepareOpGradInfo(const FrontendOpRunInfoPtr &op_run_info);
};

// Data convert
struct DataConvert {
  static py::object ValueToPyObj(const ValuePtr &v);
  static ValuePtr PyObjToValue(const py::object &obj, bool stub = false);
  static ValuePtr BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence);
  static ValuePtr VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence);
  static void FlattenValueSeqArg(const ValuePtr &v, std::vector<ValuePtr> *flatten_v);
  static void FlattenArgs(const std::vector<ValuePtr> &v_vec, std::vector<ValuePtr> *flatten_v, bool has_sens);
  static void GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const TopCellInfoPtr &top_cell);
  static void ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                           const tensor::CSRTensorPtr &csr_tensor, const TopCellInfoPtr &top_cell,
                                           size_t index);
  static void ConvertMapTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::MapTensorPtr &map_tensor,
                               const TopCellInfoPtr &top_cell, size_t index);
  static ValuePtr ConvertValueDictToValueTuple(const ValuePtr &v);
  static void PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                       size_t index, const TopCellInfoPtr &top_cell);
  static void ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                        size_t index, const TopCellInfoPtr &top_cell);
  static void MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                         const TopCellInfoPtr &top_cell);
  static bool RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                           size_t input_index);
};

struct PyBoost {
  static FrontendOpRunInfoPtr Init(const py::args &args);
  static void DoGrad(const FrontendOpRunInfoPtr &op_run_info);
  static void MakeOutputValue(const FrontendOpRunInfoPtr &op_run_info, const std::vector<TensorPtr> &outputs);
  static void UpdateOutputTensorGradInfo(const std::vector<TensorPtr> &outputs);
  static void UpdateStubOutput(const FrontendOpRunInfoPtr &op_run_info, const AbstractBasePtr &abstract);
  static void UpdateOpRunInfo(const kernel::pyboost::OpPtr &op, const vector<ValuePtr> &op_inputs,
                              const FrontendOpRunInfoPtr &op_run_info);
  static py::object RunPyFunction(const py::args &args);
  template <typename T>
  static ValuePtr OptionalToValue(const std::optional<T> &val) {
    if (!val.has_value()) {
      return kNone;
    }
    return val.value();
  }

  template <typename Tuple, size_t... N>
  static std::vector<ValuePtr> TupleToVector(const Tuple &tuple, std::index_sequence<N...>) {
    std::vector<ValuePtr> inputs;
    ((void)inputs.emplace_back(OptionalToValue(std::get<N>(tuple))), ...);
    return inputs;
  }

  template <typename T>
  static T OptionalToValue(const T &val) {
    return val;
  }

  template <typename... T>
  static auto SetPyBoostCastForInputs(const FrontendOpRunInfoPtr &op_run_info, T... t) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    // For auto grad use

    if (op_run_info->base_op_run_info.op_name == kCast) {
      return std::make_tuple(t...);
    }
    const auto &pyboost_cast_operation = Common::GetPyNativeExecutor()->forward_executor()->pyboost_cast_operation();
    const auto &ret = pyboost_cast_operation->DoMixPrecisionCast(op_run_info, t...);
    op_run_info->op_grad_info->input_value = TupleToVector(ret, std::make_index_sequence<sizeof...(t)>());
    op_run_info->input_size = sizeof...(t);
    return pyboost_cast_operation->DoImplicitCast(op_run_info, ret);
  }
};

// Some common functions used in both jit and PackFunc grad
struct GradCommon {
  static bool IsRealOp(const AnfNodePtr &cnode);
  static void GetUsedCNodeInBpropGraph(const CNodePtr &cnode, const mindspore::HashSet<size_t> &unused_inputs,
                                       AnfNodePtrList *node_list);
  static void SetForward(const AnfNodePtrList &node_list);
};
};  // namespace PyNativeAlgo

void DispatchOp(const std::shared_ptr<FrontendTask> &task);
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
