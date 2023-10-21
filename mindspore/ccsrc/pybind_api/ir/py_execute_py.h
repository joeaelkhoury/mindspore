/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

// NOTICE: This header file should only be included once in the whole project.
// We change the cpp file to header file, to avoid MSVC compiler problem.
#ifndef MINDSPORE_CCSRC_PYBINDAPI_IR_PY_EXECUTE_PY_H_
#define MINDSPORE_CCSRC_PYBINDAPI_IR_PY_EXECUTE_PY_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"

#include "include/common/fallback.h"
#include "mindspore/core/ops/py_execute.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils_py.h"
#include "mindspore/ccsrc/include/common/utils/python_utils.h"
#include "mindspore/ccsrc/include/common/utils/python_adapter.h"
#include "mindspore/ccsrc/include/common/utils/python_fallback_running.h"
#include "mindspore/ccsrc/include/backend/optimizer/helper.h"
#include "mindspore/ccsrc/pipeline/jit/ps/parse/data_converter.h"
#include "mindspore/ccsrc/pybind_api/ir/tensor_py.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "mindspore/ccsrc/pipeline/jit/ps/parse/resolve.h"
#include "include/common/utils/convert_utils_py.h"

namespace py = pybind11;
namespace mindspore {
namespace abstract {
using PyObjectWrapperPtr = std::shared_ptr<parse::PyObjectWrapper>;
namespace pyexecute_user_data_catcher {
std::pair<bool, ValuePtr> PyExecuteUserDataCatcher(const AbstractBasePtr &element_abs) {
  MS_EXCEPTION_IF_NULL(element_abs);
  if (element_abs->has_user_data<kernel::PyExecuteOutputUserData>()) {
    const auto &data = element_abs->user_data<kernel::PyExecuteOutputUserData>();
    MS_EXCEPTION_IF_NULL(data);
    auto python_obj = std::make_shared<parse::PyObjectWrapper>(data->obj, "graph python obj");
    return {true, python_obj};
  }
  return {false, nullptr};
}

struct PyExecuteUserDataCatcherRegister {
  PyExecuteUserDataCatcherRegister() noexcept {
    abstract::AbstractBase::set_pyexecute_user_data_catcher(
      [](const AbstractBasePtr &element_abs) { return PyExecuteUserDataCatcher(element_abs); });
  }
  ~PyExecuteUserDataCatcherRegister() {}
} pyexecute_user_data_catcher_register;
}  // namespace pyexecute_user_data_catcher
}  // namespace abstract

bool ContainStubTensor(const py::object &obj) {
  if (py::isinstance<py::list>(obj)) {
    auto list_obj = py::cast<py::list>(obj);
    return std::any_of(list_obj.begin(), list_obj.end(),
                       [](const auto &e) { return ContainStubTensor(py::cast<py::object>(e)); });
  }
  if (py::isinstance<py::tuple>(obj)) {
    auto tuple_obj = py::cast<py::tuple>(obj);
    return std::any_of(tuple_obj.begin(), tuple_obj.end(),
                       [](const auto &e) { return ContainStubTensor(py::cast<py::object>(e)); });
  }
  if (py::isinstance<py::dict>(obj)) {
    auto dict_obj = py::cast<py::dict>(obj);
    return std::any_of(dict_obj.begin(), dict_obj.end(), [](const auto &e) {
      return ContainStubTensor(py::cast<py::object>(e.first)) || ContainStubTensor(py::cast<py::object>(e.second));
    });
  }
  return IsStubTensor(obj);
}

class PyExecuteInitializer {
 public:
  PyExecuteInitializer() {
    mindspore::ops::PyExecuteInfer::set_infer_handler(PyExecuteInferPy);
    mindspore::opt::SetCppInferPyHanbdler(CppInferShapeAndTypePy);
  }

  ~PyExecuteInitializer() = default;

 private:
  static abstract::AbstractBasePtr PyExecuteInferPy(const std::vector<AbstractBasePtr> &input_args) {
    const auto &script_abs = input_args[0];
    const auto &script = script_abs->BuildValue();
    const auto &script_str = dyn_cast<StringImm>(script);

    const auto &keys_tuple_abs = input_args[1];
    const auto &keys_tuple = keys_tuple_abs->BuildValue();
    const auto &keys = dyn_cast<ValueSequence>(keys_tuple);
    if (keys == nullptr) {
      MS_LOG(DEBUG) << "The keys is not tuple value, but got " << keys_tuple->ToString();
      const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
      return abstract::MakeAbstract(infer_shape, kFloat64);
    }
    constexpr auto number_two = 2;
    const auto &values_tuple_abs = input_args[number_two];
    const auto &values_tuple = values_tuple_abs->BuildValue();
    if (values_tuple == kValueAny) {
      MS_LOG(EXCEPTION) << "Value tuple should not be anyvalue.";
    }
    const auto &values = dyn_cast<ValueSequence>(values_tuple);
    if (values == nullptr) {
      MS_LOG(DEBUG) << "The values is not tuple value, but got " << keys_tuple->ToString();
      const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
      return abstract::MakeAbstract(infer_shape, kFloat64);
    }
    MS_LOG(DEBUG) << "The script is: " << script->ToString() << ", keys_tuple: " << keys_tuple->ToString()
                  << ", values_tuple: " << values_tuple->ToString();
    if (keys->size() != values->size()) {
      MS_LOG(EXCEPTION) << "The length of keys(" << keys->size() << ") is not equal of the length of values("
                        << values->size() << ").";
    }
    py::gil_scoped_acquire gil_acquire;
    py::dict local_dict;

    for (size_t i = 0; i < keys->size(); ++i) {
      const auto &key = (*keys)[i];
      const auto &key_str = dyn_cast<StringImm>(key);
      MS_EXCEPTION_IF_NULL(key_str);
      const auto &value = (*values)[i];
      MS_LOG(DEBUG) << "input[" << i << "], value : " << value->ToString();
      const auto &tuple_abs = values_tuple_abs->cast<abstract::AbstractSequencePtr>();
      const auto &value_abs = (*tuple_abs)[i];
      if (value_abs->has_user_data<kernel::PyExecuteOutputUserData>()) {
        const auto &output_data = value_abs->user_data<kernel::PyExecuteOutputUserData>();
        auto obj = output_data->obj;
        MS_LOG(DEBUG) << "input[" << i << "] convert value from user data, obj: " << obj;
        local_dict[py::str(key_str->value())] = obj;
      } else {
        auto obj = ValueToPyData(value, value_abs);
        local_dict[py::str(key_str->value())] = obj;
        MS_LOG(DEBUG) << "input[" << i << "] convert value from abstract, obj: " << obj;
      }
    }
    const auto &py_script = py::str(script_str->value());
    auto params = py::tuple(number_two);
    params[0] = py::dict();
    params[1] = local_dict;
    MS_LOG(DEBUG) << "Python script: " << py_script << ", local_dict: " << local_dict;
    try {
      mindspore::ScopedFallbackRunning fallback_running;
      const auto &output = parse::data_converter::CallPythonScript(py_script, params);
      if (ContainStubTensor(output)) {
        MS_EXCEPTION(TypeError) << "PyExecute node output can not contain stub tensor.";
      }
      MS_LOG(DEBUG) << "Python output type: " << py::str(output.get_type()) << ", output: " << output;
      fallback::PushPyExecuteOutput(output);
      if (py::isinstance<tensor::Tensor>(output) || IsStubTensor(output)) {
        const auto &tensor = IsStubTensor(output) ? ConvertStubTensor(output) : output.cast<tensor::TensorPtr>();
        const auto &infer_shape = std::make_shared<abstract::Shape>(tensor->shape());
        return tensor->ToAbstract();
      }
    } catch (const py::error_already_set &e) {
      auto error_type_name = py::cast<std::string>(python_adapter::GetPyObjAttr(e.type(), "__name__"));
      auto error_iter = exception_types_map.find(error_type_name);
      if (error_iter != exception_types_map.end()) {
        auto &handler = LogWriter::GetExceptionHandler();
        if (handler != nullptr) {
          handler(error_iter->second, py::str(e.value()));
        }
      }
      throw std::runtime_error(py::str(e.value()));
    }

    const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
    return abstract::MakeAbstract(infer_shape, kFloat64);
  }

  static abstract::AbstractBasePtr CppInferShapeAndTypePy(const CNodePtr &cnode, const PrimitivePtr &primitive,
                                                          const AbstractBasePtrList &args_abs_list) {
    // We can't catch the pybind11 exception by py::builtin_exception or its base class,
    // so we have to list all pybind11 exceptions and catch one by one here.
    AbstractBasePtr res;
    std::function<void(void)> already_set_error_handler;
    std::function<void(void)> other_error_handler;
    std::function<void(void)> default_error_handler;
    HandleExceptionRethrow(
      [&res, &cnode, &primitive, &args_abs_list]() {
        res = opt::CppInferShapeAndType(primitive, args_abs_list);
        MS_LOG(DEBUG) << "The abstract of " << cnode->fullname_with_scope() << " changes from " << cnode->abstract()
                      << " to " << res;
        return res;
      },
      already_set_error_handler, other_error_handler, default_error_handler,
      cnode->debug_info());  // Use debug_info to re-throw.
    return res;
  }
};

static PyExecuteInitializer py_execute_initializer;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBINDAPI_IR_PY_EXECUTE_PY_H_
