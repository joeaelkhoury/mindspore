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

#include "frontend/optimizer/py_interpret_to_execute.h"

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_function.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "utils/interpret_node_recorder.h"
#include "utils/symbolic.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/fallback.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
namespace {
py::object CallPythonPushGlobalParams(const py::object &dict) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  constexpr auto python_merge_dict = "merge_global_params";
  return python_adapter::CallPyModFn(mod, python_merge_dict, dict);
}

void FuncGraphToPyData(const ValueDictionaryPtr &value_dict, py::object *global_params_dict) {
  MS_EXCEPTION_IF_NULL(value_dict);
  MS_EXCEPTION_IF_NULL(global_params_dict);
  for (const auto &element : value_dict->value()) {
    const auto &element_name = element.first;
    const auto &element_abs = element.second;
    if (element_abs->IsFromTypeId(FuncGraph::kTypeId)) {
      auto fg = element_abs->cast<FuncGraphPtr>();
      MS_EXCEPTION_IF_NULL(fg);
      auto wrapper_obj = fg->python_obj();
      if (wrapper_obj != nullptr && wrapper_obj->isa<parse::PyObjectWrapper>()) {
        auto fn_py_obj = wrapper_obj->cast_ptr<parse::PyObjectWrapper>()->obj();
        (*global_params_dict)[ValueToPyData(element_name)] = fn_py_obj;
        MS_LOG(DEBUG) << "Found python function object for " << element_name << ", add it to global dict.";
      }
    }
  }
  return;
}
}  // namespace

// Convert PyInterpret into PyExecute:
//   PyInterpret(script, global_dict, local_dict)
//   -->
//   PyExecute(script, local_dict_keys, local_dict_values),
//   with side-effect operation:
//     Push global_dict into global parameters list.
//     (So it requires no same key name.)
bool PyInterpretToExecute(const pipeline::ResourcePtr &resource) {
  auto manager = resource->manager();
  const auto &all_nodes = manager->all_nodes();
  auto transact = manager->Transact();
  constexpr auto input_index_one = 1;
  constexpr auto input_index_two = 2;
  constexpr auto input_index_three = 3;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimPyInterpret)) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_LOG(DEBUG) << "cnode: " << cnode->DebugString();
    auto new_cnode = std::make_shared<CNode>(*cnode);
    new_cnode->CloneUserData(cnode);
    new_cnode->set_input(0, NewValueNode(prim::kPrimPyExecute));

    if (!IsValueNode<parse::Script>(cnode->input(input_index_one))) {
      MS_LOG(INTERNAL_EXCEPTION) << "The first input should be a Script, but got "
                                 << cnode->input(input_index_one)->DebugString();
    }
    const auto &script = GetValueNode<std::shared_ptr<parse::Script>>(cnode->input(input_index_one));
    const auto &script_str = script->script();
    const auto &script_strimm_node = NewValueNode(std::make_shared<StringImm>(script_str));
    new_cnode->set_input(input_index_one, script_strimm_node);

    if (!IsValueNode<ValueDictionary>(cnode->input(input_index_two))) {
      MS_LOG(INTERNAL_EXCEPTION) << "The second input should be a dictionary, but got "
                                 << cnode->input(input_index_two)->DebugString();
    }
    const auto &global_dict = GetValueNode<ValueDictionaryPtr>(cnode->input(input_index_two));
    auto value_dict = global_dict->cast<ValueDictionaryPtr>();
    py::object py_global_dict = ValueToPyData(global_dict);
    FuncGraphToPyData(value_dict, &py_global_dict);
    MS_LOG(DEBUG) << "py_global_dict: " << py::str(py_global_dict);
    (void)CallPythonPushGlobalParams(py_global_dict);

    const auto &three_input = cnode->input(input_index_three);
    if (!IsPrimitiveCNode(three_input, prim::kPrimMakeDict) && !IsValueNode<ValueDictionary>(three_input)) {
      MS_LOG(INTERNAL_EXCEPTION) << "The 3rd input should be a dictionary, but got " << three_input->DebugString();
    }

    AnfNodePtr local_dict_keys = nullptr;
    AnfNodePtr local_dict_values = nullptr;
    if (IsValueNode<ValueDictionary>(three_input)) {
      const auto &local_dict = GetValueNode<ValueDictionaryPtr>(three_input);
      auto value_local_dict = local_dict->cast<ValueDictionaryPtr>();
      if (value_local_dict->value().empty()) {
        auto str_value = std::make_shared<StringImm>("None");
        std::vector<ValuePtr> none_value{str_value};
        const auto none_tuple = std::make_shared<ValueTuple>(none_value);
        auto none_tuple_node = NewValueNode(none_tuple);
        local_dict_keys = none_tuple_node;
        local_dict_values = none_tuple_node;
      } else {
        std::vector<ValuePtr> key_vec;
        std::vector<ValuePtr> value_vec;
        for (auto key_value : value_local_dict->value()) {
          (void)key_vec.emplace_back(key_value.first);
          (void)value_vec.emplace_back(key_value.second);
        }
        local_dict_keys = NewValueNode(std::make_shared<ValueTuple>(key_vec));
        local_dict_values = NewValueNode(std::make_shared<ValueTuple>(value_vec));
      }
    } else {
      const auto &local_dict_cnode = dyn_cast<CNode>(three_input);
      MS_EXCEPTION_IF_NULL(local_dict_cnode);
      local_dict_keys = local_dict_cnode->input(input_index_one);
      local_dict_values = local_dict_cnode->input(input_index_two);
    }

    if ((!IsValueNode<ValueTuple>(local_dict_keys) && !IsPrimitiveCNode(local_dict_keys, prim::kPrimMakeTuple)) ||
        (!IsValueNode<ValueTuple>(local_dict_values) && !IsPrimitiveCNode(local_dict_values, prim::kPrimMakeTuple))) {
      MS_LOG(INTERNAL_EXCEPTION) << "The dictionary's keys and values should be a tuple, but got "
                                 << three_input->DebugString();
    }

    if (local_dict_values->isa<CNode>()) {
      // Handle values and convert InterpretedObject element.
      const auto &make_tuple_cnode = dyn_cast<CNode>(local_dict_values);
      MS_EXCEPTION_IF_NULL(make_tuple_cnode);
      const auto fg = make_tuple_cnode->func_graph();
      for (size_t i = 1; i < make_tuple_cnode->size(); ++i) {
        // Convert InterpretedObject value node to PyExecute CNode.
        const auto &input = make_tuple_cnode->input(i);
        const auto &value = GetValueNode<parse::InterpretedObjectPtr>(input);
        if (value != nullptr) {
          const auto interpreted_value = dyn_cast<parse::InterpretedObject>(value);
          const std::string &key = interpreted_value->name();
          const py::object &obj = interpreted_value->obj();
          const auto &interpreted_node = fallback::ConvertPyObjectToPyExecute(fg, key, obj, input, true);
          interpreted_node->set_debug_info(input->debug_info());
          (void)transact.Replace(input, interpreted_node);
        }
      }
    }

    new_cnode->set_input(input_index_two, local_dict_keys);
    new_cnode->set_input(input_index_three, local_dict_values);
    (void)transact.Replace(cnode, new_cnode);

    // Record the PyExecute node.
    InterpretNodeRecorder::GetInstance().PushPyExecuteNode(new_cnode);
  }
  transact.Commit();
  return true;
}
}  // namespace opt
}  // namespace mindspore
