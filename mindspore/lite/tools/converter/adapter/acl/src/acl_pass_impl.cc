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

#define USE_DEPRECATED_API
#include "tools/converter/adapter/acl/src/acl_pass_impl.h"
#include <algorithm>
#include <deque>
#include <set>
#include <map>
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/spatial_node_adapter.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/converter_context.h"
#include "include/registry/pass_registry.h"
#include "ops/custom.h"
#include "ops/op_utils.h"
#include "ops/transpose.h"
#include "ops/standard_normal.h"
#include "ops/tuple_get_item.h"
#include "cxx_api/model/acl/model_converter.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "src/common/file_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/graph/specify_graph_input_format.h"
#include "mindspore/core/utils/ms_utils_secure.h"
#include "mindspore/ccsrc/include/backend/optimizer/graph_optimizer.h"
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "tools/optimizer/graph/clip_convert_activation_pass.h"
#include "tools/optimizer/fusion/transpose_fusion.h"
#include "tools/optimizer/fusion/batchnorm_to_scale_fusion.h"
#include "tools/converter/quantizer/quantization_optimizer.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/converter/parser/unify_format.h"
#include "tools/converter/adapter/acl/src/acl_custom_opp_installer.h"
#include "tools/graph_kernel/converter/graph_kernel_optimization.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/converter/quantizer/quant_helper/ascend_distribute_fake_quant_transform.h"
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include "src/common/common.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
namespace opt {
static const std::set<std::string> kAdjustCnodeName = {"Resize", "Conv2dTransposeFusion", "Concat"};
static const std::map<int64_t, std::string> kEnumFormatToStrMap = {{Format::NCHW, "NCHW"}, {Format::NHWC, "NHWC"}};
namespace {
constexpr auto kMakeTuple = "MakeTuple";
constexpr auto kCustomPrimTypeACL = "ACL";
constexpr auto kCustomNodeName = "custom_0";
constexpr auto kNCHWFormat = "NCHW";
constexpr auto kToNHWCFormatPass = "ToNHWCFormat";
constexpr auto kToNCHWFormatPass = "ToNCHWFormat";
constexpr auto kInferShapePass = "InferShapePass";
constexpr auto kConstFoldPass = "ConstFoldPass";
constexpr auto kRemoveRedundantOpPass = "RemoveRedundantOpPass";
constexpr auto kDelRedundantTranspose = "DeleteRedundantTranspose";
constexpr auto kRemoveUnusedAddNodePass = "RemoveUnusedAddNodePass";
constexpr auto kFuncType = "func_type";
constexpr auto kUniqueName = "uniq_name";
constexpr size_t kDependInputNum = 3;
constexpr size_t kDependFirstInputIdx = 1;
constexpr size_t kTupleGetItemFirstInputIdx = 1;

STATUS PreProcForMindIr(const FuncGraphPtr &func_graph, bool offline) {
  auto value = func_graph->get_attr(ops::kFormat);
  if (value == nullptr || GetValue<int64_t>(value) == mindspore::NCHW) {
    return lite::RET_OK;
  }
  if (offline) {
    if (!lite::RunOptimizerPass(func_graph, {kInferShapePass})) {
      MS_LOG(ERROR) << "Infer shape pass failed.";
      return lite::RET_ERROR;
    }
  }
  if (!lite::RunOptimizerPass(func_graph, {kToNCHWFormatPass, "DecreaseTransposeAlgo"})) {
    MS_LOG(ERROR) << "Run ToNCHWFormat pass failed";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS PreProcForTF(const FuncGraphPtr &func_graph, bool offline) {
  if (offline) {
    if (!lite::RunOptimizerPass(func_graph, {kInferShapePass})) {
      MS_LOG(ERROR) << "Infer shape pass failed.";
      return lite::RET_ERROR;
    }
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    CHECK_NULL_RETURN(node);
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    CHECK_NULL_RETURN(cnode);
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    CHECK_NULL_RETURN(prim);
    if (prim->GetAttr(ops::kFormat) != nullptr) {
      auto node_format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
      if (kEnumFormatToStrMap.find(node_format) != kEnumFormatToStrMap.end()) {
        std::string format = kEnumFormatToStrMap.at(node_format);
        prim->AddAttr("io_format", MakeValue(format));
      }
    }
  }
  return lite::RET_OK;
}

STATUS PreProcForCaffe(const FuncGraphPtr &func_graph, bool offline) {
  if (offline) {
    if (!lite::RunOptimizerPass(func_graph, {kInferShapePass})) {
      MS_LOG(ERROR) << "Infer shape pass failed.";
      return lite::RET_ERROR;
    }
  }
  if (!lite::RunOptimizerPass(func_graph, {kToNCHWFormatPass, "DecreaseTransposeAlgo"})) {
    MS_LOG(ERROR) << "To nchw format failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS PreProcForOnnx(const FuncGraphPtr &func_graph, bool offline) {
  if (offline) {
    if (!lite::RunOptimizerPass(func_graph, {kInferShapePass})) {
      MS_LOG(ERROR) << "Infer shape pass failed.";
      return lite::RET_ERROR;
    }
  }
  if (!lite::RunOptimizerPass(func_graph, {kToNCHWFormatPass, "DecreaseTransposeAlgo"})) {
    MS_LOG(ERROR) << "To nchw format failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace

AclPassImpl::AclPassImpl(const std::shared_ptr<ConverterPara> &param)
    : param_(param),
      fmk_type_(param->fmk_type),
      export_mindir_(param->save_type),
      user_options_cfg_(std::move(param->aclModelOptionCfgParam)),
      om_parameter_(nullptr),
      custom_node_(nullptr) {
  is_ptq_quant_ = (param_->commonQuantParam.quant_type == lite::quant::QUANT_ALL &&
                   param_->fullQuantParam.target_device == lite::quant::ASCEND) ||
                  (param_->commonQuantParam.quant_type == lite::quant::QUANT_WEIGHT &&
                   param_->weightQuantParam.dequant_strategy == lite::quant::ON_THE_FLY);
}

bool AclPassImpl::IsDynamicInput() {
  return !user_options_cfg_.dynamic_image_size.empty() || !user_options_cfg_.dynamic_batch_size.empty();
}

STATUS AclPassImpl::RemoveSingleInputConcatNode(const FuncGraphPtr &func_graph) {
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Failed to get func graph manager";
    return lite::RET_ERROR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (!cnode) {
      continue;
    }
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimConcat)) {
      continue;
    }
    if (cnode->size() != kInputSize1) {
      continue;
    }
    auto src_node = cnode->input(kIndex1);
    if (src_node && src_node->abstract() && !src_node->abstract()->isa<abstract::AbstractSequence>()) {
      manager->Replace(cnode, src_node);
    }
  }
  return RET_OK;
}

STATUS AclPassImpl::CommonPass(const FuncGraphPtr &func_graph) {
  if (param_->ascendQuantParam.mode == lite::quant::GE) {
    MS_LOG(INFO) << "Ascend quant mode is GE. It will dont run common pass.";
    return lite::RET_OK;
  }
  if (RemoveSingleInputConcatNode(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "Remove single input concat node failed.";
    return lite::RET_ERROR;
  }
  if (MakeListToMakeTuple(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "Convert make_list to MakeTuple failed.";
    return lite::RET_ERROR;
  }
  if (!lite::RunOptimizerPass(func_graph, {kRemoveRedundantOpPass, kRemoveUnusedAddNodePass})) {
    MS_LOG(ERROR) << "Remove redundant op pass failed.";
    return lite::RET_ERROR;
  }
  if (fmk_type_ == converter::kFmkTypeMs) {
    MS_LOG(INFO) << "Ms model no need to run const fold pass.";
    return lite::RET_OK;
  }
  // Quantization dynamic model must set inputShape for calibration.
  if (is_ptq_quant_ && IsDynamicInput()) {
    MS_LOG(INFO) << "Dynamic input no need to run const fold pass.";
    return lite::RET_OK;
  }
  if (!lite::RunOptimizerPass(func_graph, {kConstFoldPass})) {
    MS_LOG(WARNING) << "Const fold pass failed.";
  }
  return lite::RET_OK;
}

// From:
//   MakeList(arg1, arg2, ...)
// To:
//   MakeTuple(arg1, arg2, ...)
static AnfNodePtr ConvertMakeListToMakeTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  std::vector<AnfNodePtr> inputs;
  inputs.reserve(node->size());
  (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  // Inputs of node should be [make_list, item1, item2, ...], so offset by 1 to get items;
  (void)inputs.insert(inputs.cend(), node->inputs().cbegin() + 1, node->inputs().cend());
  return node->func_graph()->NewCNode(std::move(inputs));
}

// From:
//   list_getitem(list, key)
// To:
//   TupleGetItem(list, key)
static AnfNodePtr ConvertListGetItemToTupleGetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [list_getitem, list, item]
  constexpr size_t expect_inputs_size = 3;
  if (node->size() != expect_inputs_size) {
    std::string op_name = GetCNodeFuncName(node);
    MS_LOG(EXCEPTION) << op_name << " should have " << expect_inputs_size << " inputs, but got " << node->size();
    return nullptr;
  }
  constexpr size_t data_index = 1;
  constexpr size_t cons_index = 2;
  const auto &inputs = node->inputs();
  auto &data = inputs[data_index];
  auto &key = inputs[cons_index];
  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, key});
}

// From:
//   ListSetItem(list, index, item)
// To:
//   TupleSetItem(list, index, item)
static AnfNodePtr ConvertListSetItemToTupleSetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [list_setitem, list, index, item]
  const size_t expect_inputs_size = 4;
  if (node->size() != expect_inputs_size) {
    std::string op_name = GetCNodeFuncName(node);
    MS_LOG(EXCEPTION) << op_name << " should have " << expect_inputs_size << " inputs, but got " << node->size();
    return nullptr;
  }

  const size_t data_index = 1;
  const size_t cons_index = 2;
  const size_t value_index = 3;
  const auto &inputs = node->inputs();
  auto &data = inputs[data_index];
  auto &key = inputs[cons_index];
  auto &value = inputs[value_index];
  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, key, value});
}

static AnfNodePtr ConvertMakeListPrimitiveCNode(const CNodePtr &cnode, const PrimitivePtr &prim) {
  if (prim->name() == prim::kPrimMakeList->name()) {
    return ConvertMakeListToMakeTuple(cnode);
  } else if (prim->name() == prim::kPrimListGetItem->name()) {
    return ConvertListGetItemToTupleGetItem(cnode);
  } else if (prim->name() == prim::kPrimListSetItem->name()) {
    return ConvertListSetItemToTupleSetItem(cnode);
  }
  return nullptr;
}

static constexpr size_t kMaxSeqRecursiveDepth = 6;
static ValuePtr ConvertValueSequenceToValueTuple(const ValuePtr &value, size_t depth, bool *need_convert) {
  MS_EXCEPTION_IF_NULL(need_convert);
  MS_EXCEPTION_IF_NULL(value);
  if (depth > kMaxSeqRecursiveDepth) {
    MS_LOG(EXCEPTION) << "List nesting is not allowed more than " << kMaxSeqRecursiveDepth << " levels.";
  }

  if (value->isa<ValueSequence>()) {
    std::vector<ValuePtr> elements;
    auto value_seq = value->cast<ValueSequencePtr>();
    (void)std::transform(value_seq->value().begin(), value_seq->value().end(), std::back_inserter(elements),
                         [&](const ValuePtr &value) -> ValuePtr {
                           bool is_convert = false;
                           auto convert_value = ConvertValueSequenceToValueTuple(value, depth + 1, &is_convert);
                           *need_convert |= is_convert;
                           return convert_value;
                         });
    *need_convert |= value->isa<ValueList>();
    if (*need_convert) {
      return std::make_shared<ValueTuple>(elements);
    }
  }
  return value;
}

static AnfNodePtr ConvertMakeListValueNode(const ValueNodePtr &value_node, const ValuePtr &value) {
  bool need_convert = false;
  auto convert_value = ConvertValueSequenceToValueTuple(value, 0, &need_convert);
  if (need_convert) {
    return std::make_shared<ValueNode>(convert_value);
  }
  return nullptr;
}

static AnfNodePtr ConvertMakeListNode(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr) {
    if (cnode->size() == 0) {
      return nullptr;
    }
    // Get primitive from cnode.
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      return nullptr;
    }
    // Call primitive cnode converter.
    return ConvertMakeListPrimitiveCNode(cnode, prim);
  }
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node != nullptr) {
    const auto &value = value_node->value();
    if (value == nullptr) {
      return nullptr;
    }
    // Call value node converter.
    return ConvertMakeListValueNode(value_node, value);
  }
  return nullptr;
}

// AbstractRowTensor --> AbstractTuple.
static AbstractBasePtr ConvertToAbstractTuple(const AbstractBasePtr &abs, size_t depth) {
  if (depth > kMaxSeqRecursiveDepth) {
    MS_LOG(EXCEPTION) << "List, tuple and dict nesting is not allowed more than " << kMaxSeqRecursiveDepth
                      << " levels.";
  }
  // Convert RowTensor in AbstractSequence to AbstractTuple.
  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  if (abs_seq != nullptr) {
    if (abs_seq->dynamic_len() && abs_seq->isa<abstract::AbstractList>()) {
      auto converted_abs_tuple =
        std::make_shared<abstract::AbstractTuple>(abs_seq->elements(), abs_seq->sequence_nodes());
      converted_abs_tuple->set_dynamic_len(true);
      converted_abs_tuple->set_dynamic_len_element_abs(abs_seq->dynamic_len_element_abs());
      return converted_abs_tuple;
    }
    const auto &seq_elements = abs_seq->elements();
    // First we check if elements should be converted,
    // changed_elements maps old element to new element.
    mindspore::HashMap<AbstractBasePtr, AbstractBasePtr> changed_elements;
    for (const auto &element : seq_elements) {
      auto new_element = ConvertToAbstractTuple(element, depth + 1);
      if (new_element != nullptr) {
        (void)changed_elements.emplace(element, new_element);
      }
    }
    if (changed_elements.empty()) {
      if (abs->isa<abstract::AbstractTuple>()) {
        // If no elements changed and it is an AbstractTuple, do not convert.
        return nullptr;
      }
      // If no elements changed but it is not an AbstractTuple, convert it by copy elements.
      return std::make_shared<abstract::AbstractTuple>(seq_elements);
    }
    // Make new abstract sequence.
    std::vector<AbstractBasePtr> elements;
    elements.reserve(seq_elements.size());
    for (const auto &element : seq_elements) {
      auto iter = changed_elements.find(element);
      if (iter != changed_elements.end()) {
        (void)elements.emplace_back(iter->second);
      } else {
        (void)elements.emplace_back(element);
      }
    }
    if (abs_seq->isa<abstract::AbstractList>()) {
      return std::make_shared<abstract::AbstractList>(std::move(elements));
    }
    return std::make_shared<abstract::AbstractTuple>(std::move(elements));
  }
  // AbstractRowTensor --> AbstractTuple.
  auto abs_row_tensor = abs->cast<std::shared_ptr<abstract::AbstractRowTensor>>();
  if (abs_row_tensor != nullptr) {
    std::vector<AbstractBasePtr> elements{abs_row_tensor->indices(), abs_row_tensor->values(),
                                          abs_row_tensor->dense_shape()};
    return std::make_shared<abstract::AbstractTuple>(std::move(elements));
  }
  return nullptr;
}

static AnfNodePtr MakeListNodeRewrite(const AnfNodePtr &node) {
  auto new_node = ConvertMakeListNode(node);
  if (IsPrimitiveCNode(new_node, prim::kPrimPyExecute)) {
    return new_node;
  }
  if (new_node != nullptr) {
    new_node->set_abstract(node->abstract());
  }
  return new_node;
}

static void UpdateMakeListAbstracts(const FuncGraphPtr &func_graph) {
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN_VOID(manager);
  const auto &nodes = manager->all_nodes();
  for (const auto &node : nodes) {
    const auto &abs = node->abstract();
    if (abs == nullptr) {
      continue;
    }
    bool is_interpret_dict = false;
    // Do not convert the abstract of Interpret node(AbstractDictionary) to AbstractSequence.
    if (abs->isa<abstract::AbstractDictionary>()) {
      abstract::AbstractDictionaryPtr abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
      auto &dict_elements = abs_dict->elements();
      for (auto &element : dict_elements) {
        TypePtr type = element.second->GetTypeTrack();
        MS_EXCEPTION_IF_NULL(type);
        auto value = element.second->BuildValue();
        MS_EXCEPTION_IF_NULL(value);
        if (type->type_id() == kMetaTypeExternal && value->isa<parse::InterpretedObject>()) {
          is_interpret_dict = true;
          break;
        }
      }
    }
    if (is_interpret_dict) {
      continue;
    }
    // Call abstract converter.
    auto new_abs = ConvertToAbstractTuple(abs, 0);
    if (new_abs != nullptr) {
      node->set_abstract(new_abs);
    }
  }
}

STATUS AclPassImpl::MakeListToMakeTuple(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo;
  auto add_todo = [&seen, &todo](const AnfNodePtr &node) {
    if (node != nullptr && node->seen_ != seen) {
      (void)todo.emplace_back(node);
    }
  };
  (void)todo.emplace_back(func_graph->return_node());
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is nullptr.");
  auto &all_nodes = manager->all_nodes();
  while (!todo.empty()) {
    AnfNodePtr node = std::move(todo.front());
    todo.pop_front();
    if (node == nullptr || node->seen_ == seen || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;
    auto cnode = node->cast_ptr<CNode>();
    if (cnode != nullptr) {
      for (auto &input : cnode->inputs()) {
        add_todo(input);
      }
    } else {
      auto fg = GetValuePtr<FuncGraph>(node);
      if (fg != nullptr) {
        add_todo(fg->return_node());
      }
    }
    TraceGuard trace_guard(std::make_shared<TraceOpt>(node->debug_info()));
    ScopeGuard scope_guard(node->scope());
    auto new_node = MakeListNodeRewrite(node);
    if (new_node != nullptr) {
      (void)manager->Replace(node, new_node);
      changed = true;
    }
  }
  if (changed) {
    UpdateMakeListAbstracts(func_graph);
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::AdjustInvalidCnodeName(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "AdjustInvalidCnodeName start.";
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_ERROR, "func_graph is nullptr.");
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(func_graph, &all_func_graphs);
  size_t index = 0;
  for (auto graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto name = cnode->fullname_with_scope();
      if (name.empty()) {
        name = "cnode_" + std::to_string(index++);
      }
      while (!name.empty() && name[0] == '/') {
        name = name.substr(1, name.size());
        if (name.empty()) {
          name = "cnode_" + std::to_string(index++);
        }
      }
      cnode->set_fullname_with_scope(name);
    }
  }
  return RET_OK;
}

STATUS AclPassImpl::PreProcGraph(const FuncGraphPtr &func_graph) {
  auto status = AdjustInvalidCnodeName(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "AdjustInvalidCnodeName failed.";
    return RET_ERROR;
  }
  if (CommonPass(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Common pass failed.";
    return lite::RET_ERROR;
  }
  static std::map<converter::FmkType, std::function<STATUS(const FuncGraphPtr &, bool)>> fmk_proc_func = {
    {converter::kFmkTypeMs, PreProcForMindIr},   {converter::kFmkTypeTf, PreProcForTF},
    {converter::kFmkTypeCaffe, PreProcForCaffe}, {converter::kFmkTypeOnnx, PreProcForOnnx},
    {converter::kFmkTypeTflite, PreProcForTF},
  };
  if (fmk_proc_func.find(fmk_type_) != fmk_proc_func.end()) {
    auto func = fmk_proc_func.at(fmk_type_);
    if (func(func_graph, user_options_cfg_.offline) != lite::RET_OK) {
      MS_LOG(ERROR) << "Pre proc failed, fmk " << fmk_type_;
      return lite::RET_ERROR;
    }
  } else {
    MS_LOG(WARNING) << "Not support fmk type " << fmk_type_;
  }
  MS_LOG(DEBUG) << "Pre proc graph success.";
  return lite::RET_OK;
}

STATUS AclPassImpl::PostProcGraph(const FuncGraphPtr &func_graph) {
  if (lite::acl::DelRedundantParameter(func_graph) != RET_SUCCESS) {
    MS_LOG(ERROR) << "Delete redundant parameters failed.";
    return lite::RET_ERROR;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is nullptr.");
  manager->Reset();
  if (fmk_type_ == converter::kFmkTypeTf) {
    MS_LOG(DEBUG) << "Tf no need to change to nhwc format.";
    return lite::RET_OK;
  }
  if (!lite::RunOptimizerPass(func_graph, {kToNHWCFormatPass})) {
    MS_LOG(ERROR) << "To NHWC Format failed.";
    return lite::RET_ERROR;
  }

  MS_LOG(DEBUG) << "Post proc graph success.";
  return lite::RET_OK;
}

std::string AclPassImpl::AdjustCnodeName(const PrimitivePtr &prim) {
  MS_CHECK_TRUE_MSG(prim != nullptr, "", "prim is nullptr.");
  std::string name = prim->name();
  if (kAdjustCnodeName.find(name) != kAdjustCnodeName.end()) {
    auto val_ptr = prim->GetAttr(ops::kOriginalOpName);
    if (val_ptr != nullptr) {
      auto origin_name = GetValue<std::string>(val_ptr);
      MS_LOG(DEBUG) << "Change name " << name << " to " << origin_name;
      name = origin_name;
    }
  }
  return name;
}

STATUS AclPassImpl::RunPrimitiveMapper(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Deparser graph start.";
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_ERROR, "func_graph is nullptr.");
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr.");
      auto prim = GetCNodePrimitive(cnode);
      CHECK_NULL_RETURN(prim);
      std::string name = AdjustCnodeName(prim);
      auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(name);
      if (mapper == nullptr) {
        MS_LOG(DEBUG) << "Name: " << name << " not need to mapper.";
        continue;
      }
      MS_LOG(INFO) << "Deparser cnode: " << name;
      auto status = mapper->Mapper(cnode);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Deparser primitive failed, cnode " << name;
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::MapperForOrgMindIR(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Deparser graph for MindIR model start.";
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_ERROR, "func_graph is nullptr.");
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(func_graph, &all_func_graphs);

  std::set<std::string> mindir_mapper = {ops::kNameTranspose, ops::kNameStandardNormal};
  for (auto graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr.");
      auto prim = GetCNodePrimitive(cnode);
      CHECK_NULL_RETURN(prim);
      std::string name = AdjustCnodeName(prim);
      if (!mindir_mapper.count(name)) {
        continue;
      }
      auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(name);
      if (mapper == nullptr) {
        MS_LOG(DEBUG) << "Name: " << name << " not need to mapper.";
        continue;
      }
      MS_LOG(INFO) << "Deparser cnode: " << name;
      auto status = mapper->Mapper(cnode);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Deparser primitive failed, cnode " << name;
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::DeparseGraph(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  if (!(is_ptq_quant_ || param_->ascendQuantParam.mode != lite::quant::NONE) && fmk_type_ == converter::kFmkTypeMs) {
    MapperForOrgMindIR(func_graph);
    return lite::RET_OK;
  }
  if (RunPrimitiveMapper(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Run mapper primitive failed.";
    return lite::RET_ERROR;
  }
  if (lite::AdapteSpatialNode(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adapter spatial node failed.";
    return lite::RET_ERROR;
  }
  if (lite::acl::DelRedundantParameter(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Delete redundant parameter failed.";
    return lite::RET_OK;
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::SetGraphInputShape(const FuncGraphPtr &func_graph) {
  if (options_ == nullptr || func_graph == nullptr) {
    MS_LOG(ERROR) << "Options or func_graph cannot be nullptr";
    return lite::RET_ERROR;
  }
  const auto &input_shape = options_->GetInputShape();
  if (input_shape.empty()) {
    MS_LOG(INFO) << "Input shape option is empty";
    return lite::RET_OK;
  }
  auto input_shape_strs = lite::StrSplit(input_shape, ";");
  auto inputs = func_graph->get_inputs();
  if (inputs.size() != input_shape_strs.size()) {
    MS_LOG(ERROR) << "FuncGraph input size " << inputs.size() << " != input size in input_shape "
                  << input_shape_strs.size() << ", input_shape " << input_shape;
    return lite::RET_ERROR;
  }
  std::map<std::string, ShapeVector> input_shapes;
  for (auto &input_shape_str : input_shape_strs) {
    auto split_pos = input_shape_str.rfind(":");
    if (split_pos == std::string::npos) {
      MS_LOG(ERROR) << "The input_shape should be in format of name:shape;name:shape, but got [" << input_shape_str
                    << "]";
      return lite::RET_ERROR;
    }
    std::string name = input_shape_str.substr(0, split_pos);
    std::string shape_str = input_shape_str.substr(split_pos + 1);
    ShapeVector shape;
    if (!lite::ParseShapeStr(shape_str, &shape)) {
      MS_LOG(ERROR) << "Invalid input shape dims: " << shape_str << ", input_shape: " << input_shape;
      return false;
    }
    input_shapes[name] = shape;
  }
  for (auto node : inputs) {
    CHECK_NULL_RETURN(node);
    auto para = node->cast<ParameterPtr>();
    CHECK_NULL_RETURN(para);
    auto it = input_shapes.find(para->name());
    if (it == input_shapes.end()) {
      MS_LOG(ERROR) << "Failed to find input " << para->name() << " in input_shape " << input_shape;
      return lite::RET_ERROR;
    }
    auto abstract = para->abstract();
    CHECK_NULL_RETURN(abstract);
    abstract->set_shape(std::make_shared<abstract::Shape>(it->second));
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::ConvertGraphToOm(const FuncGraphPtr &func_graph, Buffer *om_data) {
  if (om_data == nullptr) {
    MS_LOG(ERROR) << "Om data is nullptr.";
    return lite::RET_ERROR;
  }
  auto ret = SetAclModelOptions(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Set acl model options error!";
    return lite::RET_ERROR;
  }
  ret = SetGraphInputShape(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Failed to set graph input shape";
    return lite::RET_ERROR;
  }
  // call interface of cloud
  ModelConverter model_converter;
  model_converter.set_options(options_);
  *om_data = model_converter.LoadMindIR(func_graph);
  if (om_data->Data() == nullptr || om_data->DataSize() == 0) {
    MS_LOG(ERROR) << "Model converter load mindir failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::SetAclModelOptions(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Set acl model options start.";
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_ERROR, "func_graph is nullptr.");
  auto model_context = user_options_cfg_.AsModelContext(param_->provider);
  CHECK_NULL_RETURN(model_context);
  options_ = std::make_shared<AclModelOptions>(model_context);
  CHECK_NULL_RETURN(options_);
  auto inputs = func_graph->get_inputs();
  if (user_options_cfg_.input_shape.empty()) {
    std::vector<std::string> input_names;
    for (auto node : inputs) {
      CHECK_NULL_RETURN(node);
      auto para = node->cast<ParameterPtr>();
      CHECK_NULL_RETURN(para);
      input_names.push_back(para->name());
    }
    options_->RenameInput(input_names);
  }
  auto pos = user_options_cfg_.om_file_path.find_last_of('/');
  std::string save_path = "./";
  if (pos != std::string::npos) {
    save_path = user_options_cfg_.om_file_path.substr(0, pos + 1);
  }
  save_path = lite::RealPath(save_path.c_str());
  if (save_path.empty()) {
    return lite::RET_ERROR;
  }
  options_->SetOmFilePath(user_options_cfg_.om_file_path);
  options_->SetDumpModelName(user_options_cfg_.dump_model_name);
  options_->SetAoeMode(user_options_cfg_.aoe_mode);
  options_->SetInitOptionsMap(user_options_cfg_.init_options_map);
  options_->SetBuildOptionsMap(user_options_cfg_.build_options_map);
  MS_LOG(INFO) << "Set acl model options success.";
  return lite::RET_OK;
}

ParameterPtr AclPassImpl::CreateOmParameter(const FuncGraphPtr &func_graph, const Buffer &om_data) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, nullptr, "func_graph is nullptr.");
  ParameterPtr om_parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(om_parameter != nullptr, nullptr, "om_parameter is nullptr.");
  om_parameter->set_name("ACL_om_data");

  auto type_ptr = TypeIdToType(kNumberTypeUInt8);
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, nullptr, "type_ptr is nullptr.");
  ShapeVector shape_vector = {static_cast<int64_t>(om_data.DataSize())};
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, nullptr, "abstract_tensor is nullptr.");
  om_parameter->set_abstract(abstract_tensor);

  auto param_value =
    std::make_shared<tensor::Tensor>(kNumberTypeUInt8, ShapeVector({static_cast<int64_t>(om_data.DataSize())}));
  MS_CHECK_TRUE_MSG(param_value != nullptr, nullptr, "param_value is nullptr.");
  auto tensor_data = param_value->data_c();
  MS_CHECK_TRUE_MSG(tensor_data != nullptr, nullptr, "New Tensor failed.");
  if (param_value->Size() < om_data.DataSize()) {
    MS_LOG(ERROR) << "Dst buff size  " << param_value->Size() << " should be greater than src buff size "
                  << om_data.DataSize();
    return nullptr;
  }
  if (common::huge_memcpy(reinterpret_cast<uint8_t *>(tensor_data), param_value->Size(),
                          reinterpret_cast<const uint8_t *>(om_data.Data()), om_data.DataSize()) != EOK) {
    MS_LOG(ERROR) << "Memcpy om data failed.";
    return nullptr;
  }
  om_parameter->set_default_param(param_value);
  return om_parameter;
}

// now build the whole graph, not split
STATUS AclPassImpl::BuildGraph(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_ERROR, "func_graph is nullptr.");
  Buffer om_data;
  if (ConvertGraphToOm(func_graph, &om_data) != lite::RET_OK) {
    MS_LOG(ERROR) << "Convert graph  to om failed.";
    return lite::RET_ERROR;
  }
  om_parameter_ = CreateOmParameter(func_graph, om_data);
  MS_CHECK_TRUE_MSG(om_parameter_ != nullptr, lite::RET_ERROR, "Convert graph  to om failed.");
  MS_LOG(DEBUG) << "Build graph success.";
  return lite::RET_OK;
}

STATUS AclPassImpl::TraceOutput(const AnfNodePtr &node) {
  static size_t iter = 0;
  CHECK_NULL_RETURN(node);
  if (node->isa<ValueNode>() || utils::isa<ParameterPtr>(node)) {
    MS_LOG(INFO) << "Name of graph output value node is : " << node->fullname_with_scope();
    graph_output_dims_.emplace_back(std::vector<int64_t>());
    graph_outputs_.emplace_back(node);
    tuple_idx_.emplace_back(0);
    return lite::RET_OK;
  }
  AnfNodePtr cur_node = node;
  CNodePtr pre_node = nullptr;
  while (cur_node->isa<CNode>() && IsPrimitiveCNode(cur_node, prim::kPrimTupleGetItem)) {
    auto tmp = cur_node->cast<CNodePtr>();
    CHECK_NULL_RETURN(tmp);
    pre_node = tmp;
    cur_node = tmp->input(kTupleGetItemFirstInputIdx);
    CHECK_NULL_RETURN(cur_node);
  }
  auto cnode = cur_node->cast<CNodePtr>();
  CHECK_NULL_RETURN(cnode);
  std::string name = lite::acl::GetCNodeTargetFuncName(cnode);
  iter++;
  MS_LOG(INFO) << "Func name of cnode " << name << " ,trace iter: " << iter;
  if (name == kMakeTuple) {
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      if (TraceOutput(cnode->input(i)) != lite::RET_OK) {
        MS_LOG(ERROR) << "The input[ " << i << "]"
                      << " trace output failed, name: " << name;
        return lite::RET_ERROR;
      }
    }
  } else if (name == prim::kPrimDepend->name()) {
    if (cnode->inputs().size() < kDependInputNum) {
      MS_LOG(ERROR) << "Length of inputs is " << cnode->inputs().size() << ", which is less than three.";
      return lite::RET_ERROR;
    }
    if (TraceOutput(cnode->input(kDependFirstInputIdx)) != lite::RET_OK) {
      MS_LOG(ERROR) << "Depend node trace output failed.";
      return lite::RET_ERROR;
    }
  } else {
    MS_LOG(INFO) << "Name of graph output node is " << cnode->fullname_with_scope();
    std::vector<int64_t> dims;
    STATUS ret;
    size_t idx = 0;
    if (pre_node != nullptr && IsPrimitiveCNode(pre_node, prim::kPrimTupleGetItem)) {
      ret = lite::acl::GetShapeVectorFromCNode(pre_node, &dims);
      idx = mindspore::opt::GetTupleGetItemOutIndex(pre_node);
    } else {
      ret = lite::acl::GetShapeVectorFromCNode(cnode, &dims);
    }
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Get node shape failed.";
      return lite::RET_ERROR;
    }
    graph_output_dims_.emplace_back(dims);
    graph_outputs_.emplace_back(cnode);
    tuple_idx_.emplace_back(idx);
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::GetFuncGraphOutputInfo(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_ERROR, "func_graph is nullptr.");
  AnfNodePtr return_input = func_graph->output();
  CHECK_NULL_RETURN(return_input);
  if (TraceOutput(return_input) != lite::RET_OK) {
    MS_LOG(ERROR) << "Trace output failed.";
    return lite::RET_ERROR;
  }
  if (graph_outputs_.empty() || graph_outputs_.size() != graph_output_dims_.size()) {
    MS_LOG(ERROR) << "Graph output size is error, num size: " << graph_outputs_.size()
                  << " dim size: " << graph_output_dims_.size();
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

STATUS AclPassImpl::SetMultiOutputs(const CNodePtr &new_cnode, std::vector<TypeId> data_type) {
  MS_CHECK_TRUE_MSG(new_cnode != nullptr, lite::RET_ERROR, "new_cnode is nullptr.");
  AbstractBasePtrList abstract_list;
  for (size_t j = 0; j < graph_outputs_.size(); j++) {
    auto abstract_tensor = lite::CreateTensorAbstract(graph_output_dims_[j], data_type[j]);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract tensor is nullptr for output " << j;
      return lite::RET_ERROR;
    }
    abstract_list.emplace_back(abstract_tensor);
  }
  new_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return lite::RET_OK;
}

STATUS AclPassImpl::SetCustomOutputs(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  STATUS ret = GetFuncGraphOutputInfo(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Get output info of graph failed.";
    return lite::RET_ERROR;
  }
  TypeId type;
  if (graph_outputs_.size() == 1) {
    type = lite::acl::GetTypeFromNode(graph_outputs_[0]);
    auto abstract_tensor = lite::CreateTensorAbstract(graph_output_dims_[0], type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract_tensor is nullptr.";
      return lite::RET_ERROR;
    }
    custom_node->set_abstract(abstract_tensor);
    return lite::RET_OK;
  }
  std::vector<TypeId> types;
  for (size_t i = 0; i < graph_outputs_.size(); i++) {
    type = lite::acl::GetTypeFromNode(graph_outputs_[i], tuple_idx_[i]);
    types.emplace_back(type);
  }
  if (SetMultiOutputs(custom_node, types) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set multi graph output failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

void AclPassImpl::SetCustomAttrs(const std::shared_ptr<ops::Custom> &prim) {
  std::string output_dim_str;
  for (const auto &item : graph_output_dims_) {
    output_dim_str += std::to_string(item.size()) + ",";
    for (const auto &val : item) {
      output_dim_str += std::to_string(val) + ",";
    }
  }
  std::vector<uint8_t> output_dim_char(output_dim_str.begin(), output_dim_str.end());
  std::map<std::string, std::vector<uint8_t>> attrs = {{lite::acl::kOutputShapes, output_dim_char}};
  prim->set_attr(attrs);
  prim->AddAttr(kFuncType, api::MakeValue<std::string>("acl_build"));
  prim->AddAttr(kUniqueName, api::MakeValue<std::string>("CustomAscend"));
}

CNodePtr AclPassImpl::CreateCustomNode(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, nullptr, "func_graph is nullptr.");
  auto prim = std::make_shared<mindspore::ops::Custom>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "New custom op failed.");
  prim->set_type(kCustomPrimTypeACL);
  auto prim_c = prim->GetPrim();
  auto graph_input = func_graph->get_inputs();
  CNodePtr custom_node = func_graph->NewCNode(prim_c, graph_input);
  MS_CHECK_TRUE_MSG(custom_node != nullptr, nullptr, "Custom cnode failed.");
  custom_node->set_fullname_with_scope(kCustomNodeName);
  custom_node->add_input(om_parameter_);

  if (SetCustomOutputs(func_graph, custom_node) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set custom outputs failed.";
    return nullptr;
  }
  SetCustomAttrs(prim);
  return custom_node;
}

CNodePtr AclPassImpl::CreateMakeTupleGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  std::vector<CNodePtr> node_list;
  for (size_t j = 0; j < graph_outputs_.size(); ++j) {
    auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
    if (tuple_get_item_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "New TupleGetItem failed for output " << j;
      return nullptr;
    }
    auto tuple_get_item_prim_ptr_c = tuple_get_item_prim_ptr->GetPrim();
    auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr_c);
    MS_CHECK_TRUE_MSG(tuple_get_item_prim != nullptr, nullptr, "item_prim is nullptr.");
    auto get_item_value = NewValueNode(MakeValue<int64_t>(j));
    MS_CHECK_TRUE_MSG(get_item_value != nullptr, nullptr, "item_value is nullptr.");
    AnfNodePtrList inputs{tuple_get_item_prim, custom_node, get_item_value};
    CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
    if (get_item_cnode == nullptr) {
      MS_LOG(ERROR) << "New get item cnode failed for output " << j;
      return nullptr;
    }
    get_item_cnode->set_fullname_with_scope(custom_node->fullname_with_scope() + "_getitem_" + std::to_string(j));
    node_list.emplace_back(get_item_cnode);
  }
  auto make_tuple_val_node = NewValueNode(prim::kPrimMakeTuple);
  MS_CHECK_TRUE_MSG(make_tuple_val_node != nullptr, nullptr, "New make tuple val node failed.");
  AnfNodePtrList new_inputs = {make_tuple_val_node};
  new_inputs.insert(new_inputs.end(), node_list.begin(), node_list.end());
  auto make_tuple_cnode = func_graph->NewCNode(new_inputs);
  MS_CHECK_TRUE_MSG(make_tuple_cnode != nullptr, nullptr, "New make tuple cnode failed.");
  return make_tuple_cnode;
}

STATUS AclPassImpl::ModifyGraphByCustomNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                            const CNodePtr &custom_node) {
  AnfNodePtr return_input = func_graph->output();
  MS_CHECK_TRUE_MSG(return_input != nullptr, lite::RET_ERROR, "return input is nullptr.");
  if (graph_outputs_.size() == 1) {
    if (!manager->Replace(return_input, custom_node)) {
      MS_LOG(ERROR) << "Replace node failed.";
      return lite::RET_ERROR;
    }
  } else {
    auto make_tuple_node = CreateMakeTupleGraphOutput(func_graph, custom_node);
    MS_CHECK_TRUE_MSG(make_tuple_node != nullptr, lite::RET_ERROR, "Create make tuple cnode failed.");
    if (!manager->Replace(return_input, make_tuple_node)) {
      MS_LOG(ERROR) << "Replace node failed for outputs of graph.";
      return lite::RET_ERROR;
    }
  }
  MS_LOG(DEBUG) << "Modify graph by custom node success.";
  return lite::RET_OK;
}

STATUS AclPassImpl::PreQuantization(const FuncGraphPtr &func_graph) {
  auto redundant_op_remove_pass = std::make_shared<mindspore::opt::RemoveRedundantOpPass>(false, true, true);
  if (!redundant_op_remove_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "Run remove redundant op failed";
    return RET_ERROR;
  }
  auto ret = lite::quant::MarkOriginDataType(func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "mark origin data type failed.";
    return ret;
  }
  if (param_->commonQuantParam.quant_type == lite::quant::QUANT_ALL) {
    auto value = func_graph->get_attr(ops::kFormat);
    if (value == nullptr) {
      auto unify_format = std::make_shared<lite::UnifyFormatToNHWC>(fmk_type_, false, param_->save_type);
      CHECK_NULL_RETURN(unify_format);
      if (!unify_format->Run(func_graph)) {
        MS_LOG(ERROR) << "Run insert transpose failed.";
        return lite::RET_ERROR;
      }
      if (!lite::RunOptimizerPass(func_graph, {"DecreaseTransposeAlgo"})) {
        MS_LOG(ERROR) << "Run ToNCHWFormat pass failed";
        return lite::RET_ERROR;
      }
    }
    if (value == nullptr || GetValue<int64_t>(value) == mindspore::NCHW) {
      if (!lite::RunOptimizerPass(func_graph, {kToNHWCFormatPass, "DecreaseTransposeAlgo"})) {
        MS_LOG(ERROR) << "Run ToNCHWFormat pass failed";
        return lite::RET_ERROR;
      }
    }
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    CHECK_NULL_RETURN(optimizer);
    auto fusion_pm = std::make_shared<opt::LitePassManager>("anf fusion pass manager", false);
    CHECK_NULL_RETURN(fusion_pm);
    std::vector<opt::PassPtr> fusions{
      std::make_shared<opt::ClipConvertActivationPass>(true),
      std::make_shared<opt::BatchNormToScaleFusion>(),
      std::make_shared<opt::ConvBiasaddFusion>(),
      std::make_shared<opt::ConvBatchNormFusion>(param_->fmk_type),
      std::make_shared<opt::ConvScaleFusion>(param_->fmk_type),
      std::make_shared<opt::TransposeFusion>(),
    };
    for (size_t index = 0; index < fusions.size(); index++) {
      auto pass_ptr = fusions.at(index);
      fusion_pm->AddPass(pass_ptr);
    }
    optimizer->AddPassManager(fusion_pm);
    if (optimizer->Optimize(func_graph) == nullptr) {
      MS_LOG(ERROR) << "run op fusion failed.";
      return RET_ERROR;
    }
    lite::quant::InsertQuantNodeManager quant_manager;
    if (quant_manager.AdjustTransposeNodeForMatMul(func_graph) != lite::RET_OK) {
      MS_LOG(ERROR) << "InsertTransposeNodeForMatMul failed.";
      return lite::RET_ERROR;
    }
    if (!lite::RunOptimizerPass(func_graph, {kInferShapePass})) {
      MS_LOG(ERROR) << "Infer shape pass failed.";
      return lite::RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS AclPassImpl::PostQuantization(const FuncGraphPtr &func_graph) {
  if (param_->commonQuantParam.quant_type != lite::quant::QUANT_ALL) {
    return lite::RET_OK;
  }
  if (!lite::RunOptimizerPass(func_graph, {kInferShapePass})) {
    MS_LOG(ERROR) << "Infer shape pass failed.";
    return lite::RET_ERROR;
  }
  // NCHW format and remove transpose
  auto value = func_graph->get_attr(ops::kFormat);
  if (value == nullptr || GetValue<int64_t>(value) == mindspore::NHWC) {
    if (!lite::RunOptimizerPass(func_graph, {kToNCHWFormatPass, "DecreaseTransposeAlgo"})) {
      MS_LOG(ERROR) << "To nchw format failed.";
      return lite::RET_ERROR;
    }
  }
  if (RemoveQuantDtypeCast(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "Remove QuantDtypeCast failed.";
    return lite::RET_ERROR;
  }
  // Insert QuantDtypeCast for conv & fc
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    lite::quant::QuantType curr_quant_type;
    if (GetQuantTypeNew(cnode, &curr_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type != lite::quant::QUANT_ALL) {
      continue;
    }
    lite::quant::InsertQuantNodeManager insert_node_manager;
    auto ret = insert_node_manager.InsertAscendDeQuantNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert AscendDeQuant node failed, cnode name: " << cnode->fullname_with_scope();
      return ret;
    }
    ret = insert_node_manager.InsertAscendQuantNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert AscendQuant node failed, cnode name: " << cnode->fullname_with_scope();
      return ret;
    }
    ret = lite::UpdateDataType(cnode, kNumberTypeInt32);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Update datatype failed, cnode name: " << cnode->fullname_with_scope();
      return ret;
    }
  }
  return RET_OK;
}

STATUS AclPassImpl::Quantization(const FuncGraphPtr &func_graph) {
  auto ret = PreQuantization(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Pre quantization execution failed.";
    return ret;
  }
  lite::quant::QuantizationOptimizer quant_optimizer(param_);
  if (param_->commonQuantParam.quant_type == lite::quant::QUANT_ALL) {
    ret = quant_optimizer.DoFullQuant(func_graph, param_);
  } else if (param_->commonQuantParam.quant_type == lite::quant::QUANT_WEIGHT) {
    ret = quant_optimizer.DoWeightQuant(func_graph, param_);
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Ascend full quant execution failed.";
    return ret;
  }
  ret = PostQuantization(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Post quantization execution failed..";
    return ret;
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::RemoveQuantDtypeCast(const FuncGraphPtr &func_graph) {
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    CHECK_NULL_RETURN(cnode_primitive);
    if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
      if (cnode_primitive->HasAttr(lite::quant::kQuantParam)) {
        auto pre_node = cnode->input(1);
        if (pre_node->isa<CNode>()) {
          auto pre_cnode = pre_node->cast<mindspore::CNodePtr>();
          auto pre_cnode_primitive = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
          CHECK_NULL_RETURN(pre_cnode_primitive);
          MS_LOG(INFO) << "Copy quant param from cnode: " << cnode->fullname_with_scope();
          pre_cnode_primitive->AddAttr(lite::quant::kQuantParam, cnode_primitive->GetAttr(lite::quant::kQuantParam));
        } else if (lite::IsGraphInput(pre_node)) {
          auto node_users = manager->node_users()[cnode];
          for (auto &node_user : node_users) {
            manager->SetEdge(node_user.first, node_user.second, cnode->input(1));
          }
          for (auto &node_user : node_users) {
            auto post_cnode = node_user.first->cast<CNodePtr>();
            CHECK_NULL_RETURN(post_cnode);
            auto post_cnode_primitive = GetValueNode<PrimitivePtr>(post_cnode->input(0));
            auto quantization_param_value = cnode_primitive->GetAttr(lite::quant::kQuantParam);
            CHECK_NULL_RETURN(quantization_param_value);
            auto quantization_param_list = GetValue<std::vector<QuantizationParamPtr>>(quantization_param_value);
            if (!quantization_param_list.empty()) {
              MS_LOG(INFO) << "Copy quant param from cnode: " << cnode->fullname_with_scope();
              post_cnode_primitive->AddAttr(lite::quant::kGraphInputQuantParam, quantization_param_list.front());
            }
          }
        }
      }
      auto node_users = manager->node_users()[cnode];
      for (auto &node_user : node_users) {
        manager->SetEdge(node_user.first, node_user.second, cnode->input(1));
      }
    }
    cnode_primitive->DelAttr("io_format");
  }
  return RET_OK;
}

bool AclPassImpl::Run(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Acl pass run start.";
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  auto manager = Manage(func_graph, true);
  MS_CHECK_TRUE_MSG(manager != nullptr, false, "manager is nullptr.");

  if (!user_options_cfg_.custom_opp_path.empty()) {
    // if set custom_opp_path config, first install custom ops to cann
    AclCustomOppInstaller::InstallCustomOpp(user_options_cfg_.custom_opp_path, "");
  }

  if (PreProcGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Pre proc graph failed.";
    return false;
  }
  if (param_->ascendQuantParam.mode != lite::quant::AscendQuantMode::NONE) {
    auto ascend_distribute_fake_quant_transform = lite::quant::AscendDistributeFakeQuantTransform(func_graph, param_);
    auto status = ascend_distribute_fake_quant_transform.Transform();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do AscendDistributeFakeQuantTransform failed.";
      return false;
    }
  }
  if (is_ptq_quant_ && Quantization(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Quantization failed.";
    return false;
  }

  if (DeparseGraph(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Deparse graph failed.";
    return false;
  }

  if (param_->ascendQuantParam.mode == lite::quant::AscendQuantMode::GE) {
    return true;
  }
#ifdef MSLITE_ENABLE_GRAPH_KERNEL
  if (param_->device.find("Ascend") != std::string::npos) {
    if (GraphKernelOptimize(func_graph, param_) != lite::RET_OK) {
      MS_LOG(ERROR) << "Run graphkernel optimization failed.";
      return false;
    }
  }
#endif

  if (BuildGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Build graph failed.";
    return false;
  }

  custom_node_ = CreateCustomNode(func_graph);
  MS_CHECK_TRUE_MSG(custom_node_ != nullptr, false, "Create custom node failed.");
  // prepare graph for export create
  if (ModifyGraphByCustomNode(func_graph, manager, custom_node_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Modify func graph by custom failed.";
    return false;
  }

  if (PostProcGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Post proc graph failed.";
    return false;
  }
  MS_LOG(INFO) << "Acl pass run end.";
  return true;
}
}  //  namespace opt
}  // namespace mindspore
