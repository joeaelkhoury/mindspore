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

#include "frontend/optimizer/fallback_rewriter.h"
#include <iterator>
#include <string>
#include <algorithm>
#include <functional>
#include <utility>
#include <memory>
#include <unordered_map>
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sparse_tensor_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "pipeline/jit/debug/trace.h"
#include "pipeline/jit/action.h"
#include "pipeline/jit/parse/parse_base.h"
#include "frontend/optimizer/opt.h"
#include "frontend/operator/composite/composite.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "pipeline/jit/fallback.h"
#include "pipeline/jit/parse/resolve.h"
#include "utils/hash_map.h"
#include "utils/anf_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractBasePtr;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractDictionaryPtr;
using mindspore::abstract::AbstractElementPair;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractListPtr;
using mindspore::abstract::AbstractRowTensor;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractSequencePtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using ClassTypePtr = std::shared_ptr<parse::ClassType>;

constexpr auto kInternalDictSelfStr = "__internal_dict_self__";
constexpr auto kInternalDictKeyStr = "__internal_dict_key__";
constexpr auto kInternalDictValueStr = "__internal_dict_value__";

namespace {
static constexpr size_t kMaxSeqRecursiveDepth = 6;
void CheckInputsSize(const CNodePtr &cnode, size_t expect_size) {
  if (cnode->size() != expect_size) {
    std::string op_name = GetCNodeFuncName(cnode);
    MS_LOG(INTERNAL_EXCEPTION) << op_name << " should have " << expect_size << " inputs, but got " << cnode->size();
  }
}

template <typename T>
std::shared_ptr<T> GetAbstract(const AnfNodePtr &node) {
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  return dyn_cast<T>(abs);
}

bool CheckContainsDict(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<AbstractDictionary>()) {
    return true;
  }
  if (abs->isa<AbstractSequence>()) {
    auto abs_seq = abs->cast<AbstractSequencePtr>();
    const auto &elements = abs_seq->elements();
    if (std::any_of(elements.begin(), elements.end(),
                    [](const AbstractBasePtr &element) { return CheckContainsDict(element); })) {
      return true;
    }
  }
  return false;
}

// ===========================================================================
// BaseRewriter provides a common framework for data struct simplify.
// ===========================================================================
class BaseRewriter : protected SimpleRewriter {
 public:
  BaseRewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager)
      : SimpleRewriter(root_graph, manager) {}
  ~BaseRewriter() override = default;

  bool need_renormalized() const { return need_renormalized_; }

  void set_need_renormalized(bool need_renormalized) { need_renormalized_ = need_renormalized; }

  virtual bool Execute() {
    bool changed = Run();
    if (changed) {
      UpdateAbstracts();
    }
    return changed;
  }

 protected:
  virtual AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode) = 0;
  virtual AnfNodePtr ConvertValueNode(const ValueNodePtr &value_node, const ValuePtr &value) = 0;
  virtual AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) = 0;

  AnfNodePtr NodeRewrite(const AnfNodePtr &node) override {
    auto new_node = ConvertNode(node);
    if (IsPrimitiveCNode(new_node, prim::kPrimPyExecute)) {
      need_renormalized_ = true;
      return new_node;
    }
    if (new_node != nullptr) {
      new_node->set_abstract(node->abstract());
    }
    return new_node;
  }

  AnfNodePtr ConvertNode(const AnfNodePtr &node) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode != nullptr) {
      if (cnode->size() == 0) {
        return nullptr;
      }
      // Call primitive cnode converter.
      return ConvertPrimitiveCNode(cnode);
    }
    auto value_node = node->cast<ValueNodePtr>();
    if (value_node != nullptr) {
      const auto &value = value_node->value();
      if (value == nullptr) {
        return nullptr;
      }
      // Call value node converter.
      return ConvertValueNode(value_node, value);
    }
    return nullptr;
  }

  virtual void UpdateAbstracts() {
    const auto &nodes = manager_->all_nodes();
    for (const auto &node : nodes) {
      const auto &abs = node->abstract();
      if (abs == nullptr) {
        continue;
      }
      bool is_interpret_dict = false;
      // Do not convert the abstract of Interpret node(AbstractDictionary) to AbstractSequence.
      if (abs->isa<AbstractDictionary>()) {
        AbstractDictionaryPtr abs_dict = abs->cast<AbstractDictionaryPtr>();
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
      auto new_abs = ConvertAbstract(abs);
      if (new_abs != nullptr) {
        node->set_abstract(new_abs);
      }
    }
  }

  static int64_t GetElementIndex(const std::vector<AbstractElementPair> &attrs, const AnfNodePtr &name) {
    auto n_attrs = attrs.size();
    auto name_abstract = GetAbstract<AbstractBase>(name);
    MS_EXCEPTION_IF_NULL(name_abstract);
    auto name_value = name_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(name_value);
    for (size_t i = 0; i < n_attrs; ++i) {
      if (*name_value == *attrs[i].first->BuildValue()) {
        return SizeToLong(i);
      }
    }
    return SizeToLong(n_attrs);
  }

 private:
  bool need_renormalized_{false};
};

// ===========================================================================
// BeforeOptARewriter convert ObjectClass, Dictionary to Tuple.
// ===========================================================================
class BeforeOptARewriter : public BaseRewriter {
 public:
  using ThisClass = BeforeOptARewriter;
  BeforeOptARewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager)
      : BaseRewriter(root_graph, manager), is_dict_output_{HasDictOutput()} {}
  ~BeforeOptARewriter() override = default;

  bool Execute() override {
    bool changed = Run();
    if (changed) {
      UpdateAbstracts();
    }
    ConvertParameter();
    return changed;
  }

 protected:
  void ConvertParameter() {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime || !is_dict_output_) {
      return;
    }
    for (const auto &para : root_graph_->parameters()) {
      auto new_node_and_abs = ConvertParameterDictAbstract(para, para->abstract());
      if (new_node_and_abs.first == para) {
        continue;
      }
      (void)manager_->Replace(para, new_node_and_abs.first);
      para->set_abstract(new_node_and_abs.second);
    }
  }

  std::pair<AnfNodePtr, AbstractBasePtr> ConvertParameterDictAbstract(const AnfNodePtr &cur_node,
                                                                      const AbstractBasePtr &cur_abs) {
    MS_EXCEPTION_IF_NULL(cur_abs);
    auto seq_abs = cur_abs->cast_ptr<AbstractSequence>();
    if (seq_abs != nullptr) {
      bool is_tuple = seq_abs->isa<AbstractTuple>();
      auto seq_prim = is_tuple ? prim::kPrimMakeTuple : prim::kPrimMakeList;
      std::vector<AnfNodePtr> seq_inputs{NewValueNode(seq_prim)};
      AbstractBasePtrList abs_list;
      for (size_t i = 0; i < seq_abs->elements().size(); ++i) {
        auto getitem_prim = is_tuple ? prim::kPrimTupleGetItem : prim::kPrimListGetItem;
        auto next_node =
          root_graph_->NewCNodeInOrder({NewValueNode(getitem_prim), cur_node, NewValueNode(SizeToLong(i))});
        auto node_and_abs = ConvertParameterDictAbstract(next_node, seq_abs->elements()[i]);
        (void)seq_inputs.emplace_back(node_and_abs.first);
        (void)abs_list.emplace_back(node_and_abs.second);
      }
      if (is_tuple) {
        return std::make_pair(root_graph_->NewCNodeInOrder(seq_inputs), std::make_shared<AbstractTuple>(abs_list));
      }
      return std::make_pair(root_graph_->NewCNodeInOrder(seq_inputs), std::make_shared<AbstractList>(abs_list));
    }
    auto dict_abs = cur_abs->cast_ptr<AbstractDictionary>();
    if (dict_abs != nullptr) {
      std::vector<AnfNodePtr> key_inputs{NewValueNode(prim::kPrimMakeTuple)};
      std::vector<AnfNodePtr> value_inputs{NewValueNode(prim::kPrimMakeTuple)};
      AbstractBasePtrList abs_list;
      for (size_t i = 0; i < dict_abs->elements().size(); ++i) {
        auto next_node =
          root_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), cur_node, NewValueNode(SizeToLong(i))});
        auto node_and_abs = ConvertParameterDictAbstract(next_node, dict_abs->elements()[i].second);
        (void)key_inputs.emplace_back(NewValueNode(dict_abs->elements()[i].first->BuildValue()));
        (void)value_inputs.emplace_back(node_and_abs.first);
        (void)abs_list.emplace_back(node_and_abs.second);
      }
      auto make_dict =
        root_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), root_graph_->NewCNodeInOrder(key_inputs),
                                      root_graph_->NewCNodeInOrder(value_inputs)});
      return std::make_pair(make_dict, std::make_shared<AbstractTuple>(abs_list));
    }
    return std::make_pair(cur_node, cur_abs);
  }

  static std::string GetStringValue(const AnfNodePtr &node) {
    auto str = GetValueNode<StringImmPtr>(node);
    if (str == nullptr) {
      return "";
    }
    return str->value();
  }

  static CNodePtr NewTupleGetCNode(const AnfNodePtr &cnode, const AnfNodePtr &data_node,
                                   const std::vector<AbstractElementPair> &elements, const AnfNodePtr &name_node) {
    int64_t index = GetElementIndex(elements, name_node);
    auto index_node = NewValueNode(index);
    auto prim_node = NewValueNode(prim::kPrimTupleGetItem);
    return cnode->func_graph()->NewCNode({prim_node, data_node, index_node});
  }

  // From:
  //   DictGetItem(data:AbstractDictionary, key:AbstractBase)
  // To:
  //   TupleGetItem(data, index:Int64Imm)
  AnfNodePtr ConvertDictGetItemToTupleGetItem(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [dict_getitem, dict, item]
    const size_t expect_inputs_size = 3;
    CheckInputsSize(node, expect_inputs_size);

    constexpr size_t data_index = 1;
    constexpr size_t key_index = 2;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[key_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    return NewTupleGetCNode(node, data, abs_dict->elements(), key);
  }

  AnfNodePtr ConvertDictGetItem(const CNodePtr &node) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime || !is_dict_output_) {
      return ConvertDictGetItemToTupleGetItem(node);
    }
    return nullptr;
  }

  // From:
  //   DictSetItem(data:AbstractDictionary, key:AbstractBase, value)
  // To:
  //   TupleSetItem(data, index:Int64Imm, value)
  // Or:
  //   tuple_add(data, value)
  AnfNodePtr ConvertDictSetItemToTupleSetItem(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [dict_setitem, dict, item, value]
    const size_t expect_inputs_size = 4;
    CheckInputsSize(node, expect_inputs_size);

    const size_t data_index = 1;
    const size_t cons_index = 2;
    const size_t item_value_index = 3;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[cons_index];
    auto &item_value = inputs[item_value_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    int64_t index = GetElementIndex(abs_dict->elements(), key);
    auto func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    if (index >= static_cast<int64_t>(abs_dict->elements().size())) {
      // For dictionary set, if the key does not exist, we should create a new item.
      std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
      for (size_t i = 0; i < abs_dict->elements().size(); ++i) {
        auto tuple_getitem_i =
          func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, NewValueNode(SizeToLong(i))});
        (void)make_tuple_inputs.emplace_back(tuple_getitem_i);
      }
      (void)make_tuple_inputs.emplace_back(item_value);
      auto new_node = func_graph->NewCNode(make_tuple_inputs);
      new_node->set_debug_info(node->debug_info());
      return new_node;
    }
    auto index_node = NewValueNode(index);
    auto new_node = func_graph->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, index_node, item_value});
    new_node->set_debug_info(node->debug_info());
    return new_node;
  }

  bool HasDictOutput() const {
    const AnfNodePtr &output = root_graph_->output();
    return CheckContainsDict(output->abstract());
  }

  AnfNodePtr ConvertDictSetItem(const CNodePtr &node) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime || !is_dict_output_) {
      return ConvertDictSetItemToTupleSetItem(node);
    }
    return nullptr;
  }

  // From:
  //   MakeDict(name, input)
  // To:
  //   input
  AnfNodePtr EraseMakeDictNode(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    constexpr size_t expect_inputs_size = 3;
    constexpr size_t input_index = 2;
    CheckInputsSize(node, expect_inputs_size);
    return node->input(input_index);
  }

  bool CheckUserHasPyExecute(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    auto func = node->func_graph();
    MS_EXCEPTION_IF_NULL(func);
    auto mng = func->manager();
    auto &users = mng->node_users()[node];
    for (auto &user : users) {
      if (IsPrimitiveCNode(user.first, prim::kPrimPyExecute)) {
        return true;
      } else if (IsPrimitiveCNode(user.first, prim::kPrimMakeTuple)) {
        if (CheckUserHasPyExecute(user.first->cast<CNodePtr>())) {
          return true;
        }
      }
    }
    return false;
  }

  AnfNodePtr ConvertMakeDict(const CNodePtr &node) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime || (!is_dict_output_ && !CheckUserHasPyExecute(node))) {
      auto new_node = EraseMakeDictNode(node);
      return new_node;
    }
    return nullptr;
  }

  // From:
  //   DictGetValues(dict:AbstractDictionary)
  // To:
  //   dict
  AnfNodePtr EraseDictGetValues(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    constexpr size_t expect_inputs_size = 2;
    CheckInputsSize(node, expect_inputs_size);
    auto input = node->input(1);
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime || !is_dict_output_) {
      return input;
    }
    auto abs_dict = GetAbstract<AbstractDictionary>(input);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    const auto &elements = abs_dict->elements();
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.reserve(elements.size() + 1);
    (void)new_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    for (const auto &element : elements) {
      MS_EXCEPTION_IF_NULL(element.first->BuildValue());
      AnfNodePtr value_node =
        fg->NewCNode({NewValueNode(prim::kPrimDictGetItem), input, NewValueNode(element.first->BuildValue())});
      (void)new_inputs.emplace_back(value_node);
    }
    return fg->NewCNode(std::move(new_inputs));
  }

  // From:
  //   DictItems(dict:AbstractDictionary)
  // To:
  //   kPrimMakeList(MakeTuple(key0, TupleGetItem(dict, 0)), ...)
  AnfNodePtr EraseDictItems(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    constexpr size_t expect_inputs_size = 2;
    CheckInputsSize(node, expect_inputs_size);

    const auto &input = node->input(1);
    auto abs_dict = GetAbstract<AbstractDictionary>(input);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    const auto &elements = abs_dict->elements();
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.reserve(elements.size() + 1);
    (void)new_inputs.emplace_back(NewValueNode(prim::kPrimMakeList));
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    bool convert_to_tuple = !allow_fallback_runtime || !is_dict_output_;
    for (size_t i = 0; i < elements.size(); ++i) {
      auto index_node = NewValueNode(static_cast<int64_t>(i));
      MS_EXCEPTION_IF_NULL(elements[i].first->BuildValue());
      auto key_node = NewValueNode(elements[i].first->BuildValue());
      AnfNodePtr value_node;
      if (convert_to_tuple) {
        value_node = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, index_node});
      } else {
        value_node =
          fg->NewCNode({NewValueNode(prim::kPrimDictGetItem), input, NewValueNode(elements[i].first->BuildValue())});
      }
      auto tuple_node = fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), key_node, value_node});
      (void)new_inputs.emplace_back(tuple_node);
    }
    return fg->NewCNode(std::move(new_inputs));
  }

  // From:
  //   MakeKeywordArg(key, value)
  // To:
  //   value
  AnfNodePtr EraseMakeKeywordArgNode(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    // Inputs should be [make_keyword_arg, key, value]
    constexpr size_t expect_input_size = 3;
    constexpr size_t value_inputs_index = 2;
    CheckInputsSize(node, expect_input_size);
    return node->input(value_inputs_index);
  }

  // From:
  //   ExtractKeywordArg(arg, key)
  // To:
  //   key
  AnfNodePtr EraseExtractKeywordArg(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    // Inputs should be [extract_keyword_arg, arg, key]
    const size_t expect_inputs_size = 3;
    CheckInputsSize(node, expect_inputs_size);
    constexpr size_t key_index = 2;
    return node->input(key_index);
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &) const;
  using ConverterMap = std::unordered_map<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
    {prim::kPrimDictGetItem, &ThisClass::ConvertDictGetItem},
    {prim::kPrimDictSetItem, &ThisClass::ConvertDictSetItem},
    {prim::kPrimDictGetValues, &ThisClass::EraseDictGetValues},
    {prim::kPrimMakeDict, &ThisClass::ConvertMakeDict},
    {prim::kPrimMakeKeywordArg, &ThisClass::EraseMakeKeywordArgNode},
    {prim::kPrimExtractKeywordArg, &ThisClass::EraseExtractKeywordArg},
    {prim::kPrimDictItems, &ThisClass::EraseDictItems},
  };

  AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode) override {
    // Get primitive from cnode.
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      return nullptr;
    }
    // Find cnode converter by primitive.
    auto iter = converters_.find(prim);
    if (iter == converters_.end()) {
      return nullptr;
    }
    // Call converter.
    return (this->*(iter->second))(cnode);
  }

  ValuePtr ConvertDictValue(const ValuePtr &value, size_t depth, bool convert_dict, bool *need_convert) const {
    MS_EXCEPTION_IF_NULL(value);
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(INTERNAL_EXCEPTION) << "List, tuple and dict nesting is not allowed more than " << kMaxSeqRecursiveDepth
                                 << " levels.";
    }
    if (value->isa<ValueSequence>()) {
      auto value_seq = value->cast<ValueSequencePtr>();
      std::vector<ValuePtr> value_vec;
      value_vec.reserve(value_seq->size());
      bool new_need_convert = false;
      for (const auto &element : value_seq->value()) {
        (void)value_vec.emplace_back(ConvertDictValue(element, depth + 1, convert_dict, &new_need_convert));
      }
      if (!new_need_convert) {
        return value;
      }
      *need_convert = true;
      if (value->isa<ValueTuple>()) {
        return std::make_shared<ValueTuple>(value_vec);
      }
      return std::make_shared<ValueList>(value_vec);
    }
    // dict(k0:v0, k1:v1, ...) --> tuple(v0, v1, ...)
    if (value->isa<ValueDictionary>() && convert_dict) {
      *need_convert = true;
      const auto &keys_values = value->cast<ValueDictionaryPtr>()->value();
      std::vector<ValuePtr> value_vec;
      value_vec.reserve(keys_values.size());
      for (const auto &element : keys_values) {
        (void)value_vec.emplace_back(ConvertDictValue(element.second, depth + 1, convert_dict, need_convert));
      }
      return std::make_shared<ValueTuple>(value_vec);
    }
    return value;
  }

  AnfNodePtr ConvertValueNode(const ValueNodePtr &value_node, const ValuePtr &value) override {
    // Convert Dictionary value node.
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    bool convert_dict = !allow_fallback_runtime || !is_dict_output_;
    bool need_convert = false;
    auto new_value = ConvertDictValue(value, 0, convert_dict, &need_convert);
    if (need_convert) {
      auto new_node = NewValueNode(new_value);
      new_node->set_debug_info(value_node->debug_info());
      return new_node;
    }
    return nullptr;
  }

  static std::shared_ptr<AbstractTuple> MakeAbstractTuple(const std::vector<AbstractElementPair> &attrs) {
    std::vector<AbstractBasePtr> elements;
    elements.reserve(attrs.size());
    (void)std::transform(attrs.begin(), attrs.end(), std::back_inserter(elements),
                         [](const auto &item) { return item.second; });
    return std::make_shared<AbstractTuple>(std::move(elements));
  }

  // AbstractDictionary --> AbstractSequence.
  AbstractSequencePtr ConvertToAbstractSequence(const AbstractBasePtr &abs, size_t depth) {
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(INTERNAL_EXCEPTION) << "List, tuple and dict nesting is not allowed more than " << kMaxSeqRecursiveDepth
                                 << " levels.";
    }
    auto abs_seq = abs->cast<AbstractSequencePtr>();
    if (abs_seq != nullptr) {
      const auto &seq_elements = abs_seq->elements();
      // First we check if elements should be converted,
      // changed_elements maps old element to new element.
      mindspore::HashMap<AbstractBasePtr, AbstractBasePtr> changed_elements;
      for (const auto &element : seq_elements) {
        auto new_element = ConvertToAbstractSequence(element, depth + 1);
        if (new_element != nullptr) {
          (void)changed_elements.emplace(element, new_element);
        }
      }
      if (changed_elements.empty()) {
        // Here the AbstractList don't need to convert to AbstractTuple.
        return nullptr;
      }
      // Always make new AbstractSequence when elements changed.
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
      // Here the AbstractList don't need to convert to AbstractTuple.
      if (abs_seq->isa<AbstractList>()) {
        return std::make_shared<AbstractList>(std::move(elements));
      } else {
        return std::make_shared<AbstractTuple>(std::move(elements));
      }
    }
    // AbstractDictionary --> AbstractTuple.
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    bool convert_to_tuple = !allow_fallback_runtime || !is_dict_output_;
    auto abs_dict = abs->cast<AbstractDictionaryPtr>();
    if (abs_dict != nullptr && convert_to_tuple) {
      const auto &dict_elements = abs_dict->elements();
      std::vector<AbstractBasePtr> elements;
      elements.reserve(dict_elements.size());
      for (const auto &element : dict_elements) {
        auto new_element = ConvertToAbstractSequence(element.second, depth + 1);
        if (new_element != nullptr) {
          (void)elements.emplace_back(new_element);
        } else {
          (void)elements.emplace_back(element.second);
        }
      }
      return std::make_shared<AbstractTuple>(elements);
    }
    return nullptr;
  }

  AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) override {
    // AbstractDictionary --> AbstractSequence.
    return ConvertToAbstractSequence(abs, 0);
  }

 private:
  bool is_dict_output_{false};
};

// ==================================================================
// AfterOptARewriter converts List, Sparse, RowTensor to Tuple.
// ==================================================================
class AfterOptARewriter : public BaseRewriter {
 public:
  using ThisClass = AfterOptARewriter;
  AfterOptARewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager, bool vm_pipeline)
      : BaseRewriter(root_graph, manager), vm_pipeline_(vm_pipeline) {}
  ~AfterOptARewriter() override = default;

 protected:
  // From:
  //   MakeSparseTensor(indices, values, dense_shape)
  // To:
  //   MakeTuple(indices, values, dense_shape)
  AnfNodePtr ConvertMakeSparseToMakeTuple(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    std::vector<AnfNodePtr> inputs;
    inputs.reserve(node->size());
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    // Inputs of node should be [make_sparse, indices, values, dense_shape], so offset by 1 to get items.
    (void)inputs.insert(inputs.cend(), node->inputs().cbegin() + 1, node->inputs().cend());
    auto new_node = node->func_graph()->NewCNode(std::move(inputs));
    new_node->set_abstract(node->abstract());
    return new_node;
  }

  static inline const mindspore::HashMap<std::string, int64_t> sparse_attr_map = {
    {prim::kCSRTensorGetIndptr, 0},     {prim::kCSRTensorGetIndices, 1}, {prim::kCSRTensorGetValues, 2},
    {prim::kCSRTensorGetDenseShape, 3}, {prim::kCOOTensorGetIndices, 0}, {prim::kCOOTensorGetValues, 1},
    {prim::kCOOTensorGetDenseShape, 2}, {prim::kRowTensorGetIndices, 0}, {prim::kRowTensorGetValues, 1},
    {prim::kRowTensorGetDenseShape, 2}};

  // From:
  //   SparseTensorGetXXX(sparse) # index
  // To:
  //   TupleGetItem(sparse, index)
  AnfNodePtr ConvertSparseGetAttrToTupleGetItem(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    constexpr size_t kExpectInputSize = 2;
    constexpr size_t kSparseAttrIndex = 1;
    CheckInputsSize(node, kExpectInputSize);

    auto prim = GetValueNode<PrimitivePtr>(node->input(0));
    if (prim != nullptr) {
      auto iter = sparse_attr_map.find(prim->name());
      if (iter != sparse_attr_map.end()) {
        const auto &sparse = node->input(kSparseAttrIndex);
        auto index_node = NewValueNode(iter->second);
        auto new_node = node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), sparse, index_node});
        new_node->set_abstract(node->abstract());
        return new_node;
      }
    }
    return nullptr;
  }

  // DictGetItem --> PyExecute()
  AnfNodePtr ConvertDictGetItem(const CNodePtr &cnode) const {
    MS_EXCEPTION_IF_NULL(cnode);
    // Inputs should be [dict_setitem, dict, item]
    const size_t expect_inputs_size = 3;
    CheckInputsSize(cnode, expect_inputs_size);

    const size_t data_index = 1;
    const size_t item_key_index = 2;
    const auto &inputs = cnode->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[item_key_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    auto func_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);

    // Script
    std::stringstream script_buffer;
    script_buffer << kInternalDictSelfStr << "[" << kInternalDictKeyStr << "]";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Pack local parameters keys.
    const auto script_dict_self_name = std::make_shared<StringImm>(kInternalDictSelfStr);
    const auto script_dict_key_name = std::make_shared<StringImm>(kInternalDictKeyStr);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_self_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_key_name));
    const auto key_value_name_tuple = func_graph->NewCNode(key_value_names_list);

    // Pack the local parameters values, not support list, tuple, or dict.
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(data);
    (void)key_value_list.emplace_back(key);
    const auto key_value_tuple = func_graph->NewCNode(key_value_list);

    // Build the new dict node.
    const auto dict_getitem_node =
      fallback::CreatePyExecuteCNodeInOrder(cnode, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);
    int64_t index = GetElementIndex(abs_dict->elements(), key);
    const auto &val = abs_dict->elements()[index].second;
    const auto &tensor_val = dyn_cast<abstract::AbstractTensor>(val);
    if (tensor_val != nullptr) {
      const auto &tensor_type = tensor_val->element()->BuildType();
      fallback::SetRealType<AnfNode, Type>(dict_getitem_node, tensor_type);
      const auto &tensor_shape = dyn_cast<abstract::Shape>(tensor_val->BuildShape());
      MS_EXCEPTION_IF_NULL(tensor_shape);
      fallback::SetRealShape<AnfNode, abstract::BaseShape>(dict_getitem_node, tensor_shape);
      MS_LOG(DEBUG) << "key: " << key->abstract()->BuildValue()->ToString() << ", type: " << tensor_type->ToString()
                    << ", shape: " << tensor_shape->ToString() << ", val: " << tensor_val->ToString();
    }
    MS_LOG(DEBUG) << "Made dict getitem node: " << dict_getitem_node->DebugString();
    return dict_getitem_node;
  }

  // DictSetItem --> PyExecute()
  AnfNodePtr ConvertDictSetItem(const CNodePtr &cnode) const {
    MS_EXCEPTION_IF_NULL(cnode);
    // Inputs should be [dict_setitem, dict, item, value]
    const size_t expect_inputs_size = 4;
    CheckInputsSize(cnode, expect_inputs_size);

    const size_t data_index = 1;
    const size_t item_key_index = 2;
    const size_t item_value_index = 3;
    const auto &inputs = cnode->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[item_key_index];
    auto &item_value = inputs[item_value_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    auto func_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);

    // Script
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._jit_fallback_utils.dict_setitem(" << kInternalDictSelfStr << ", "
                  << kInternalDictKeyStr << ", " << kInternalDictValueStr << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Pack local parameters keys.
    const auto script_dict_self_name = std::make_shared<StringImm>(kInternalDictSelfStr);
    const auto script_dict_key_name = std::make_shared<StringImm>(kInternalDictKeyStr);
    const auto script_dict_value_name = std::make_shared<StringImm>(kInternalDictValueStr);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_self_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_key_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_value_name));
    const auto key_value_name_tuple = func_graph->NewCNode(key_value_names_list);

    // Pack the local parameters values, not support list, tuple, or dict.
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(data);
    (void)key_value_list.emplace_back(key);
    (void)key_value_list.emplace_back(item_value);
    const auto key_value_tuple = func_graph->NewCNode(key_value_list);

    // Build the new dict node.
    const auto dict_setitem_node =
      fallback::CreatePyExecuteCNodeInOrder(cnode, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);
    MS_LOG(DEBUG) << "Made dict setitem node: " << dict_setitem_node->DebugString();
    return dict_setitem_node;
  }

  AnfNodePtr ConstructInternalTupleKeysNode(const FuncGraphPtr &fg, const AnfNodePtr &keys_node) const {
    constexpr auto internal_tuple_keys_str = "__internal_tuple_keys__";
    MS_EXCEPTION_IF_NULL(fg);
    const auto script_key_tuple_str = std::make_shared<StringImm>(internal_tuple_keys_str);
    auto dict_py_exec_key = std::make_shared<ValueTuple>(std::vector<ValuePtr>{script_key_tuple_str});
    auto dict_tuple_key_value = fg->NewCNode({std::make_shared<ValueNode>(prim::kPrimMakeTuple), keys_node});
    const auto make_key_tuple_node =
      fallback::CreatePyExecuteCNode(fg, NewValueNode(script_key_tuple_str), NewValueNode(dict_py_exec_key),
                                     dict_tuple_key_value, keys_node->debug_info());
    return make_key_tuple_node;
  }

  AnfNodePtr ConstructInternalTupleValueNode(const FuncGraphPtr &fg, const AnfNodePtr &values_node) const {
    constexpr auto internal_tuple_values_str = "__internal_tuple_values__";
    MS_EXCEPTION_IF_NULL(fg);
    const auto script_value_tuple_str = std::make_shared<StringImm>(internal_tuple_values_str);
    auto dict_py_exec_value = std::make_shared<ValueTuple>(std::vector<ValuePtr>{script_value_tuple_str});
    auto dict_tuple_node = fg->NewCNode({std::make_shared<ValueNode>(prim::kPrimMakeTuple), values_node});
    const auto make_value_tuple_node =
      fallback::CreatePyExecuteCNode(fg, NewValueNode(script_value_tuple_str), NewValueNode(dict_py_exec_value),
                                     dict_tuple_node, values_node->debug_info());
    return make_value_tuple_node;
  }

  AnfNodePtr ConstructNewDictNode(const FuncGraphPtr &fg, const AnfNodePtr &make_key_tuple_node,
                                  const AnfNodePtr &make_value_tuple_node) const {
    constexpr auto internal_dict_zip_keys_str = "__internal_dict_zip_keys__";
    constexpr auto internal_dict_zip_values_str = "__internal_dict_zip_values__";
    // Pack the local parameters values
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(make_key_tuple_node);
    (void)key_value_list.emplace_back(make_value_tuple_node);
    const auto key_value_tuple = fg->NewCNode(key_value_list);

    // Pack local parameters keys.
    const auto script_dict_key_name = std::make_shared<StringImm>(internal_dict_zip_keys_str);
    const auto script_dict_value_name = std::make_shared<StringImm>(internal_dict_zip_values_str);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_key_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_value_name));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);

    // Construct Script Node
    std::stringstream script_buffer;
    script_buffer << "dict(zip(" << internal_dict_zip_keys_str << "," << internal_dict_zip_values_str << "),)";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Build the new dict node.
    const auto make_dict_node = fallback::CreatePyExecuteCNodeInOrder(
      fg, NewValueNode(script_str), key_value_name_tuple, key_value_tuple, make_key_tuple_node->debug_info());
    MS_LOG(DEBUG) << "Made dict node: " << make_dict_node->DebugString();
    return make_dict_node;
  }

  // MakeDict(keys, values) --> PyExecute('dict(zip(keys, values))', ...)
  AnfNodePtr ConvertMakeDict(const CNodePtr &node) const {
    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // Local parameters values.
    // Get the key tuple.
    constexpr size_t keys_input_index = 1;
    auto keys_node = node->input(keys_input_index);
    const auto make_key_tuple_node = ConstructInternalTupleKeysNode(fg, keys_node);
    make_key_tuple_node->set_debug_info(node->input(keys_input_index)->debug_info());
    // Get the value tuple.
    constexpr size_t values_input_index = 2;
    auto values_node = node->input(values_input_index);
    const auto make_value_tuple_node = ConstructInternalTupleValueNode(fg, values_node);
    make_value_tuple_node->set_debug_info(node->input(values_input_index)->debug_info());

    auto new_dict_node = ConstructNewDictNode(fg, make_key_tuple_node, make_value_tuple_node);
    new_dict_node->set_debug_info(node->debug_info());
    return new_dict_node;
  }

  AnfNodePtr GenerateTupleInput(const CNodePtr &node) const {
    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    const auto &inputs = node->inputs();
    constexpr auto internal_element_str_prefix = "__internal_list_element_";
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    std::stringstream script_buffer;
    script_buffer << "(";
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (IsValueNode<None>(inputs[i])) {
        script_buffer << "None, ";
        continue;
      }
      std::string cur_element = internal_element_str_prefix + std::to_string(i) + "_";
      (void)key_value_names_list.emplace_back(NewValueNode(cur_element));
      (void)key_value_list.emplace_back(inputs[i]);
      script_buffer << cur_element << ", ";
    }
    script_buffer << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);
    const auto key_value_tuple = fg->NewCNode(key_value_list);
    auto list_node =
      fallback::CreatePyExecuteCNode(node, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);
    return list_node;
  }

  // MakeList(x1, x2, ...) --> PyExecute('[x1, x2, ...]', ...)
  AnfNodePtr ConvertMakeList(const CNodePtr &node) const {
    if (!fallback::EnableFallbackList()) {
      return nullptr;
    }

    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);

    auto list_node_input = GenerateTupleInput(node);

    if (!fallback::HasPySeqObject(node->abstract())) {
      MS_LOG(EXCEPTION) << "MakeList node: " << node->DebugString() << " do not have python list object.";
    }
    py::list list_object = *fallback::GetPySeqObject<AbstractBase, py::list>(node->abstract());
    const std::string list_obj_str_prefix = "__list_py_object_";
    auto list_obj_id = fallback::GetPyObjectPtrStr(list_object);
    MS_LOG(DEBUG) << "Current python object id: " << list_obj_id;
    auto list_obj_str = list_obj_str_prefix + list_obj_id + "_";
    fallback::SetPyObjectToLocalVariable(list_obj_str, list_object);

    const auto list_key_input = "__internal_list_key__";
    const auto list_value_input = "__internal_list_value__";
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._jit_fallback_utils.generate_list(" << list_key_input << ", "
                  << list_value_input << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(list_key_input));
    (void)key_value_names_list.emplace_back(NewValueNode(list_value_input));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(NewValueNode(list_obj_str));
    (void)key_value_list.emplace_back(list_node_input);
    const auto key_value_tuple = fg->NewCNode(key_value_list);
    auto res = fallback::CreatePyExecuteCNode(node, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);

    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(list_abs);

    res->set_debug_info(node->debug_info());
    // Set real type and shape
    fallback::SetRealType(res, list_abs->BuildType());
    fallback::SetRealShape(res, list_abs->BuildShape());
    fallback::SetPySeqObject<AnfNode, py::list>(res, std::make_shared<py::list>(list_object));

    MS_LOG(DEBUG) << "Convert make_list node to PyExecute node: " << res->DebugString();
    return res;
  }

  // x.extend(y) --> PyExecute(_jit_fallback_list_inplace_extend(x, y))
  AnfNodePtr ConvertListInplaceExtend(const CNodePtr &node) const {
    if (!fallback::EnableFallbackList()) {
      return nullptr;
    }
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(list_abs);

    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    constexpr auto internal_list_input = "__internal_list_input__";
    constexpr auto internal_target_input = "__internal_target_input__";
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._jit_fallback_utils.list_inplace_extend(" << internal_list_input
                  << ", " << internal_target_input << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(internal_list_input));
    (void)key_value_names_list.emplace_back(NewValueNode(internal_target_input));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);

    const auto &node_inputs = node->inputs();
    constexpr size_t min_node_inputs_size = 3;
    constexpr size_t max_node_inputs_size = 4;
    size_t inputs_size = node_inputs.size();
    if (inputs_size != min_node_inputs_size && inputs_size != max_node_inputs_size) {
      MS_LOG(EXCEPTION) << "The size of input to ListInplaceExtend should be " << min_node_inputs_size << " or "
                        << max_node_inputs_size << " but got " << inputs_size;
    }
    constexpr size_t node_list_index = 1;
    constexpr size_t node_target_index = 2;
    auto list_input_node = node_inputs[node_list_index];
    if (IsPrimitiveCNode(list_input_node, prim::kPrimMakeList)) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(list_input_node->debug_info()));
      auto new_node = ConvertMakeList(list_input_node->cast<CNodePtr>());
      (void)manager_->Replace(list_input_node, new_node);
      list_input_node = new_node;
    }
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(list_input_node);
    (void)key_value_list.emplace_back(node_inputs[node_target_index]);
    const auto key_value_tuple = fg->NewCNode(key_value_list);

    auto res = fallback::CreatePyExecuteCNode(node, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);

    if (inputs_size == max_node_inputs_size) {
      res->add_input(node_inputs[max_node_inputs_size - 1]);
    }

    res->set_debug_info(node->debug_info());

    // Set real type and shape
    fallback::SetRealType(res, list_abs->BuildType());
    fallback::SetRealShape(res, list_abs->BuildShape());
    if (list_abs->has_list_py_obj()) {
      fallback::SetPySeqObject<AnfNode, py::list>(res, list_abs->list_py_obj<py::list>());
    }

    MS_LOG(DEBUG) << "Convert list inplace append node to PyExecute node: " << res->DebugString();
    return res;
  }

  // x.insert(index, y) --> PyExecute(_jit_fallback_list_inplace_insert(x, index, y))
  AnfNodePtr ConvertListInplaceInsert(const CNodePtr &node) const {
    if (!fallback::EnableFallbackList()) {
      return nullptr;
    }
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(list_abs);

    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    constexpr auto internal_list_input = "__internal_list_input__";
    constexpr auto internal_index_input = "__internal_index_input__";
    constexpr auto internal_target_input = "__internal_target_input__";
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._jit_fallback_utils.list_inplace_insert(" << internal_list_input
                  << ", " << internal_index_input << ", " << internal_target_input << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(internal_list_input));
    (void)key_value_names_list.emplace_back(NewValueNode(internal_index_input));
    (void)key_value_names_list.emplace_back(NewValueNode(internal_target_input));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);

    const auto &node_inputs = node->inputs();
    constexpr size_t min_node_inputs_size = 4;
    constexpr size_t max_node_inputs_size = 5;
    size_t inputs_size = node_inputs.size();
    if (inputs_size != min_node_inputs_size && inputs_size != max_node_inputs_size) {
      MS_LOG(EXCEPTION) << "The size of input to ListInplaceInsert should be " << min_node_inputs_size << " or "
                        << max_node_inputs_size << " but got " << inputs_size;
    }
    constexpr size_t node_list_index = 1;
    constexpr size_t node_index_index = 2;
    constexpr size_t node_target_index = 3;
    auto list_input_node = node_inputs[node_list_index];
    if (IsPrimitiveCNode(list_input_node, prim::kPrimMakeList)) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(list_input_node->debug_info()));
      auto new_node = ConvertMakeList(list_input_node->cast<CNodePtr>());
      (void)manager_->Replace(list_input_node, new_node);
      list_input_node = new_node;
    }
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(list_input_node);
    (void)key_value_list.emplace_back(node_inputs[node_index_index]);
    (void)key_value_list.emplace_back(node_inputs[node_target_index]);
    const auto key_value_tuple = fg->NewCNode(key_value_list);

    auto res = fallback::CreatePyExecuteCNode(node, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);

    if (inputs_size == max_node_inputs_size) {
      res->add_input(node_inputs[max_node_inputs_size - 1]);
    }

    res->set_debug_info(node->debug_info());

    // Set real type and shape
    fallback::SetRealType(res, list_abs->BuildType());
    fallback::SetRealShape(res, list_abs->BuildShape());
    if (list_abs->has_list_py_obj()) {
      fallback::SetPySeqObject<AnfNode, py::list>(res, list_abs->list_py_obj<py::list>());
    }

    MS_LOG(DEBUG) << "Convert list inplace insert node to PyExecute node: " << res->DebugString();
    return res;
  }

  // x.pop(index) --> PyExecute(_jit_fallback_list_inplace_pop(x, index, y))
  AnfNodePtr ConvertListInplacePop(const CNodePtr &node) const {
    if (!fallback::EnableFallbackList()) {
      return nullptr;
    }

    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(list_abs);

    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    constexpr auto internal_list_input = "__internal_list_input__";
    constexpr auto internal_index_input = "__internal_index_input__";
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._jit_fallback_utils.list_inplace_pop(" << internal_list_input
                  << ", " << internal_index_input << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(internal_list_input));
    (void)key_value_names_list.emplace_back(NewValueNode(internal_index_input));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);

    const auto &node_inputs = node->inputs();
    constexpr size_t min_node_inputs_size = 3;
    constexpr size_t max_node_inputs_size = 4;
    size_t inputs_size = node_inputs.size();
    if (inputs_size != min_node_inputs_size && inputs_size != max_node_inputs_size) {
      MS_LOG(EXCEPTION) << "The size of input to ListInplacePop should be " << min_node_inputs_size << " or "
                        << max_node_inputs_size << " but got " << inputs_size;
    }
    constexpr size_t node_list_index = 1;
    constexpr size_t node_index_index = 2;
    auto list_input_node = node_inputs[node_list_index];
    if (IsPrimitiveCNode(list_input_node, prim::kPrimMakeList)) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(list_input_node->debug_info()));
      auto new_node = ConvertMakeList(list_input_node->cast<CNodePtr>());
      (void)manager_->Replace(list_input_node, new_node);
      list_input_node = new_node;
    }
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(list_input_node);
    (void)key_value_list.emplace_back(node_inputs[node_index_index]);
    const auto key_value_tuple = fg->NewCNode(key_value_list);

    auto res = fallback::CreatePyExecuteCNode(node, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);

    if (inputs_size == max_node_inputs_size) {
      res->add_input(node_inputs[max_node_inputs_size - 1]);
    }

    res->set_debug_info(node->debug_info());

    // Set real type and shape
    fallback::SetRealType(res, list_abs->BuildType());
    fallback::SetRealShape(res, list_abs->BuildShape());
    if (list_abs->has_list_py_obj()) {
      fallback::SetPySeqObject<AnfNode, py::list>(res, list_abs->list_py_obj<py::list>());
    }

    MS_LOG(DEBUG) << "Convert list inplace pop node to PyExecute node: " << res->DebugString();
    return res;
  }

  // x.reverse() --> PyExecute(_jit_fallback_list_inplace_reverse(x))
  AnfNodePtr ConvertListInplaceReverse(const CNodePtr &node) const {
    if (!fallback::EnableFallbackList()) {
      return nullptr;
    }
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(list_abs);

    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    constexpr auto internal_list_input = "__internal_list_input__";
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._jit_fallback_utils.list_inplace_reverse(" << internal_list_input
                  << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(internal_list_input));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);

    const auto &node_inputs = node->inputs();
    constexpr size_t min_node_inputs_size = 2;
    constexpr size_t max_node_inputs_size = 3;
    size_t inputs_size = node_inputs.size();
    if (inputs_size != min_node_inputs_size && inputs_size != max_node_inputs_size) {
      MS_LOG(EXCEPTION) << "The size of input to ListInplaceAppend should be " << min_node_inputs_size << " or "
                        << max_node_inputs_size << " but got " << inputs_size;
    }
    constexpr size_t node_list_index = 1;
    auto list_input_node = node_inputs[node_list_index];
    if (IsPrimitiveCNode(list_input_node, prim::kPrimMakeList)) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(list_input_node->debug_info()));
      auto new_node = ConvertMakeList(list_input_node->cast<CNodePtr>());
      (void)manager_->Replace(list_input_node, new_node);
      list_input_node = new_node;
    }
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(list_input_node);
    const auto key_value_tuple = fg->NewCNode(key_value_list);
    auto res = fallback::CreatePyExecuteCNode(node, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);

    if (inputs_size == max_node_inputs_size) {
      res->add_input(node_inputs[max_node_inputs_size - 1]);
    }

    res->set_debug_info(node->debug_info());

    // Set real type and shape
    fallback::SetRealType(res, list_abs->BuildType());
    fallback::SetRealShape(res, list_abs->BuildShape());
    if (list_abs->has_list_py_obj()) {
      fallback::SetPySeqObject<AnfNode, py::list>(res, list_abs->list_py_obj<py::list>());
    }

    MS_LOG(DEBUG) << "Convert list inplace reverse node to PyExecute node: " << res->DebugString();
    return res;
  }

  // x.clear() --> PyExecute(_jit_fallback_list_inplace_clear(x))
  AnfNodePtr ConvertListInplaceClear(const CNodePtr &node) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime) {
      return nullptr;
    }
    static const auto allow_inplace_ops = common::GetEnv("MS_DEV_FALLBACK_SUPPORT_LIST") == "1";
    if (!allow_inplace_ops) {
      return nullptr;
    }

    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    constexpr auto internal_list_input = "__internal_list_input__";
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._jit_fallback_utils.list_inplace_clear(" << internal_list_input
                  << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(internal_list_input));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);

    const auto &node_inputs = node->inputs();
    constexpr size_t node_inputs_size = 2;
    if (node_inputs.size() != node_inputs_size) {
      MS_LOG(EXCEPTION) << "The size of input to ListInplaceClear should be " << node_inputs_size << " but got "
                        << node_inputs.size();
    }
    constexpr size_t node_list_index = 1;
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(node_inputs[node_list_index]);
    const auto key_value_tuple = fg->NewCNode(key_value_list);

    auto res = fallback::CreatePyExecuteCNode(node, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);

    res->set_debug_info(node->debug_info());
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(list_abs);

    if (!list_abs->has_list_py_obj()) {
      MS_LOG(ERROR) << "ListInplaceAppend abstract has no python object.";
    }
    py::list list_object =
      list_abs->has_list_py_obj() ? *(list_abs->list_py_obj<py::list>()) : ValueToPyData(abs->BuildValue());

    // Set real type and shape
    fallback::SetRealType(res, list_abs->BuildType());
    fallback::SetRealShape(res, list_abs->BuildShape());
    fallback::SetPySeqObject<AnfNode, py::list>(res, std::make_shared<py::list>(list_object));

    MS_LOG(DEBUG) << "Convert list inplace clear node to PyExecute node: " << res->DebugString();
    return res;
  }

  // raise(string, keys, values, io) --> PyExecute(string, keys, values, io)
  AnfNodePtr ConvertRaise(const CNodePtr &cnode) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime) {
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    MS_LOG(DEBUG) << "Raise node: " << cnode->DebugString();
    const auto &inputs = cnode->inputs();
    std::shared_ptr<raiseutils::KeyValueInfo> key_value = std::make_shared<raiseutils::KeyValueInfo>();
    key_value->keys = {NewValueNode(prim::kPrimMakeTuple)};
    key_value->values = {NewValueNode(prim::kPrimMakeTuple)};
    size_t index_begin = 2;
    constexpr auto end_num = 2;
    size_t index_end = inputs.size() - end_num;
    size_t size_if_empty = 4;
    std::string exception_type = raiseutils::GetExceptionType(inputs[1]->abstract(), inputs[index_end], key_value);
    std::string exception_string;
    // Process raise ValueError()
    if (inputs.size() == size_if_empty) {
      std::string key = raiseutils::MakeRaiseKey(key_value->num_str);
      (void)key_value->keys.emplace_back(NewValueNode(std::make_shared<StringImm>(key)));
      (void)key_value->values.emplace_back(NewValueNode(std::make_shared<StringImm>("")));
      exception_string = key;
    }
    // Processed in units of nodes. Raise ValueError(xxxx)
    for (size_t index = index_begin; index < index_end; ++index) {
      const auto input = inputs[index];
      auto input_abs = input->abstract();
      MS_EXCEPTION_IF_NULL(input_abs);
      const bool need_symbol = raiseutils::CheckNeedSymbol(input_abs);
      if (need_symbol) {
        exception_string += "'";
      }
      bool need_comma = !IsPrimitiveCNode(input, prim::kPrimMakeTuple);
      exception_string += raiseutils::GetExceptionString(input_abs, input, key_value, need_symbol, need_comma);
      if (need_symbol) {
        exception_string += "'";
      }
      if (index != inputs.size() - 1) {
        exception_string += ", ";
      }
    }
    bool need_out_symbol = inputs.size() > 5;
    if (need_out_symbol) {
      exception_string = "(" + exception_string + ")";
    }
    // Condition has variable but script does not.
    if (key_value->keys.size() <= 1) {
      std::string key = raiseutils::MakeRaiseKey(key_value->num_str);
      (void)key_value->keys.emplace_back(NewValueNode(std::make_shared<StringImm>(key)));
      (void)key_value->values.emplace_back(NewValueNode(std::make_shared<StringImm>(exception_string)));
      exception_string = key;
    }
    // Build PyExecute node for raise
    const std::string error_msg =
      "__import__('mindspore').common._utils._jit_fallback_raise_func(" + exception_type + "," + exception_string + ")";
    const auto script_str = std::make_shared<StringImm>(error_msg);
    // Pack local parameter keys
    const auto key_value_name_tuple = fg->NewCNodeInOrder(key_value->keys);
    // Pack local parameter values
    const auto key_value_tuple = fg->NewCNodeInOrder(key_value->values);
    // Build the PyExecute node for raise error.
    const auto raise_pyexecute_node = fallback::CreatePyExecuteCNodeInOrder(
      fg, NewValueNode(script_str), key_value_name_tuple, key_value_tuple, cnode->debug_info());
    raise_pyexecute_node->add_input(inputs[inputs.size() - 1]);
    auto old_abs = cnode->abstract();
    MS_EXCEPTION_IF_NULL(old_abs);
    const auto &type = old_abs->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    fallback::SetRealType(raise_pyexecute_node, type);
    MS_LOG(DEBUG) << "Raise convert to PyExecute node: " << raise_pyexecute_node->DebugString();
    return raise_pyexecute_node;
  }

  // ScalarCast(x, dtype) --> PyExecute(string, keys, values)
  AnfNodePtr ConvertScalarCast(const CNodePtr &cnode) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime) {
      return nullptr;
    }
    constexpr size_t x_index = 1;
    constexpr size_t dtype_index = 2;
    auto x_node = cnode->input(x_index);
    auto dtype_node = cnode->input(dtype_index);
    auto x_abs = GetAbstract<abstract::AbstractAny>(x_node);
    if (x_abs == nullptr) {
      return nullptr;
    }
    auto dtype_abs = GetAbstract<abstract::AbstractType>(dtype_node);
    MS_EXCEPTION_IF_NULL(dtype_abs);
    auto dtype_val = dtype_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(dtype_val);
    auto scalar_type = dtype_val->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(scalar_type);
    std::string target_type_str;
    auto type_id = NormalizeTypeId(scalar_type->type_id());
    if (type_id == kNumberTypeInt) {
      target_type_str = "int";
    } else if (type_id == kNumberTypeFloat) {
      target_type_str = "float";
    } else if (type_id == kNumberTypeBool) {
      target_type_str = "bool";
    } else {
      MS_LOG(EXCEPTION) << "Unsupported type: " << scalar_type->ToString();
    }

    const auto &fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    std::string internal_scalar_arg_str = "__internal_scalar_arg__";
    std::string script = target_type_str + "(" + internal_scalar_arg_str + ")";
    auto script_node = NewValueNode(std::make_shared<StringImm>(script));
    auto arg_name_node = NewValueNode(std::make_shared<StringImm>(internal_scalar_arg_str));
    auto keys_tuple_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), arg_name_node});
    auto values_tuple_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), x_node});
    keys_tuple_node->set_debug_info(cnode->debug_info());
    values_tuple_node->set_debug_info(cnode->debug_info());
    auto scalar_cast_node =
      fallback::CreatePyExecuteCNodeInOrder(cnode, script_node, keys_tuple_node, values_tuple_node);
    MS_LOG(DEBUG) << "Convert CastToScalar: " << cnode->DebugString() << " -> " << scalar_cast_node->DebugString();
    return scalar_cast_node;
  }

  AnfNodePtr ConvertMakeSlice(const CNodePtr &cnode) const {
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    MS_LOG(DEBUG) << " make_slice node: " << cnode->DebugString();
    constexpr size_t slice_size = 4;
    if (cnode->size() != slice_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "The size of input to make_slice should be " << slice_size << ", but got "
                                 << cnode->size();
    }
    constexpr size_t start_index = 1;
    constexpr size_t stop_index = 2;
    constexpr size_t step_index = 3;
    bool is_start_none = IsValueNode<None>(cnode->input(start_index));
    bool is_stop_none = IsValueNode<None>(cnode->input(stop_index));
    bool is_step_none = IsValueNode<None>(cnode->input(step_index));
    auto start_str = is_start_none ? "None" : "__start__";
    auto stop_str = is_stop_none ? "None" : "__stop__";
    auto step_str = is_step_none ? "None" : "__step__";
    // Script
    std::stringstream script_buffer;
    script_buffer << "slice(" << start_str << ", " << stop_str << ", " << step_str << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Pack local parameters keys and values.
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    if (!is_start_none) {
      (void)key_value_names_list.emplace_back(NewValueNode(start_str));
      (void)key_value_list.emplace_back(cnode->input(start_index));
    }
    if (!is_stop_none) {
      (void)key_value_names_list.emplace_back(NewValueNode(stop_str));
      (void)key_value_list.emplace_back(cnode->input(stop_index));
    }
    if (!is_step_none) {
      (void)key_value_names_list.emplace_back(NewValueNode(step_str));
      (void)key_value_list.emplace_back(cnode->input(step_index));
    }
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);
    const auto key_value_tuple = fg->NewCNode(key_value_list);

    // Build the new slice node.
    const auto slice_node =
      fallback::CreatePyExecuteCNodeInOrder(cnode, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);
    MS_LOG(DEBUG) << "Made slice node: " << slice_node->DebugString();
    return slice_node;
  }

  // Only process the node that have a PyExecute node(the abstract is AbstractAny).
  bool CheckInputsHasAnyType(const CNodePtr &cnode) const {
    bool exist_any_type = false;
    for (const auto &input : cnode->inputs()) {
      auto input_abs = input->abstract();
      if (fallback::ContainsSequenceAnyType(input_abs)) {
        exist_any_type = true;
        break;
      }
    }
    return exist_any_type;
  }

  AnfNodePtr ConvertIsInstance(const CNodePtr &cnode) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime) {
      return nullptr;
    }
    const auto &fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    if (!CheckInputsHasAnyType(cnode)) {
      return nullptr;
    }
    const auto &prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    string name = prim->name();
    auto pyexecute_node = fallback::ConvertCNodeToPyExecuteForPrim(cnode, name);
    MS_LOG(DEBUG) << "Convert: " << cnode->DebugString() << " -> " << pyexecute_node->DebugString();
    return pyexecute_node;
  }

  // JoinedStr(XXXXXX)
  // TO
  // A = PyExecute("list(map(str, __inner_convert_object__), ("__inner_convert_object__",), ((XXXXXX,),)")
  // B = PyExecute("".join(__inner_str_list__)", ("__inner_str_list__",), (A,)).
  // replace(B --> JoinedStr)
  AnfNodePtr ConvertJoinedStr(const CNodePtr &cnode) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime) {
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    MS_LOG(DEBUG) << " make_slice node: " << cnode->DebugString();
    // Convert all node to list[str]
    constexpr auto kConvertToListString = "list(map(str, __inner_convert_object__))";
    constexpr auto kConvertToListKey = "__inner_convert_object__";
    std::vector<AnfNodePtr> list_str_value_list = {NewValueNode(prim::kPrimMakeTuple)};
    (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(list_str_value_list),
                         [&fg](const AnfNodePtr &node) {
                           auto abs = node->abstract();
                           if (abs == nullptr || !abs->isa<abstract::AbstractList>()) {
                             return node;
                           }
                           // Do not handle nested sequence with list element now.
                           const std::string input_str = "__internal_list_input__";
                           const std::string func_str = "list";
                           const std::string script_str = func_str + "(" + input_str + ")";
                           const auto script_str_val = std::make_shared<StringImm>(script_str);
                           const auto input_str_val = std::make_shared<StringImm>(input_str);
                           auto key_str_val = std::make_shared<ValueTuple>(std::vector<ValuePtr>{input_str_val});
                           auto key_val_node = fg->NewCNode({std::make_shared<ValueNode>(prim::kPrimMakeTuple), node});
                           AnfNodePtr ret =
                             fallback::CreatePyExecuteCNode(fg, NewValueNode(script_str_val), NewValueNode(key_str_val),
                                                            key_val_node, node->debug_info());
                           return ret;
                         });

    std::vector<AnfNodePtr> list_str_key_list = {NewValueNode(prim::kPrimMakeTuple), NewValueNode(kConvertToListKey)};
    auto list_str_key_node = fg->NewCNode(list_str_key_list);
    auto list_str_value_node = fg->NewCNode(list_str_value_list);
    auto convet_list_str_node = fallback::CreatePyExecuteCNodeInOrder(
      fg, NewValueNode(kConvertToListString), list_str_key_node,
      fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), list_str_value_node}), cnode->debug_info());

    // change to string.
    constexpr auto eval_string_script = "\"\".join(__inner_str_list__)";
    constexpr auto eval_key_string = "__inner_str_list__";
    auto eval_key_node = fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), NewValueNode(eval_key_string)});
    auto eval_value_node = fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), convet_list_str_node});

    auto joined_result_node = fallback::CreatePyExecuteCNode(fg, NewValueNode(eval_string_script), eval_key_node,
                                                             eval_value_node, cnode->debug_info());
    return joined_result_node;
  }

  AnfNodePtr ConvertPrint(const CNodePtr &cnode) const {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime) {
      return nullptr;
    }
    const auto &fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    if (!CheckInputsHasAnyType(cnode)) {
      return nullptr;
    }
    // Skip the io_monad input
    auto inputs = cnode->inputs();
    if (!HasAbstractMonad(inputs.back())) {
      MS_LOG(EXCEPTION) << "The print node has no monad input:" << cnode->DebugString();
    }
    inputs.pop_back();
    auto no_io_print = fg->NewCNode(inputs);
    auto pyexecute_node = fallback::ConvertCNodeToPyExecuteForPrim(no_io_print, "print");

    // Add io_monad input
    auto new_pyexecute_inputs = pyexecute_node->cast<CNodePtr>()->inputs();
    (void)new_pyexecute_inputs.emplace_back(cnode->inputs().back());
    auto new_pyexecute_node = fg->NewCNode(new_pyexecute_inputs);
    MS_LOG(DEBUG) << "Convert: " << cnode->DebugString() << " -> " << new_pyexecute_node->DebugString();
    return new_pyexecute_node;
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &) const;
  using ConverterMap = std::unordered_map<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
    // SparseProcess: 1.MakeSparse->MakeTuple 2.SparseGetAttr->TupleGetItem
    {prim::kPrimMakeRowTensor, &ThisClass::ConvertMakeSparseToMakeTuple},
    {prim::kPrimRowTensorGetIndices, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimRowTensorGetValues, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimRowTensorGetDenseShape, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimMakeCSRTensor, &ThisClass::ConvertMakeSparseToMakeTuple},
    {prim::kPrimCSRTensorGetIndptr, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCSRTensorGetIndices, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCSRTensorGetValues, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCSRTensorGetDenseShape, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimMakeCOOTensor, &ThisClass::ConvertMakeSparseToMakeTuple},
    {prim::kPrimCOOTensorGetIndices, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCOOTensorGetValues, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCOOTensorGetDenseShape, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimDictGetItem, &ThisClass::ConvertDictGetItem},
    {prim::kPrimDictSetItem, &ThisClass::ConvertDictSetItem},
    {prim::kPrimListInplaceExtend, &ThisClass::ConvertListInplaceExtend},
    {prim::kPrimListInplaceInsert, &ThisClass::ConvertListInplaceInsert},
    {prim::kPrimListInplacePop, &ThisClass::ConvertListInplacePop},
    {prim::kPrimListInplaceReverse, &ThisClass::ConvertListInplaceReverse},
    {prim::kPrimListInplaceClear, &ThisClass::ConvertListInplaceClear},
    {prim::kPrimMakeDict, &ThisClass::ConvertMakeDict},
    {prim::kPrimRaise, &ThisClass::ConvertRaise},
    {prim::kPrimScalarCast, &ThisClass::ConvertScalarCast},
    {prim::kPrimMakeSlice, &ThisClass::ConvertMakeSlice},
    {prim::kPrimIsInstance, &ThisClass::ConvertIsInstance},
    {prim::kPrimJoinedStr, &ThisClass::ConvertJoinedStr},
    {prim::kPrimPrint, &ThisClass::ConvertPrint}};

  // Convert ValueNode<None> to PyExecute("None", ("None"), ("None")).
  AnfNodePtr ConvertNoneToPyExecute(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto str_value = std::make_shared<StringImm>("None");
    auto script_node = NewValueNode(str_value);

    std::vector<ValuePtr> none_value{str_value};
    const auto none_tuple = std::make_shared<ValueTuple>(none_value);
    auto none_tuple_node = NewValueNode(none_tuple);
    AbstractBasePtrList abs_list{std::make_shared<abstract::AbstractScalar>(MakeValue("None"))};
    none_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));

    AnfNodePtr none_execute_node = fallback::CreatePyExecuteCNodeInOrder(
      func_graph, script_node, none_tuple_node, none_tuple_node, none_tuple_node->debug_info());
    MS_LOG(DEBUG) << "none_execute_node:" << none_execute_node->DebugString();

    set_need_renormalized(true);
    return none_execute_node;
  }

  AnfNodePtr GetPyExecuteFromValueSequence(const FuncGraphPtr &fg, const ValueNodePtr &value_node,
                                           const ValueSequencePtr &value_sequence, const PrimitivePtr &prim,
                                           bool py_execute_input) {
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.reserve(value_sequence->size());
    (void)new_inputs.emplace_back(NewValueNode(prim));
    bool changed = false;
    auto abs = value_node->abstract();
    if (abs == nullptr) {
      for (const auto &v : value_sequence->value()) {
        auto v_node = NewValueNode(v);
        v_node->set_debug_info(value_node->debug_info());
        auto new_node = GetPyExecuteFromValue(fg, v_node, v, py_execute_input);
        new_node->set_debug_info(value_node->debug_info());
        (void)new_inputs.emplace_back(new_node);
        if (new_node != v_node) {
          changed = true;
        }
      }
    } else {
      auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(abs_seq);
      const auto &abs_seq_elements = abs_seq->elements();
      const auto &value_sequence_values = value_sequence->value();
      if (abs_seq_elements.size() != value_sequence_values.size()) {
        MS_LOG(EXCEPTION) << "The size of value sequence should be same as the size of abstract sequence.";
      }
      for (size_t i = 0; i < value_sequence_values.size(); ++i) {
        auto v = value_sequence_values[i];
        auto v_node = NewValueNode(v);
        v_node->set_debug_info(value_node->debug_info());
        v_node->set_abstract(abs_seq_elements[i]);
        auto new_node = GetPyExecuteFromValue(fg, v_node, v, py_execute_input);
        new_node->set_debug_info(value_node->debug_info());
        (void)new_inputs.emplace_back(new_node);
        if (new_node != v_node) {
          changed = true;
        }
      }
    }
    if (changed) {
      auto ret = fg->NewCNode(new_inputs);
      ret->set_abstract(value_node->abstract());
      return ret;
    }
    return value_node;
  }

  AnfNodePtr ConvertTypeToPyExecute(const FuncGraphPtr &fg, const ValueNodePtr &node, const TypePtr &type) const {
    // Support convert type to PyExecute.
    const auto py_type = ValueToPyData(type);
    MS_LOG(DEBUG) << "py_type: " << py_type;
    auto res = fallback::ConvertPyObjectToPyExecute(fg, py::str(py_type).cast<std::string>(), py_type, node, false);
    fallback::SetRealType(res, type);
    return res;
  }

  AnfNodePtr ConvertClassTypeToPyExecute(const FuncGraphPtr &fg, const ValueNodePtr &node,
                                         const ClassTypePtr &class_type) const {
    // Support convert class type to PyExecute.
    const auto py_type = ValueToPyData(class_type);
    MS_LOG(DEBUG) << "py_type: " << py_type;
    auto res = fallback::ConvertPyObjectToPyExecute(fg, py::str(py_type).cast<std::string>(), py_type, node, true);
    fallback::SetRealType(res, class_type);
    MS_LOG(DEBUG) << "res: " << res->DebugString();
    return res;
  }

  AnfNodePtr GetPyExecuteFromValue(const FuncGraphPtr &fg, const ValueNodePtr &value_node, const ValuePtr &value,
                                   bool py_execute_input) {
    MS_EXCEPTION_IF_NULL(fg);
    MS_EXCEPTION_IF_NULL(value_node);
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<None>()) {
      constexpr auto vmap_prefix = "VmapRule";
      if (value_node->scope() != nullptr &&
          value_node->scope()->name().compare(0, strlen(vmap_prefix), vmap_prefix) == 0) {
        return value_node;
      }
      return ConvertNoneToPyExecute(fg);
    }
    if (vm_pipeline_ && MsContext::GetInstance()->GetJitSyntaxLevel() == kLax) {
      if (value->isa<Type>()) {
        return ConvertTypeToPyExecute(fg, value_node, value->cast<TypePtr>());
      } else if (value->isa<parse::ClassType>()) {
        auto class_type = GetValueNode<ClassTypePtr>(value_node);
        MS_EXCEPTION_IF_NULL(class_type);
        return ConvertClassTypeToPyExecute(fg, value_node, class_type);
      }
    }
    if (value->isa<parse::MsClassObject>()) {
      return fallback::ConvertMsClassObjectToPyExecute(fg, value, value_node);
    }
    if (value->isa<parse::InterpretedObject>()) {
      const auto interpreted_value = dyn_cast<parse::InterpretedObject>(value);
      const std::string &key = interpreted_value->name();
      return fallback::ConvertPyObjectToPyExecute(fg, key, interpreted_value->obj(), value_node, true);
    }
    if (value->isa<ValueTuple>()) {
      return GetPyExecuteFromValueSequence(fg, value_node, value->cast<ValueSequencePtr>(), prim::kPrimMakeTuple,
                                           py_execute_input);
    }
    if (value->isa<ValueList>()) {
      if (!fallback::EnableFallbackList()) {
        if (py_execute_input) {
          return ValueListConvertPyExecute(fg, value_node, value);
        } else {
          return GetPyExecuteFromValueSequence(fg, value_node, value->cast<ValueSequencePtr>(), prim::kPrimMakeList,
                                               py_execute_input);
        }
      }
      return RebuildValueList(fg, value_node);
    }
    if (value->isa<ValueDictionary>()) {
      return RebuildValueDict(fg, value_node, value->cast<ValueDictionaryPtr>());
    }
    return value_node;
  }

  void ConvertValueInputToPyExecute(const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (!allow_fallback_runtime) {
      return;
    }
    const PrimitiveSet convert_prim_set{
      prim::kPrimPyExecute,         prim::kPrimListInplaceAppend, prim::kPrimListInplaceReverse,
      prim::kPrimListInplaceExtend, prim::kPrimListInplaceInsert, prim::kPrimListInplacePop,
    };
    if (AnfUtils::IsRealKernel(cnode) && !IsOneOfPrimitiveCNode(cnode, convert_prim_set)) {
      return;
    }
    const auto &inputs = cnode->inputs();
    auto cur_func = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_func);
    for (const auto &input : inputs) {
      auto value_node = dyn_cast<ValueNode>(input);
      if (value_node == nullptr) {
        continue;
      }
      const auto &value = value_node->value();
      if (vm_pipeline_ && MsContext::GetInstance()->GetJitSyntaxLevel() == kLax) {
        // Not convert the 'type' used by Cast primitive.
        if (value->isa<Type>() && IsPrimitiveCNode(cnode, prim::kPrimCast)) {
          continue;
        }
      }
      auto debug_info = value_node->debug_info();
      auto location_info = trace::GetDebugInfo(debug_info, std::string(""));
      if (location_info.empty()) {
        value_node->set_debug_info(cnode->debug_info());
      }
      auto new_input = GetPyExecuteFromValue(cur_func, value_node, value, false);
      if (new_input == input) {
        continue;
      }
      new_input->set_debug_info(value_node->debug_info());
      (void)manager_->Replace(input, new_input);
      set_need_renormalized(true);
    }
  }

  AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode) override {
    // Get primitive from cnode.
    const auto &prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      return nullptr;
    }
    ConvertValueInputToPyExecute(cnode);

    // Find cnode converter by primitive.
    auto iter = converters_.find(prim);
    if (iter == converters_.end()) {
      return nullptr;
    }
    // Call converter.
    return (this->*(iter->second))(cnode);
  }

  AnfNodePtr ValueListConvertPyExecute(const FuncGraphPtr &func_graph, const ValueNodePtr &value_node,
                                       const ValuePtr &value) {
    MS_EXCEPTION_IF_NULL(value);
    MS_EXCEPTION_IF_NULL(func_graph);
    auto value_list = value->cast<ValueListPtr>();
    MS_EXCEPTION_IF_NULL(value_list);
    auto values = value_list->value();

    // Script and local parameters keys
    constexpr auto internal_element_str_prefix = "__internal_list_element_";
    std::stringstream script_buffer;
    script_buffer << "list((";
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < values.size(); ++i) {
      std::string element_name_str = internal_element_str_prefix + std::to_string(i) + "__";
      script_buffer << element_name_str << ", ";
      const auto element_name = std::make_shared<StringImm>(element_name_str);
      (void)key_value_names_list.emplace_back(NewValueNode(element_name));
    }
    script_buffer << "))";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    const auto key_value_name_tuple = func_graph->NewCNode(key_value_names_list);

    // Pack the local parameters values, not support list, tuple, or dict.
    auto abs_list = dyn_cast<abstract::AbstractList>(value_node->abstract());
    MS_EXCEPTION_IF_NULL(abs_list);
    const auto &abs_list_elements = abs_list->elements();
    if (abs_list_elements.size() != values.size()) {
      MS_LOG(EXCEPTION) << "The size of value list should be same as the size of abstract list.";
    }
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < values.size(); ++i) {
      auto element = values[i];
      auto element_vnode = NewValueNode(element);
      element_vnode->set_debug_info(value_node->debug_info());
      element_vnode->set_abstract(abs_list_elements[i]);
      auto converted_element = GetPyExecuteFromValue(func_graph, element_vnode, element, true);
      converted_element->set_debug_info(value_node->debug_info());
      (void)key_value_list.emplace_back(converted_element);
    }
    const auto key_value_tuple = func_graph->NewCNode(key_value_list);

    // Build the new dict node.
    const auto list_value_node = fallback::CreatePyExecuteCNodeInOrder(
      func_graph, NewValueNode(script_str), key_value_name_tuple, key_value_tuple, value_node->debug_info());
    list_value_node->set_abstract(value_node->abstract());
    MS_LOG(DEBUG) << "List value node convert to PyExecute node: " << list_value_node->DebugString();
    return list_value_node;
  }

  AnfNodePtr PackDictValue(const FuncGraphPtr &fg, const ValueNodePtr &value_node, const ValueDictionaryPtr &dict) {
    const auto &keys_values = dict->value();
    auto abs_dict = dyn_cast<abstract::AbstractDictionary>(value_node->abstract());
    const auto &abs_keys_values = abs_dict->elements();
    if (keys_values.size() != abs_keys_values.size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The size of value dict should be same as the size of abstract dict.";
    }
    std::vector<AnfNodePtr> value_list{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < keys_values.size(); ++i) {
      auto key_value = keys_values[i];
      auto new_vnode = NewValueNode(key_value.second);
      new_vnode->set_debug_info(value_node->debug_info());
      new_vnode->set_abstract(abs_keys_values[i].second);
      auto iter_value = GetPyExecuteFromValue(fg, new_vnode, key_value.second, true);
      iter_value->set_debug_info(value_node->debug_info());
      (void)value_list.emplace_back(iter_value);
    }
    auto value_tuple_node = fg->NewCNode(value_list);
    return value_tuple_node;
  }

  // dict(k0:v0, k1:v1, ...) --> PyExecute('dict(zip(keys, values))', ...)
  AnfNodePtr RebuildValueDict(const FuncGraphPtr &fg, const ValueNodePtr &value_node, const ValueDictionaryPtr &dict) {
    const auto &keys_values = dict->value();

    // Local parameters values.
    // Pack the key tuple.
    std::vector<ValuePtr> key_list;
    key_list.reserve(keys_values.size());
    for (const auto &key_value : keys_values) {
      (void)key_list.emplace_back(key_value.first);
    }
    const auto key_tuple = std::make_shared<ValueTuple>(key_list);
    auto key_tuple_node = NewValueNode(key_tuple);
    key_tuple_node->set_debug_info(value_node->debug_info());
    // Pack the value tuple.
    auto value_tuple_node = PackDictValue(fg, value_node, dict);

    // Generate Make Dict PyExecute Node value
    auto make_key_tuple_node = ConstructInternalTupleKeysNode(fg, key_tuple_node);
    auto make_value_tuple_node = ConstructInternalTupleValueNode(fg, value_tuple_node);

    auto make_dict_node = ConstructNewDictNode(fg, make_key_tuple_node, make_value_tuple_node);
    make_dict_node->set_debug_info(value_node->debug_info());
    return make_dict_node;
  }

  AnfNodePtr RebuildValueList(const FuncGraphPtr &fg, const ValueNodePtr &value_node) const {
    MS_EXCEPTION_IF_NULL(fg);

    auto abs = value_node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(list_abs);

    py::list list_object =
      list_abs->has_list_py_obj() ? *(list_abs->list_py_obj<py::list>()) : ValueToPyData(value_node->value());

    // Generate PyExecute node: __list_object__
    const std::string list_obj_str_prefix = "__list_py_object_";
    auto list_obj_id = fallback::GetPyObjectPtrStr(list_object);
    MS_LOG(DEBUG) << "Current python object id: " << list_obj_id;
    auto list_obj_str = list_obj_str_prefix + list_obj_id + "_";
    auto res = fallback::ConvertPyObjectToPyExecute(fg, list_obj_str, list_object, value_node, false);

    // Set real type, shape and corresponding list python object.
    fallback::SetRealType(res, list_abs->BuildType());
    fallback::SetRealShape(res, list_abs->BuildShape());
    fallback::SetPySeqObject<AnfNode, py::list>(res, std::make_shared<py::list>(list_object));
    return res;
  }

  AnfNodePtr ConvertInterpretedObjectValue(const ValueNodePtr &node, const parse::InterpretedObjectPtr &value) const {
    // Convert InterpretedObject value node to PyExecute CNode.
    const auto interpreted_value = dyn_cast<parse::InterpretedObject>(value);
    const std::string &key = interpreted_value->name();
    return fallback::ConvertPyObjectToPyExecute(root_graph_, key, interpreted_value->obj(), node, true);
  }

  AnfNodePtr ConvertValueNode(const ValueNodePtr &value_node, const ValuePtr &value) override {
    const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() >= kCompatible);
    if (allow_fallback_runtime) {
      if (value->isa<ValueDictionary>()) {
        return RebuildValueDict(root_graph_, value_node, value->cast<ValueDictionaryPtr>());
      } else if (value->isa<parse::InterpretedObject>()) {
        return ConvertInterpretedObjectValue(value_node, value->cast<parse::InterpretedObjectPtr>());
      } else if (value->isa<parse::MsClassObject>()) {
        return fallback::ConvertMsClassObjectToPyExecute(root_graph_, value, value_node);
      }
    }
    return nullptr;
  }

  // AbstractRowTensor --> AbstractTuple.
  static AbstractBasePtr ConvertToAbstractTuple(const AbstractBasePtr &abs, size_t depth) {
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(INTERNAL_EXCEPTION) << "List, tuple and dict nesting is not allowed more than " << kMaxSeqRecursiveDepth
                                 << " levels.";
    }
    // Convert RowTensor in AbstractSequence to AbstractTuple.
    auto abs_seq = abs->cast<AbstractSequencePtr>();
    if (abs_seq != nullptr) {
      // Dynamic length sequence do not convert.
      if (abs_seq->dynamic_len()) {
        return nullptr;
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
        // If no RowTensor in sequence is changed, do not convert.
        return nullptr;
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
      if (abs_seq->isa<AbstractList>()) {
        return std::make_shared<AbstractList>(std::move(elements));
      }
      return std::make_shared<AbstractTuple>(std::move(elements));
    }
    // AbstractRowTensor --> AbstractTuple.
    auto abs_row_tensor = abs->cast<std::shared_ptr<AbstractRowTensor>>();
    if (abs_row_tensor != nullptr) {
      std::vector<AbstractBasePtr> elements{abs_row_tensor->indices(), abs_row_tensor->values(),
                                            abs_row_tensor->dense_shape()};
      return std::make_shared<AbstractTuple>(std::move(elements));
    }
    return nullptr;
  }

  AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) override {
    // AbstractSequence, AbstractDict, AbstractRowTensor --> AbstractTuple.
    return ConvertToAbstractTuple(abs, 0);
  }

 private:
  bool vm_pipeline_{true};
};

// ===========================================================================
// ExportRewriter convert List to Tuple, used for export.
// ===========================================================================
class ExportRewriter : public BaseRewriter {
 public:
  using ThisClass = ExportRewriter;
  ExportRewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager)
      : BaseRewriter(root_graph, manager) {}
  ~ExportRewriter() override = default;

 protected:
  // From:
  //   MakeList(arg1, arg2, ...)
  // To:
  //   MakeTuple(arg1, arg2, ...)
  AnfNodePtr ConvertMakeListToMakeTuple(const CNodePtr &node) {
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
  AnfNodePtr ConvertListGetItemToTupleGetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [list_getitem, list, item]
    constexpr size_t expect_input_size = 3;
    CheckInputsSize(node, expect_input_size);
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
  AnfNodePtr ConvertListSetItemToTupleSetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [list_setitem, list, index, item]
    const size_t expect_inputs_size = 4;
    CheckInputsSize(node, expect_inputs_size);

    const size_t data_index = 1;
    const size_t cons_index = 2;
    const size_t value_index = 3;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[cons_index];
    auto &value = inputs[value_index];
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, key, value});
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &);
  using ConverterMap = mindspore::HashMap<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
    {prim::kPrimMakeList, &ThisClass::ConvertMakeListToMakeTuple},
    {prim::kPrimListGetItem, &ThisClass::ConvertListGetItemToTupleGetItem},
    {prim::kPrimListSetItem, &ThisClass::ConvertListSetItemToTupleSetItem},
  };
  AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode) override {
    // Get primitive from cnode.
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      return nullptr;
    }
    // Find cnode converter by primitive.
    auto iter = converters_.find(prim);
    if (iter == converters_.end()) {
      return nullptr;
    }
    // Call converter.
    return (this->*(iter->second))(cnode);
  }

  static ValuePtr ConvertValueSequenceToValueTuple(const ValuePtr &value, size_t depth, bool *need_convert) {
    MS_EXCEPTION_IF_NULL(need_convert);
    MS_EXCEPTION_IF_NULL(value);
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(INTERNAL_EXCEPTION) << "List nesting is not allowed more than " << kMaxSeqRecursiveDepth << " levels.";
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

  AnfNodePtr ConvertValueNode(const ValueNodePtr &value_node, const ValuePtr &value) override {
    bool need_convert = false;
    auto convert_value = ConvertValueSequenceToValueTuple(value, 0, &need_convert);
    if (need_convert) {
      return std::make_shared<ValueNode>(convert_value);
    }
    return nullptr;
  }

  // AbstractList --> AbstractTuple.
  static AbstractBasePtr ConvertToAbstractTuple(const AbstractBasePtr &abs, size_t depth) {
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(INTERNAL_EXCEPTION) << "List or Dict nesting is not allowed more than " << kMaxSeqRecursiveDepth
                                 << " levels.";
    }
    // AbstractList --> AbstractTuple.
    auto abs_seq = abs->cast<AbstractSequencePtr>();
    if (abs_seq != nullptr) {
      if (abs_seq->dynamic_len() && abs_seq->isa<AbstractList>()) {
        auto converted_abs_tuple = std::make_shared<AbstractTuple>(abs_seq->elements(), abs_seq->sequence_nodes());
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
        if (abs->isa<AbstractTuple>()) {
          // If no elements changed and it is an AbstractTuple, do not convert.
          return nullptr;
        }
        // If no elements changed but it is not an AbstractTuple, convert it by copy elements.
        return std::make_shared<AbstractTuple>(seq_elements);
      }
      // Always make new AbstractTuple when elements changed.
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
      return std::make_shared<AbstractTuple>(std::move(elements));
    }
    return nullptr;
  }

  AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) override {
    // AbstractList --> AbstractTuple.
    return ConvertToAbstractTuple(abs, 0);
  }
};
}  // namespace

bool RewriterBeforeOptA(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);
  BeforeOptARewriter rewriter(root, manager);
  return rewriter.Execute();
}

bool RewriterAfterOptA(const FuncGraphPtr &root, const pipeline::ResourcePtr &resource, bool vm_pipeline) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(resource);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);
  AfterOptARewriter rewriter(root, manager, vm_pipeline);
  bool change = rewriter.Execute();
  if (rewriter.need_renormalized()) {
    abstract::AbstractBasePtrList new_args_spec;
    (void)std::transform(root->parameters().begin(), root->parameters().end(), std::back_inserter(new_args_spec),
                         [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
    (void)pipeline::Renormalize(resource, root, new_args_spec);
  }
  return change;
}

bool RewriterForExport(const FuncGraphPtr &root, const pipeline::ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(resource);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);
  ExportRewriter rewriter(root, manager);
  return rewriter.Execute();
}

static inline bool OrderPyExecuteCNode(const FuncGraphPtr &graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(graph);
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);
  CNodePtr former_node = nullptr;
  CNodePtr latter_node = nullptr;
  bool change = false;
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimPyExecute) || node->func_graph() != graph) {
      continue;
    }
    if (former_node == nullptr) {
      former_node = dyn_cast<CNode>(node);
      continue;
    } else {
      latter_node = dyn_cast<CNode>(node);
    }
    MS_EXCEPTION_IF_NULL(former_node);
    MS_EXCEPTION_IF_NULL(latter_node);

    // Make former node as latter node's input.
    auto tr = manager->Transact();
    size_t latest_index = latter_node->size() - 1;
    const auto &last_input_abs = latter_node->input(latest_index)->abstract();
    if (last_input_abs != nullptr && last_input_abs->isa<abstract::AbstractMonad>()) {  // Should be IO monad.
      const auto &monad_node = latter_node->input(latest_index);
      tr.SetEdge(latter_node, latest_index, former_node);
      tr.AddEdge(latter_node, monad_node);
    } else {
      tr.AddEdge(latter_node, former_node);
    }
    tr.Commit();

    former_node = latter_node;
    change = true;
  }
  return change;
}

bool OrderPyExecuteAfterRewriter(const FuncGraphPtr &root, const pipeline::ResourcePtr &resource) {
  auto manager = resource->manager();
  const auto func_graphs_used_total = root->func_graphs_used_total();
  bool change = false;
  for (const auto &fg : func_graphs_used_total) {
    auto cur_change = OrderPyExecuteCNode(fg, manager);
    change = change || cur_change;
  }
  bool root_change = OrderPyExecuteCNode(root, manager);
  change = change || root_change;
  if (change) {
    abstract::AbstractBasePtrList new_args_spec;
    (void)std::transform(root->parameters().begin(), root->parameters().end(), std::back_inserter(new_args_spec),
                         [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
    (void)pipeline::Renormalize(resource, root, new_args_spec);
  }
  return change;
}
}  // namespace opt
}  // namespace mindspore
