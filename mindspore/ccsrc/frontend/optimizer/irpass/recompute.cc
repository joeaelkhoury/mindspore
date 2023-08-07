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

#include "frontend/optimizer/irpass/recompute.h"
#include <set>
#include <unordered_map>

namespace mindspore {
namespace opt {
namespace irpass {
bool EnableGraphReuse() {
  static const auto cell_reuse_env = common::GetEnv("MS_DEV_CELL_REUSE");
  static const auto cell_reuse_enable = cell_reuse_env == "1" || cell_reuse_env == "2";
  return cell_reuse_enable;
}

bool HasBpropGetter(const OptimizerPtr &opt, const AnfNodePtr &k_fg_caller) {
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users = manager->node_users();
  auto iter = node_users.find(k_fg_caller);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "The node " << k_fg_caller->DebugString() << " should have users.";
  }

  return std::any_of(iter->second.begin(), iter->second.end(), [](const std::pair<AnfNodePtr, int> &node_and_idx) {
    auto user = node_and_idx.first;
    return IsPrimitiveCNode(user, prim::kPrimTupleGetItem) &&
           common::AnfAlgo::GetTupleGetItemOutIndex(user->cast<CNodePtr>()) == 1;
  });
}

AnfNodePtr GetBpropCaller(const FuncGraphManagerPtr &manager, const AnfNodePtr &bprop_getter) {
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users = manager->node_users();
  auto iter = node_users.find(bprop_getter);
  if (iter == node_users.end()) {
    return nullptr;
  }
  if (iter->second.size() != 1) {
    MS_LOG(EXCEPTION) << "The number of bprop caller should be 1, but got " << iter->second.size()
                      << ", bprop_getter: " << bprop_getter->DebugString();
  }
  auto user_node_idx = iter->second.begin();
  if (user_node_idx->second != 0) {
    MS_LOG(EXCEPTION) << "The bprop_getter should be used in input 0, but got " << user_node_idx->second;
  }
  return user_node_idx->first;
}

namespace {
constexpr auto kGradientsFlag = "Gradients";

bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0;
}

bool IsRecomputeKGraphCaller(const AnfNodePtr &node) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr) {
    return false;
  }
  auto call_fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
  if (call_fg != nullptr && call_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
    return true;
  }
  return false;
}

bool IsGradNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->fullname_with_scope().compare(0, strlen(kGradientsFlag), kGradientsFlag) == 0;
}

bool AddNewPrimalNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const AnfNodePtr &origin_primal,
                      const AnfNodePtr &new_primal, bool recompute_cell = false) {
  bool changed = false;
  auto node_users = manager->node_users()[origin_primal];
  for (auto &node_and_idx : node_users) {
    auto user = node_and_idx.first;
    MS_EXCEPTION_IF_NULL(user);
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem) && (!IsGradNode(user) || recompute_cell)) {
      // Make new tuple_getitem to get corresponding output.
      auto new_primal_getitem = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_primal,
                                              user->cast_ptr<CNode>()->input(kInputNodeOutputIndexInTupleGetItem)});
      changed = AddNewPrimalNode(manager, fg, user, new_primal_getitem, recompute_cell) || changed;
      continue;
    }
    // Set edge to not recomputed primal nodes.
    if (recompute_cell || (!IsRecomputeKGraphCaller(user) && !IsGradNode(user))) {
      MS_LOG(DEBUG) << "Set edge to user: " << user->DebugString() << ", new primal: " << new_primal->DebugString();
      manager->SetEdge(user, node_and_idx.second, new_primal);
      changed = true;
    }
  }
  return changed;
}

bool IsRecomputeCell(const FuncGraphPtr &k_fg) {
  auto primal_iter = k_fg->transforms().find("primal");
  if (primal_iter == k_fg->transforms().end()) {
    MS_LOG(EXCEPTION) << "The k_fg: " << k_fg << " should have primal part.";
  }
  return primal_iter->second.func_graph() != nullptr;
}

bool HasRecomputedInput(const CNodePtr &k_fg_caller_cnode) {
  for (auto &input : k_fg_caller_cnode->inputs()) {
    if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
      return HasRecomputedInput(input->cast<CNodePtr>());
    }
    if (IsPrimitiveCNode(input, prim::kPrimDepend) && HasRecomputedInput(input->cast<CNodePtr>())) {
      return true;
    }
    // The recomputed input should be a tuple_getitem to get the forward part of recomputed k graph.
    if (!IsPrimitiveCNode(input, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto tmp = input->cast<CNodePtr>()->input(1);
    auto input_k_fg_caller = tmp;
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(tmp, prim::kPrimTupleGetItem)) {
      input_k_fg_caller = tmp->cast<CNodePtr>()->input(1);
    }

    auto cnode = dyn_cast_ptr<CNode>(input_k_fg_caller);
    if (cnode == nullptr) {
      continue;
    }
    auto call_fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    // The output of recomputed cell would not be recomputed.
    if (call_fg != nullptr && call_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH) && !IsRecomputeCell(call_fg)) {
      return true;
    }
  }
  return false;
}

AnfNodePtr GetForwardGetter(const FuncGraphManagerPtr &manager, const CNodePtr &node) {
  const auto &user_nodes = manager->node_users()[node];
  for (const auto &iter : user_nodes) {
    if (IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      if (idx != nullptr && idx->value() == 0) {
        return iter.first;
      }
    }
  }
  return nullptr;
}

AnfNodePtr GetBpropGetter(const FuncGraphManagerPtr &manager, const CNodePtr &node) {
  const auto &user_nodes = manager->node_users()[node];
  for (const auto &iter : user_nodes) {
    if (IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      if (idx != nullptr && idx->value() == 1) {
        return iter.first;
      }
    }
  }
  return nullptr;
}

bool HasRecomputedOutput(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  // The forward part may have multiple outputs.
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimDepend})) {
    const auto &user_nodes = manager->node_users()[node];
    return std::any_of(user_nodes.begin(), user_nodes.end(),
                       [&manager](const auto &iter) { return HasRecomputedOutput(manager, iter.first); });
  }
  return IsRecomputeKGraphCaller(node);
}

void GetGradUsers(const FuncGraphManagerPtr &manager, const CNodePtr &node, std::vector<AnfNodePtr> *grad_users) {
  // The forward part may have multiple outputs.
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    const auto &user_nodes = manager->node_users()[node];
    for (const auto &iter : user_nodes) {
      GetGradUsers(manager, iter.first->cast<CNodePtr>(), grad_users);
    }
    return;
  }
  if (IsGradNode(node)) {
    const auto &inputs = node->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (IsGradNode(inputs[i])) {
        (void)grad_users->emplace_back(inputs[i]);
      }
    }
  }
}

void GetDependencies(const FuncGraphManagerPtr &manager, const CNodePtr &k_fg_caller, std::set<CNodePtr> *final_nodes,
                     std::set<AnfNodePtr> *dependencies) {
  if (final_nodes->find(k_fg_caller) != final_nodes->end()) {
    return;
  }
  bool is_recompute_k_fg_caller = IsRecomputeKGraphCaller(k_fg_caller);
  // We only handle the recomputed k graph caller.
  if (!is_recompute_k_fg_caller &&
      !IsOneOfPrimitiveCNode(k_fg_caller, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimDepend})) {
    return;
  }
  if (is_recompute_k_fg_caller) {
    auto forward_getter = GetForwardGetter(manager, k_fg_caller);
    // If the k graph caller has no forward getter, it should not output to any other recomputed nodes.
    if (forward_getter == nullptr) {
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller));
      // Add the dout input of its bprop function to the dependencies.
      if (bprop_caller == nullptr) {
        return;
      }
      (void)final_nodes->emplace(k_fg_caller);
      (void)dependencies->emplace(bprop_caller->cast<CNodePtr>()->input(1));
      return;
    }
    if (!HasRecomputedOutput(manager, forward_getter)) {
      std::vector<AnfNodePtr> grad_users;
      // Add the other inputs of the grad node to the dependencies.
      GetGradUsers(manager, forward_getter->cast<CNodePtr>(), &grad_users);
      if (!grad_users.empty()) {
        for (auto &user : grad_users) {
          (void)final_nodes->emplace(k_fg_caller);
          (void)dependencies->emplace(user);
        }
        return;
      }
      // Add the dout input of its bprop function to the dependencies.
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller));
      if (bprop_caller == nullptr) {
        return;
      }
      (void)final_nodes->emplace(k_fg_caller);
      (void)dependencies->emplace(bprop_caller->cast<CNodePtr>()->input(1));
      return;
    }
  }

  const auto &user_nodes = manager->node_users()[k_fg_caller];
  for (const auto &iter : user_nodes) {
    if (IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      // Skip bprop getter.
      if (idx != nullptr && idx->value() == 1 && is_recompute_k_fg_caller) {
        continue;
      }
    }
    GetDependencies(manager, iter.first->cast<CNodePtr>(), final_nodes, dependencies);
  }
}

CNodePtr MoveKCallerToBprop(const FuncGraphManagerPtr &manager, const FuncGraphPtr &bprop_fg, const CNodePtr &node,
                            const AnfNodePtr &depend_nodes,
                            std::unordered_map<CNodePtr, CNodePtr> *origin_to_new_nodes) {
  auto iter = origin_to_new_nodes->find(node);
  if (iter != origin_to_new_nodes->end()) {
    return iter->second;
  }
  std::vector<AnfNodePtr> new_inputs;
  if (IsRecomputeKGraphCaller(node)) {
    if (!HasRecomputedInput(node)) {
      (void)std::copy(node->inputs().begin(), node->inputs().end(), std::back_inserter(new_inputs));
      new_inputs[1] = bprop_fg->NewCNode({NewValueNode(prim::kPrimDepend), new_inputs[1], depend_nodes});
    } else {
      for (auto &input : node->inputs()) {
        if (!input->isa<CNode>()) {
          (void)new_inputs.emplace_back(input);
          continue;
        }
        (void)new_inputs.emplace_back(
          MoveKCallerToBprop(manager, bprop_fg, input->cast<CNodePtr>(), depend_nodes, origin_to_new_nodes));
      }
    }
    if (IsRecomputeCell(GetValueNode<FuncGraphPtr>(node->input(0)))) {
      // Add the dout input of its bprop function to the dependencies.
      auto new_depend_nodes = depend_nodes;
      auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, node));
      if (bprop_caller != nullptr) {
        std::vector<AnfNodePtr> new_depend_nodes_inputs;
        (void)std::copy(depend_nodes->cast<CNodePtr>()->inputs().begin(),
                        depend_nodes->cast<CNodePtr>()->inputs().end(), std::back_inserter(new_depend_nodes_inputs));
        new_depend_nodes_inputs.emplace_back(bprop_caller->cast<CNodePtr>()->input(1));
        new_depend_nodes = bprop_fg->NewCNode(new_depend_nodes_inputs);
      }
      for (size_t i = 1; i < new_inputs.size(); ++i) {
        new_inputs[i] = bprop_fg->NewCNode({NewValueNode(prim::kPrimDepend), new_inputs[i], new_depend_nodes});
      }
    }
    auto new_k_fg_caller = bprop_fg->NewCNode(new_inputs);
    new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    auto primal_fg_caller = node->user_data<CNode>("primal_fg_caller");
    if (primal_fg_caller != nullptr) {
      new_k_fg_caller->set_user_data("primal_fg_caller", primal_fg_caller);
    }
    // Replace the bprop getter with the new k graph caller in bprop graph.
    auto origin_bprop_getter = GetBpropGetter(manager, node);
    if (origin_bprop_getter != nullptr) {
      auto new_bprop_getter = bprop_fg->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), new_k_fg_caller, NewValueNode(static_cast<int64_t>(1))});
      (void)manager->Replace(origin_bprop_getter, new_bprop_getter);
    }
    origin_to_new_nodes->emplace(node, new_k_fg_caller);
    return new_k_fg_caller;
  }
  // If it is not tuple_getitem, it should be node which is not set recomputed.
  if (!IsOneOfPrimitiveCNode(
        node, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple, prim::kPrimDepend, prim::kPrimUpdateState})) {
    return node;
  }
  for (auto &input : node->inputs()) {
    if (!input->isa<CNode>()) {
      (void)new_inputs.emplace_back(input);
      continue;
    }
    (void)new_inputs.emplace_back(
      MoveKCallerToBprop(manager, bprop_fg, input->cast<CNodePtr>(), depend_nodes, origin_to_new_nodes));
  }
  auto new_node = bprop_fg->NewCNode(new_inputs);
  origin_to_new_nodes->emplace(node, new_node);
  return new_node;
}

CNodePtr GetKGraphCallerFromTupleGetitem(const AnfNodePtr &node) {
  auto idx = GetValueNode<Int64ImmPtr>(node->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
  // The k_fg_caller return a tuple of forward result and bprop.
  if (idx == nullptr || idx->value() != 0) {
    return nullptr;
  }
  auto k_fg_caller = node->cast<CNodePtr>()->input(1);
  MS_EXCEPTION_IF_NULL(k_fg_caller);
  return k_fg_caller->cast<CNodePtr>();
}

bool IsFromBpropCaller(const AnfNodePtr &bprop_caller, const AnfNodePtr &depend_node) {
  if (bprop_caller == depend_node) {
    return true;
  }
  if (!IsPrimitiveCNode(depend_node, prim::kPrimTupleGetItem)) {
    return false;
  }
  return IsFromBpropCaller(depend_node->cast<CNodePtr>()->input(1), bprop_caller);
}

bool FilterDependency(const FuncGraphManagerPtr &manager, const std::set<CNodePtr> &final_nodes,
                      const AnfNodePtr &depend_node) {
  return std::all_of(final_nodes.begin(), final_nodes.end(), [&manager, &depend_node](const auto &final_node) {
    return !IsFromBpropCaller(GetBpropCaller(manager, GetBpropGetter(manager, final_node)), depend_node);
  });
}

void ReplaceFinalForwardGetter(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg,
                               const AnfNodePtr &origin_forward_getter, const AnfNodePtr &new_forward_getter) {
  auto node_users = manager->node_users()[origin_forward_getter];
  for (auto &node_and_idx : node_users) {
    auto user = node_and_idx.first;
    MS_EXCEPTION_IF_NULL(user);
    MS_LOG(DEBUG) << "User: " << user->DebugString();
    // The forward part may have multiple outputs.
    if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem)) {
      // Make new tuple_getitem to get corresponding output.
      auto new_getitem = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_forward_getter,
                                       user->cast_ptr<CNode>()->input(kInputNodeOutputIndexInTupleGetItem)});
      ReplaceFinalForwardGetter(manager, fg, user, new_getitem);
      continue;
    }
    MS_LOG(DEBUG) << "Set edge for user: " << user->DebugString();
    manager->SetEdge(user, node_and_idx.second, new_forward_getter);
  }
}

void AddDependNodes(const FuncGraphManagerPtr &manager, const FuncGraphPtr &fg, const CNodePtr &k_fg_caller_cnode) {
  // Get the nodes which the recomputed part should depend on;
  std::set<CNodePtr> final_nodes;
  std::set<AnfNodePtr> dependencies;
  GetDependencies(manager, k_fg_caller_cnode, &final_nodes, &dependencies);
  if (dependencies.empty()) {
    return;
  }
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy_if(
    dependencies.begin(), dependencies.end(), std::back_inserter(depend_inputs),
    [&manager, &final_nodes](const auto &dependency) { return FilterDependency(manager, final_nodes, dependency); });
  FuncGraphPtr bprop_fg;
  // Add the dependency nodes to the first recomputed nodes.
  auto bprop_caller = GetBpropCaller(manager, GetBpropGetter(manager, k_fg_caller_cnode));
  if (bprop_caller == nullptr) {
    bprop_fg = (*dependencies.begin())->func_graph();
  } else {
    bprop_fg = bprop_caller->func_graph();
  }

  auto depend_nodes = bprop_fg->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(bprop_fg);
  if (bprop_fg == fg) {
    if (!IsRecomputeCell(GetValueNode<FuncGraphPtr>(k_fg_caller_cnode->input(0)))) {
      auto depend = fg->NewCNode({NewValueNode(prim::kPrimDepend), k_fg_caller_cnode->input(1), depend_nodes});
      depend->AddAttr("recompute_insert", MakeValue(true));
      manager->SetEdge(k_fg_caller_cnode, 1, depend);
      k_fg_caller_cnode->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    } else {
      std::vector<AnfNodePtr> new_k_fg_caller_inputs{k_fg_caller_cnode->input(0)};
      (void)std::transform(k_fg_caller_cnode->inputs().begin() + 1, k_fg_caller_cnode->inputs().end(),
                           std::back_inserter(new_k_fg_caller_inputs),
                           [&fg, &depend_nodes](const AnfNodePtr &input) -> AnfNodePtr {
                             return fg->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), input, depend_nodes});
                           });
      auto new_k_fg_caller = fg->NewCNodeInOrder(new_k_fg_caller_inputs);
      (void)manager->Replace(k_fg_caller_cnode, new_k_fg_caller);
      new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    }
    return;
  }
  // If the graph of the bprop caller is not the same as the graph of k graph caller, we should move the k graph
  // caller to the graph of the bprop.
  std::unordered_map<CNodePtr, CNodePtr> origin_to_new_nodes;
  for (const auto &final_node : final_nodes) {
    auto new_k_fg_caller = MoveKCallerToBprop(manager, bprop_fg, final_node, depend_nodes, &origin_to_new_nodes);
    new_k_fg_caller->AddAttr(kAddedRecomputeDependAttr, MakeValue(true));
    auto forward_getter = GetForwardGetter(manager, final_node);
    if (forward_getter == nullptr) {
      (void)manager->Replace(final_node, new_k_fg_caller);
    } else {
      auto new_forward_getter = bprop_fg->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), new_k_fg_caller, NewValueNode(static_cast<int64_t>(0))});
      ReplaceFinalForwardGetter(manager, bprop_fg, forward_getter, new_forward_getter);
    }
  }
}

void AddDuplicatedAttr(const FuncGraphPtr &k_fg) {
  for (const auto &node : k_fg->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    node->cast_ptr<CNode>()->AddAttr(kAttrDuplicated, MakeValue(true));
  }
}
}  // namespace

bool AddRecomputeNodes(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  if (!EnableGraphReuse()) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool changed = false;
  auto all_node = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (auto iter = all_node.crbegin(); iter != all_node.crend(); (void)iter++) {
    const auto &node = *iter;
    if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto k_fg_caller_cnode = GetKGraphCallerFromTupleGetitem(node);
    if (k_fg_caller_cnode == nullptr || k_fg_caller_cnode->HasAttr(kAddedRecomputeDependAttr)) {
      continue;
    }
    auto k_fg = GetValueNode<FuncGraphPtr>(k_fg_caller_cnode->input(0));
    if (k_fg == nullptr || !k_fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
      continue;
    }
    auto primal_iter = k_fg->transforms().find("primal");
    if (primal_iter == k_fg->transforms().end()) {
      continue;
    }
    AnfNodePtr primal = nullptr;
    bool recompute_cell = false;
    auto primal_fg = primal_iter->second.func_graph();
    if (primal_fg != nullptr) {
      primal = NewValueNode(primal_fg);
      recompute_cell = true;
    } else {
      auto primal_primitive = primal_iter->second.primitive();
      if (primal_primitive != nullptr) {
        primal = NewValueNode(primal_primitive);
      }
    }
    if (primal == nullptr) {
      continue;
    }
    // Replace the forward getter with the origin primal.
    constexpr auto recursive_level = 2;
    MS_LOG(DEBUG) << "Handle recompute k graph forward getter: " << node->DebugString(recursive_level);
    std::vector<AnfNodePtr> inputs{primal};
    (void)inputs.insert(inputs.cend(), k_fg_caller_cnode->inputs().begin() + 1, k_fg_caller_cnode->inputs().end());
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_primal = fg->NewCNodeInOrder(inputs);
    bool change = AddNewPrimalNode(manager, fg, node, new_primal, recompute_cell);
    changed = change || changed;
    if (change && recompute_cell) {
      k_fg_caller_cnode->set_user_data("primal_fg_caller", new_primal);
    }
    // Add duplicated attr to help debugging.
    AddDuplicatedAttr(k_fg);
    if (HasRecomputedInput(k_fg_caller_cnode)) {
      continue;
    }

    MS_LOG(DEBUG) << "Not has recomputed input k_fg_caller_cnode: " << k_fg_caller_cnode->DebugString();
    AddDependNodes(manager, fg, k_fg_caller_cnode);
  }
  if (changed) {
    all_node = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
    for (const auto &node : all_node) {
      if (WithRecomputedScope(node)) {
        node->cast<CNodePtr>()->AddAttr(kAttrNeedCseAfterRecompute, MakeValue(true));
      }
    }
  }
  return changed;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
