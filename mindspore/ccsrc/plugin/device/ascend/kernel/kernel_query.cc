/**
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

#include "plugin/device/ascend/kernel/kernel_query.h"
#include <algorithm>
#include <string>
#include "ops/sequence_ops.h"
#include "ops/framework_ops.h"
#include "plugin/device/ascend/kernel/bisheng/bisheng_kernel_metadata.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_metadata.h"
#include "plugin/device/ascend/kernel/host/host_kernel_metadata.h"
#include "plugin/device/ascend/kernel/rts/rt_kernel_info.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_select.h"
#ifdef ENABLE_AKG
#include "plugin/device/ascend/kernel/akg/akg_kernel_metadata.h"
#endif
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "kernel/common_utils.h"
#include "kernel/oplib/oplib.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace kernel {
namespace {
void FilterInvalidKernelInfo(const CNodePtr &kernel_node,
                             std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  if (kernel_info_list->empty()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t unfold_output_tensor_num = AnfAlgo::GetOutputElementNum(kernel_node);
  size_t unfold_input_tensor_num = AnfAlgo::GetInputElementNum(kernel_node);
  size_t fold_output_tensor_num = 1;
  size_t fold_input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> filtered_list;
  std::ostringstream buffer;
  size_t info_index = 0;
  for (const auto &kernel_info : *kernel_info_list) {
    MS_EXCEPTION_IF_NULL(kernel_info);
    bool is_fold = kernel::IsFoldKernelBuildInfo(kernel_info);
    if (is_fold) {
      if (!common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, kernel_node)) {
        if (kernel_info->GetInputNum() != fold_input_tensor_num) {
          buffer << "Kernel [ " << info_index << " ] [Fold]: Kernel build info's input size ["
                 << kernel_info->GetInputNum() << "] cannot match the node's input size [" << fold_input_tensor_num
                 << "]\n";
          buffer << "\n kernel info:" << kernel_info->ToString();
          info_index++;
          continue;
        }
      } else {
        // compare input num
        std::vector<int64_t> dyn_input_sizes =
          common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrDynInputSizes);
        size_t real_input_num = 0;
        for (size_t i = 0; i < fold_input_tensor_num; ++i) {
          if (kernel_info->GetInputKernelObjectType(i) == kernel::KernelObjectType::TUPLE || dyn_input_sizes[i] == -1) {
            ++real_input_num;
          } else {
            real_input_num += LongToSize(dyn_input_sizes[i]);
          }
        }
        if (kernel_info->GetInputNum() != real_input_num) {
          buffer << "Kernel [ " << info_index << " ] [Fold]: Kernel build info's input size ["
                 << kernel_info->GetInputNum() << "] cannot match the node's input size [" << real_input_num << "]\n";
          buffer << "\n kernel info:" << kernel_info->ToString();
          info_index++;
          continue;
        }
      }

      // compare output num
      size_t real_output_num = unfold_output_tensor_num;
      if (kernel_info->GetOutputKernelObjectType(0) == kernel::KernelObjectType::TUPLE) {
        real_output_num = fold_output_tensor_num;
      }

      if (kernel_info->GetOutputNum() != real_output_num) {
        buffer << "Kernel [ " << info_index << " ] [Fold]: Kernel build info's output size ["
               << kernel_info->GetOutputNum() << "] cannot match the node's output size [" << real_output_num << "]\n";
        buffer << "\n kernel info:" << kernel_info->ToString();
        info_index++;
        continue;
      }

      (void)filtered_list.emplace_back(kernel_info);
    } else {
      if ((kernel_info->GetInputNum() == unfold_input_tensor_num) &&
          (kernel_info->GetOutputNum() == unfold_output_tensor_num)) {
        (void)filtered_list.emplace_back(kernel_info);
      } else if ((kernel_info->GetInputNum() == fold_input_tensor_num) &&
                 (kernel_info->GetOutputNum() == unfold_output_tensor_num)) {
        // For the condition that node input is tuple and kernel input is tensor, and this condition will insert
        // TupleToTensor after kernel_select
        (void)filtered_list.emplace_back(kernel_info);
      } else {
        buffer << "Kernel [ " << info_index << " ] [Unfold]:";
        if (kernel_info->GetOutputNum() != unfold_output_tensor_num) {
          buffer << "Kernel build info's output size [" << kernel_info->GetOutputNum() << "]"
                 << " cannot match the node's output size [" << unfold_output_tensor_num << "]\n";
        } else {
          buffer << "Kernel build info's input size [" << kernel_info->GetInputNum() << "]"
                 << " cannot match the node's input size [" << unfold_input_tensor_num << "]\n";
        }
        buffer << "\n kernel info:" << kernel_info->ToString();
      }
    }
    info_index++;
  }

  if (!filtered_list.empty()) {
    kernel_info_list->clear();
    (void)std::copy(filtered_list.begin(), filtered_list.end(), std::back_inserter(*kernel_info_list));
  } else {
    MS_LOG(INFO) << buffer.str();
    kernel_info_list->clear();
    MS_LOG(INFO) << "Node: " << kernel_node->DebugString() << "'s fold output size : [" << fold_output_tensor_num << "]"
                 << ", fold input size : [" << fold_input_tensor_num << "], unfold output size : ["
                 << unfold_output_tensor_num << "]"
                 << ", unfold input size : [" << unfold_input_tensor_num << "] can not match any kernelInfo !";
  }
}

bool SelectAicpuReshapeInTaskSink(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (common::AnfAlgo::GetCNodeName(kernel_node) != "Reshape") {
    return false;
  }
  const size_t AicpuReshapeSize = 2;
  if (kernel_node->size() != AicpuReshapeSize) {
    return false;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  return is_task_sink;
}
}  // namespace

void CheckKernelInfoListEmpty(const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list,
                              const std::string &type) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  if (kernel_info_list->empty()) {
    MS_LOG(INFO) << "Warning: kernel info list is empty, kernel type: " << type;
  }
}

abstract::AbstractBasePtr GenerateAbsByOpInfer(const PrimitivePtr &primitive, const AnfNodePtrList &input_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto found = abstract::GetPrimitiveInferImpl(primitive);
  if (!found.has_value()) {
    MS_LOG(INTERNAL_EXCEPTION) << primitive->name() << "infer is not registered.";
  }

  std::vector<AbstractBasePtr> input_args;
  std::for_each(input_list.begin(), input_list.end(),
                [&input_args](const auto &input) { input_args.emplace_back(input->abstract()); });
  auto infer_impl = found.value();
  auto abs = infer_impl.InferShapeAndType(nullptr, primitive, input_args);
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for " << primitive->name() << " is " << abs->ToString();
  return abs;
}

AnfNodePtr ConvertAllTupleInputsToTensor(const FuncGraphPtr &graph, const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  MS_EXCEPTION_IF_NULL(graph);
  if (common::AnfAlgo::CheckPrimitiveType(cnode_ptr, prim::kPrimCall) ||
      common::AnfAlgo::CheckPrimitiveType(cnode_ptr, prim::kPrimPartial)) {
    return nullptr;
  }

  if (common::AnfAlgo::HasDynamicTupleInput(cnode_ptr)) {
    MS_LOG(INFO) << "Node " << cnode_ptr->fullname_with_scope()
                 << " has dynamic tuple input, can't convert. Node debug string:" << cnode_ptr->DebugString();
    return nullptr;
  }

  if (!common::AnfAlgo::HasTupleInput(cnode_ptr)) {
    return nullptr;
  }

  size_t input_num = cnode_ptr->inputs().size() - 1;
  std::vector<AnfNodePtr> new_inputs;
  new_inputs.push_back(common::AnfAlgo::GetCNodePrimitiveNode(cnode_ptr));
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode_ptr, i);
    MS_EXCEPTION_IF_NULL(input_node);
    if (common::AnfAlgo::IsTupleOutput(input_node)) {
      auto prim = NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleToTensor->name()));
      MS_EXCEPTION_IF_NULL(prim);
      AnfNodePtrList inputs = {prim, input_node};
      CNodePtr tuple_to_tensor = graph->NewCNode(inputs);
      MS_EXCEPTION_IF_NULL(tuple_to_tensor);
      // attr dtype
      auto data_type = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
      common::AnfAlgo::SetNodeAttr(kAttrDType, TypeIdToType(data_type), tuple_to_tensor);

      // set abstract
      auto abs = GenerateAbsByOpInfer(GetCNodePrimitive(tuple_to_tensor), {input_node});
      MS_EXCEPTION_IF_NULL(abs);
      MS_LOG(DEBUG) << "Abstract for TupleToTensor op is " << abs->ToString();
      tuple_to_tensor->set_abstract(abs);
      new_inputs.push_back(tuple_to_tensor);
    } else {
      new_inputs.push_back(input_node);
    }
  }

  auto new_cnode = opt::NewCNode({new_inputs}, graph, {cnode_ptr});
  new_cnode->set_abstract(cnode_ptr->abstract());
  new_cnode->set_scope(cnode_ptr->scope());
  new_cnode->set_primal_attrs(cnode_ptr->primal_attrs());
  new_cnode->set_attrs(cnode_ptr->attrs());

  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, new_cnode)) {
    common::AnfAlgo::EraseNodeAttr(kAttrDynInputSizes, new_cnode);
  }

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  if (kernel_graph != nullptr) {
    kernel_graph->FrontBackendlMapUpdate(cnode_ptr, new_cnode);
  }
  return new_cnode;
}

void KernelQueryAllDetail(const CNodePtr &select_cnode,
                          std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  TbeMetadataInfo(select_cnode, kernel_info_list);
  CheckKernelInfoListEmpty(kernel_info_list, "TBE_Kernel");

  if (kernel_info_list->empty()) {
    GetRtKelInfo(select_cnode, kernel_info_list);
    CheckKernelInfoListEmpty(kernel_info_list, "RT_Kernel");
  }
  if (kernel_info_list->empty()) {
    HcclMetadataInfo(select_cnode, kernel_info_list);
    CheckKernelInfoListEmpty(kernel_info_list, "HCCL_Kernel");
  }
  if (SelectAicpuReshapeInTaskSink(select_cnode)) {
    return;
  }
  if (kernel_info_list->empty()) {
    HostMetadataInfo(select_cnode, kernel_info_list);
    CheckKernelInfoListEmpty(kernel_info_list, "HOST_Kernel");
  }
  if (kernel_info_list->empty()) {
    BiShengMetadataInfo(select_cnode, kernel_info_list);
    CheckKernelInfoListEmpty(kernel_info_list, "BISHENG_Kernel");
  }
}

void KernelQueryAll(const CNodePtr &kernel_node,
                    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  auto select_cnode = kernel_node;
  std::vector<int64_t> dyn_input_sizes = {};
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, select_cnode)) {
    dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(select_cnode, kAttrDynInputSizes);
  }

  // unfold  all tuple inputs and select
  auto tuple_unfold_node = opt::ConvertMakeTupleInputToPlantInputs(kernel_node->func_graph(), kernel_node);
  if (tuple_unfold_node != nullptr) {
    auto tuple_unfold_cnode = tuple_unfold_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_unfold_cnode);
    select_cnode = tuple_unfold_cnode;
    select_cnode->set_fullname_with_scope(kernel_node->fullname_with_scope());
    MS_LOG(INFO) << "Create tuple unfold node " << tuple_unfold_node->fullname_with_scope() << ", debug string ["
                 << tuple_unfold_node->DebugString() << "] from " << kernel_node->fullname_with_scope()
                 << ", debug string [" << kernel_node->DebugString() << "].";
  }

  KernelQueryAllDetail(select_cnode, kernel_info_list);

  if (kernel_info_list->empty()) {
    // convert all tuple inputs to tensor and select
    auto tensor_input_node = ConvertAllTupleInputsToTensor(kernel_node->func_graph(), kernel_node);
    if (tensor_input_node != nullptr) {
      auto tensor_input_cnode = tensor_input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tensor_input_cnode);
      select_cnode = tensor_input_cnode;
      select_cnode->set_fullname_with_scope(kernel_node->fullname_with_scope());
      MS_LOG(INFO) << "Create all tensor inputs node " << tensor_input_cnode->fullname_with_scope()
                   << ", debug string [" << tensor_input_cnode->DebugString() << "] from "
                   << kernel_node->fullname_with_scope() << ", debug string [" << kernel_node->DebugString() << "].";
    }
    KernelQueryAllDetail(select_cnode, kernel_info_list);
  }

  if (!kernel_info_list->empty()) {
    common::AnfAlgo::CopyNodeAttrs(select_cnode, kernel_node);
  }
  // recover the kAttrDynInputSizes of origin kernel_node
  if (!dyn_input_sizes.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), kernel_node);
  } else if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, kernel_node)) {
    common::AnfAlgo::EraseNodeAttr(kAttrDynInputSizes, kernel_node);
  }
}

void KernelQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list,
                 KernelType kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

  const PrimitivePtr kPrimProdForceSeA = std::make_shared<Primitive>("ProdForceSeA");
  if (IsPrimitiveCNode(kernel_node, kPrimProdForceSeA)) {
    kernel_type = KernelType::AKG_KERNEL;
  }

  const PrimitivePtr kPrimLoadIm2Col = std::make_shared<Primitive>("LoadIm2Col");
  if (IsPrimitiveCNode(kernel_node, kPrimLoadIm2Col)) {
    kernel_type = KernelType::AKG_KERNEL;
  }  // use LoadIm2Col only for THOR optimizer

  switch (kernel_type) {
    case KernelType::AKG_KERNEL:
#ifdef ENABLE_AKG
      AkgMetadataInfo(kernel_node, kernel_info_list);
#else
      MS_LOG(EXCEPTION) << "Not supported AKG kernel: " << kernel_node->fullname_with_scope()
                        << " because ENABLE_AKG is not defined";
#endif
      break;
    default:
      KernelQueryAll(kernel_node, kernel_info_list);
      break;
  }
  // check output
  FilterInvalidKernelInfo(kernel_node, kernel_info_list);
}

void AICPUQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  kernel_info_list->clear();
  AicpuMetadataInfo(kernel_node, kernel_info_list);
  FilterInvalidKernelInfo(kernel_node, kernel_info_list);
}

bool IsSupportedByAICPU(const AnfNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(select_kernel_build_info);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto cnode = kernel_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AICPUQuery(cnode, &kernel_info_list);
  return std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                     [&select_kernel_build_info](const kernel::KernelBuildInfoPtr item) {
                       MS_EXCEPTION_IF_NULL(item);
                       return item->IsSimilarityKernelBuildInfo(*select_kernel_build_info);
                     });
}
}  // namespace kernel
}  // namespace mindspore
