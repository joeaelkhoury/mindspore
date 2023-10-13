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

#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <tuple>
#include <functional>
#include "include/backend/optimizer/helper.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/debug/profiler/profiling.h"
#include "ops/ascend_op_name.h"
#include "ops/math_op_name.h"
#include "ops/conv_pool_ops.h"
#include "ops/other_ops.h"
#include "ops/nn_ops.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "ops/op_name.h"
#include "kernel/common_utils.h"
#include "kernel/oplib/oplib.h"
#include "kernel/oplib/super_bar.h"
#include "plugin/device/ascend/kernel/kernel_query.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "plugin/device/cpu/hal/device/kernel_select_cpu.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_attr_and_input_convert_regist.h"
#include "frontend/operator/ops.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
using KernelBuildInfoBuilderPtr = std::shared_ptr<kernel::KernelBuildInfo::KernelBuildInfoBuilder>;
using OriginTypeConverter = std::function<TypeId(const TypeId &)>;
const OriginTypeConverter kDefaultOriginTypeConverter = [](const TypeId &origin_type) { return origin_type; };
constexpr int kWeightUnInitScore = 1;
constexpr int kWeightInitScore = 2;
constexpr int kFeatureMapBaseScore = 10;
constexpr auto kPatternOpaque = "Opaque";
constexpr auto kPriChoosenFormat = "pri_format";
constexpr auto kOriSelectFormat = "ori_select_format";
constexpr auto kForceFp32Strategy = "force_fp32";
enum MatchCountPriority : size_t {
  MATCH_COUNT_PRIORITY_BEGIN = 0,
  MATCH_DTYPE_COUNT = MATCH_COUNT_PRIORITY_BEGIN,
  MATCH_FORMAT_COUNT,
  MATCH_SPECIAL_FORMAT_COUNT,
  MATCH_DEFAULT_FORMAT_COUNT,
  MATCH_OUTPUT_DTYPE_COUNT,
  MATCH_COUNT_PRIORITY_END
};

const std::map<std::string, std::vector<std::string>> kNextOpFormatList = {
  {prim::kPrimConv2D->name(), {kOpFormat_NC1HWC0, kOpFormat_FRAC_Z}}};

mindspore::HashSet<std::string> kHighPrecisionOp = {kConv2DOpName,
                                                    kMatMulOpName,
                                                    kBatchMatMulOpName,
                                                    kConv2DBackpropInputOpName,
                                                    kConv2DBackpropFilterOpName,
                                                    kBiasAddGradOpName,
                                                    kSigmoidCrossEntropyWithLogitsV2OpName};

bool IsCubeKernel(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  static const std::unordered_set<std::string> kCubeKernelSet = {
    // matmul
    kMatMulOpName, kMatMulV2OpName, kBatchMatMulOpName, kBatchMatMulV2OpName,
    // conv
    kConv2DOpName, kConv3DOpName,
    // conv dx
    kConv2DBackpropInputOpName, kConv2DBackpropInputDOpName, kConv2DTransposeOpName, kConv2DTransposeDOpName,
    kDepthwiseConv2DBackpropInputOpName, kDepthwiseConv2DBackpropInputDOpName, kConv3DBackpropInputOpName,
    kConv3DBackpropInputDOpName, kConv3DTransposeOpName, kConv3DTransposeDOpName,
    // conv dw
    kConv2DBackpropFilterOpName, kConv2DBackpropFilterDOpName, kDepthwiseConv2DBackpropFilterOpName,
    kDepthwiseConv2DBackpropFilterDOpName, kConv3DBackpropFilterOpName, kConv3DBackpropFilterDOpName};

  auto op_name = common::AnfAlgo::GetCNodeName(node);
  return kCubeKernelSet.find(op_name) != kCubeKernelSet.end();
}

void FallbackOps(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto inputs = kernel_node->inputs();
  const auto &fallback_idx = kernel::SuperBar::GetSBFallbackOpIndex(op_name);
  if (fallback_idx.empty() || inputs.empty()) {
    return;
  }
  AnfNodePtrList new_inputs = {inputs[0]};
  for (const auto &idx : fallback_idx) {
    if (idx >= inputs.size()) {
      MS_LOG(EXCEPTION) << "Invalid idx: " << idx << ", node: " << kernel_node->fullname_with_scope()
                        << ", total input size: " << inputs.size();
    }
    (void)new_inputs.emplace_back(inputs[idx]);
  }
  kernel_node->set_inputs(new_inputs);
}

namespace {
bool ProcessKernelInputIdx(const kernel::KernelBuildInfoPtr &kernel_build_info, const TypeId &input_type,
                           size_t kernel_input_index) {
  if (kernel_input_index >= kernel_build_info->GetInputNum()) {
    return false;
  }
  if (kernel_build_info->GetInputDeviceType(kernel_input_index) != input_type) {
    return false;
  }
  return true;
}

// Judge if the tuple input of the node should be unfold to match the kernel_build_info
bool IsNodeTupleInputNeedUnfold(const CNodePtr &cnode, const kernel::KernelBuildInfoPtr &kernel_build_info) {
  if (common::AnfAlgo::HasDynamicTupleInput(cnode)) {
    return false;
  }
  size_t fold_input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode);
  size_t unfold_input_tensor_num = AnfAlgo::GetInputElementNum(cnode);
  return !(fold_input_tensor_num != unfold_input_tensor_num &&
           fold_input_tensor_num == kernel_build_info->GetInputNum());
}
}  // namespace

bool MatchUnfoldInferOutputDataType(const CNodePtr &cnode, const kernel::KernelBuildInfoPtr &kernel_build_info,
                                    const OriginTypeConverter &input_type_converter = kDefaultOriginTypeConverter,
                                    const OriginTypeConverter &output_type_converter = kDefaultOriginTypeConverter) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Check input data type
  size_t kernel_input_index = 0;
  size_t fold_input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode);

  bool is_input_need_unfold = IsNodeTupleInputNeedUnfold(cnode, kernel_build_info);

  for (size_t input_index = 0; input_index < fold_input_tensor_num; ++input_index) {
    std::vector<TypeId> inputs_type = common::AnfAlgo::GetRealPrevNodesOutputInferDataType(cnode, input_index);
    for (auto &i : inputs_type) {
      if (!ProcessKernelInputIdx(kernel_build_info, input_type_converter(i), kernel_input_index)) {
        return false;
      }
      if (is_input_need_unfold) {
        ++kernel_input_index;
      }
    }
    if (!is_input_need_unfold) {
      ++kernel_input_index;
    }
  }

  // Check output data type
  for (size_t output_index = 0; output_index < kernel_build_info->GetOutputNum(); ++output_index) {
    if (kernel_build_info->GetOutputDeviceType(output_index) !=
        output_type_converter(common::AnfAlgo::GetOutputInferDataType(cnode, output_index))) {
      return false;
    }
  }
  return true;
}

bool MatchFoldInferOutputDataType(const CNodePtr &cnode, const kernel::KernelBuildInfoPtr &kernel_build_info,
                                  const OriginTypeConverter &input_type_converter = kDefaultOriginTypeConverter,
                                  const OriginTypeConverter &output_type_converter = kDefaultOriginTypeConverter) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Check input data type
  size_t fold_input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode);
  size_t kernel_index = 0;
  for (size_t input_index = 0; input_index < fold_input_tensor_num; ++input_index) {
    if (kernel_build_info->GetInputKernelObjectType(kernel_index) == kernel::KernelObjectType::TUPLE) {
      auto input_node = cnode->inputs()[input_index + 1];
      TypeId input_origin_type = input_type_converter(common::AnfAlgo::GetOutputInferDataType(input_node, 0));
      if (kernel_build_info->GetInputDeviceType(kernel_index) != input_origin_type) {
        return false;
      }
      ++kernel_index;
    } else {
      std::vector<TypeId> inputs_type = common::AnfAlgo::GetRealPrevNodesOutputInferDataType(cnode, input_index);
      for (auto &i : inputs_type) {
        if (!ProcessKernelInputIdx(kernel_build_info, input_type_converter(i), kernel_index)) {
          return false;
        }
        ++kernel_index;
      }
    }
  }
  // Check output data type
  for (size_t output_index = 0; output_index < kernel_build_info->GetOutputNum(); ++output_index) {
    if (kernel_build_info->GetOutputDeviceType(output_index) !=
        output_type_converter(common::AnfAlgo::GetOutputInferDataType(cnode, output_index))) {
      return false;
    }
  }
  return true;
}

bool MatchInferOutputDataType(const CNodePtr &cnode, const kernel::KernelBuildInfoPtr &kernel_build_info,
                              const OriginTypeConverter &input_type_converter = kDefaultOriginTypeConverter,
                              const OriginTypeConverter &output_type_converter = kDefaultOriginTypeConverter) {
  MS_EXCEPTION_IF_NULL(cnode);
  bool is_fold = kernel::IsFoldKernelBuildInfo(kernel_build_info);
  if (is_fold) {
    return MatchFoldInferOutputDataType(cnode, kernel_build_info, input_type_converter, output_type_converter);
  } else {
    return MatchUnfoldInferOutputDataType(cnode, kernel_build_info, input_type_converter, output_type_converter);
  }
}

string GetPriorityMatchFormat(const CNodePtr &cnode) {
  constexpr size_t k5dSize = 5;
  constexpr size_t k4dSize = 4;
  string priority_matched_format = kOpFormat_NC1HWC0;
  bool is_init = false;
  bool need_change_nd = false;
  bool is_5d_input = false;
  bool is_dyn_rank = common::AnfAlgo::IsDynamicRankNode(cnode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t index = 0; index < input_num; ++index) {
    auto pre_output_format = AnfAlgo::GetPrevNodeOutputFormat(cnode, index);
    if (AnfAlgo::IsFeatureMapInput(cnode, index) && IsOneOfHWSpecialFormat(pre_output_format)) {
      priority_matched_format = !is_init ? pre_output_format : priority_matched_format;
      is_init = true;
    }
    // feature map has two or more special format;
    if (priority_matched_format != pre_output_format && pre_output_format != kOpFormat_DEFAULT) {
      priority_matched_format = kOpFormat_DEFAULT;
    }
    const auto &prev_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
    if (IsDynamicRank(prev_shape)) {
      is_dyn_rank = true;
    }
    auto input_shape_size = prev_shape.size();
    if (input_shape_size == k5dSize) {
      is_5d_input = true;
    }
    need_change_nd = (need_change_nd || (input_shape_size != k4dSize && input_shape_size > 1));
  }
  if (need_change_nd && priority_matched_format != kOpFormat_FRAC_NZ) {
    priority_matched_format = kOpFormat_DEFAULT;
  }
  if (is_5d_input && priority_matched_format != kOpFormat_FRAC_NZ) {
    priority_matched_format = kOpFormat_NDC1HWC0;
  }
  if (is_dyn_rank) {
    priority_matched_format = kOpFormat_ND;
  }
  common::AnfAlgo::SetNodeAttr(kPriChoosenFormat, MakeValue(priority_matched_format), cnode);
  return priority_matched_format;
}

/**
 * Compare two vector by priority, select a better vector, like compare two num, first compare highest num location,
 * if equal then next num location
 * example:[3,1,1,1] > [2,2,2,2] > [2,2,1,2] > [2,1,1,3]
 */
bool PriorityChooseItem(const std::vector<int> &cur_item, std::vector<int> *best_item) {
  MS_EXCEPTION_IF_NULL(best_item);
  if (cur_item.size() != best_item->size()) {
    MS_LOG(ERROR) << "Item size should be same!";
    return false;
  }
  // Update the best_item by comparing the cur_item and best_item
  for (size_t i = 0; i < cur_item.size(); i++) {
    if (cur_item[i] > best_item->at(i)) {
      *best_item = cur_item;
      return true;
    } else if (cur_item[i] == best_item->at(i)) {
      continue;
    } else {
      return false;
    }
  }
  return false;
}

void UpdateCurMatchCounts(const kernel::KernelBuildInfo &kernel_build_info, const std::shared_ptr<CNode> &kernel_node,
                          std::vector<int> *const cur_kernelinfo_match_counts) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(cur_kernelinfo_match_counts);
  if (cur_kernelinfo_match_counts->size() < MATCH_COUNT_PRIORITY_END) {
    MS_LOG(EXCEPTION) << "Out of range cur_kernel info_match_counts " << MATCH_COUNT_PRIORITY_END;
  }
  auto pri_match_format = GetPriorityMatchFormat(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_anf_node =
      common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(kernel_node, input_index), 0).first;
    MS_EXCEPTION_IF_NULL(input_anf_node);
    // we do not take ValueNode into consideration in graph kernel.
    auto base_score = AnfAlgo::IsFeatureMapInput(kernel_node, input_index) ? kFeatureMapBaseScore : kWeightInitScore;
    if (AnfAlgo::GetOutputDeviceDataType(input_anf_node, 0) == kTypeUnknown) {
      base_score = kWeightUnInitScore;
    }
    if (kernel_build_info.GetInputFormat(input_index) == AnfAlgo::GetPrevNodeOutputFormat(kernel_node, input_index)) {
      (*cur_kernelinfo_match_counts)[static_cast<size_t>(MATCH_FORMAT_COUNT)] += base_score;
    }
    // we match output fix precision first.
    auto prev_device_type = common::AnfAlgo::GetPrevNodeOutputPrecision(kernel_node, input_index);
    if (prev_device_type == kTypeUnknown) {
      prev_device_type = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index);
    }
    if (kernel_build_info.GetInputDeviceType(input_index) == prev_device_type) {
      (*cur_kernelinfo_match_counts)[static_cast<size_t>(MATCH_DTYPE_COUNT)] += base_score;
    }
    if (kernel_build_info.GetInputFormat(input_index) == pri_match_format) {
      (*cur_kernelinfo_match_counts)[static_cast<size_t>(MATCH_SPECIAL_FORMAT_COUNT)] += base_score;
    }
    if (kernel_build_info.GetInputFormat(input_index) == kOpFormat_DEFAULT ||
        kernel_build_info.GetInputFormat(input_index) == kOpFormat_NCDHW) {
      (*cur_kernelinfo_match_counts)[static_cast<size_t>(MATCH_DEFAULT_FORMAT_COUNT)] += base_score;
    }
  }

  size_t output_num = AnfAlgo::GetOutputElementNum(kernel_node);
  if (output_num > 0 && kernel_build_info.GetOutputKernelObjectType(0) == kernel::TUPLE) {
    output_num = 1;
  }

  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    // cal count of same output dtype between abstract and kernel info
    if (kernel_build_info.GetOutputDeviceType(output_index) ==
        common::AnfAlgo::GetOutputInferDataType(kernel_node, output_index)) {
      (*cur_kernelinfo_match_counts)[static_cast<size_t>(MATCH_OUTPUT_DTYPE_COUNT)] += 1;
    }
    if (kernel_build_info.GetOutputFormat(output_index) == pri_match_format) {
      (*cur_kernelinfo_match_counts)[static_cast<size_t>(MATCH_SPECIAL_FORMAT_COUNT)] += 1;
    }
  }
}

std::string PrintRaiseOrReducePrecisionSelectedInfo(
  const CNodePtr &cnode, const std::shared_ptr<kernel::KernelBuildInfo> &selected_kernel_build_info,
  KernelSelectStatus KernelSelectStatus) {
  MS_EXCEPTION_IF_NULL(selected_kernel_build_info);
  MS_EXCEPTION_IF_NULL(cnode);
  std::ostringstream buffer;
  buffer << cnode->DebugString();
  if (KernelSelectStatus == kStatusReducePrecision) {
    buffer << " Reduce precision, node datatype: \n";
  } else if (KernelSelectStatus == kStatusRaisePrecision) {
    buffer << " Raise precision, node datatype: \n";
  }
  GatherInputAndOutputInferType(buffer, cnode);
  buffer << ", select kernel:" << selected_kernel_build_info->ToString();
  return buffer.str();
}

std::shared_ptr<kernel::KernelBuildInfo> ChooseMatchedKernelInfo(
  const CNodePtr &kernel_node, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  if (kernel_info_list.empty()) {
    return nullptr;
  }
  std::vector<int> most_match_counts = {-1, -1, -1, -1, -1};
  size_t selected_index = 0;
  for (size_t info_index = 0; info_index < kernel_info_list.size(); ++info_index) {
    std::vector<int> cur_kernel_info_match_counts = {0, 0, 0, 0, 0};
    auto kernel_info_ptr = kernel_info_list[info_index];
    MS_EXCEPTION_IF_NULL(kernel_info_ptr);
    UpdateCurMatchCounts(*kernel_info_ptr, kernel_node, &cur_kernel_info_match_counts);
    // Currently the selection policy is the match format count first, and then is datatype counts.
    if (PriorityChooseItem(cur_kernel_info_match_counts, &most_match_counts)) {
      selected_index = info_index;
    }
  }
  return kernel_info_list[selected_index];
}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> FilteredKernelInfoByDtype(
  const CNodePtr &cnode, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list,
  const OriginTypeConverter &input_type_converter = kDefaultOriginTypeConverter,
  const OriginTypeConverter &output_type_converter = kDefaultOriginTypeConverter) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> result;
  for (const auto &kernel_build_info : kernel_info_list) {
    MS_EXCEPTION_IF_NULL(kernel_build_info);
    if (!MatchInferOutputDataType(cnode, kernel_build_info, input_type_converter, output_type_converter)) {
      continue;
    }
    result.push_back(kernel_build_info);
  }
  return result;
}

bool MatchObjectType(const kernel::KernelObjectType &node_object, const kernel::KernelObjectType &kernel_object) {
  if (node_object == kernel_object) {
    return true;
  }

  if (node_object == kernel::SCALAR && kernel_object == kernel::TENSOR) {
    return true;
  }

  // for monad output op such as labelset labelswitch labelgoto ...
  if (node_object == kernel::UNKNOWN_TYPE && kernel_object == kernel::TENSOR) {
    return true;
  }
  // This condition will insert TensorToTuple in the InsertTypeTransformOp pass later
  if (node_object == kernel::TENSOR && kernel_object == kernel::TUPLE) {
    return true;
  }

  if (node_object == kernel::TUPLE && kernel_object == kernel::TENSOR) {
    return true;
  }

  MS_LOG(INFO) << "Object mismatch. node object type : " << node_object << ", kernel object type: " << kernel_object;
  return false;
}
// kernel:tuple, node:tuple  -> compare objecttype
// kernel:tuple, node:tensor -> compare objecttype
// kernel:tensor, node:tensor -> compare objecttype
// kernel:tensor, node:tuple -> unfold node, then compare object type
namespace {
bool ProcessInputObjectType(const CNodePtr &cnode, const std::shared_ptr<kernel::KernelBuildInfo> &kernel_build_info,
                            std::vector<kernel::KernelObjectType> *new_input_object_types, size_t *kernel_input_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(new_input_object_types);
  MS_EXCEPTION_IF_NULL(kernel_input_index);
  // Check input object type
  auto kernel_inputs_object_type = kernel_build_info->GetAllInputKernelObjectTypes();
  auto node_inputs_object_type = kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllInputObjectType(cnode));

  bool is_node_input_need_unfold = IsNodeTupleInputNeedUnfold(cnode, kernel_build_info);
  for (size_t input_index = 0; input_index < node_inputs_object_type.size(); ++input_index) {
    if (kernel_inputs_object_type[(*kernel_input_index)] != kernel::KernelObjectType::TUPLE &&
        node_inputs_object_type[input_index] == kernel::KernelObjectType::TUPLE && is_node_input_need_unfold) {
      // tuple_unfold condition
      std::vector<KernelWithIndex> index_inputs = common::AnfAlgo::GetRealPrevNodesOutput(cnode, input_index);
      for (auto &index_input : index_inputs) {
        auto real_input_node = index_input.first;
        MS_EXCEPTION_IF_NULL(real_input_node);
        if ((*kernel_input_index) >= kernel_inputs_object_type.size()) {
          MS_LOG(DEBUG) << "index is large equal than list size: " << (*kernel_input_index) << " vs "
                        << kernel_inputs_object_type.size();
          return false;
        }
        if (!MatchObjectType(
              kernel::TypeIdToKernelObjectType(AnfAlgo::GetAbstractObjectType(real_input_node->abstract())),
              kernel_inputs_object_type[(*kernel_input_index)])) {
          return false;
        }
        ++(*kernel_input_index);
      }
      // For one element tuple, set the objecttype according to prevnode's output object type
      if (index_inputs.size() == 1) {
        auto prevnode_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, input_index);
        auto prevnode_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(prevnode_with_index.first);
        if (prevnode_kernel_build_info != nullptr) {
          auto prevnode_object_type = prevnode_kernel_build_info->GetOutputKernelObjectType(prevnode_with_index.second);
          if (prevnode_object_type != kernel::KernelObjectType::UNKNOWN_TYPE) {
            (*new_input_object_types).push_back(prevnode_object_type);
            continue;
          }
        }
      }
      (*new_input_object_types).push_back(kernel::KernelObjectType::TUPLE_UNFOLD);
    } else {
      auto node_object = node_inputs_object_type[input_index];
      auto kernel_object = kernel_inputs_object_type[(*kernel_input_index)];
      if (!MatchObjectType(node_object, kernel_object)) {
        return false;
      }
      if (node_object == kernel::KernelObjectType::SCALAR && kernel_object == kernel::KernelObjectType::TENSOR) {
        (*new_input_object_types).push_back(kernel::KernelObjectType::SCALAR);
      } else {
        (*new_input_object_types).push_back(kernel_inputs_object_type[(*kernel_input_index)]);
      }
      ++(*kernel_input_index);
    }
  }
  return true;
}
}  // namespace

bool MatchObjectType(const CNodePtr &cnode, const std::shared_ptr<kernel::KernelBuildInfo> &kernel_build_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t kernel_input_index = 0;
  std::vector<kernel::KernelObjectType> new_input_object_types = {};
  if (!ProcessInputObjectType(cnode, kernel_build_info, &new_input_object_types, &kernel_input_index)) {
    return false;
  }
  // Check output object type
  auto kernel_outputs_object_type = kernel_build_info->GetAllOutputKernelObjectTypes();
  auto node_output_object_type = AnfAlgo::GetAbstractObjectType(cnode->abstract());
  std::vector<kernel::KernelObjectType> new_output_object_types = {};

  if (node_output_object_type == kObjectTypeTuple && kernel_outputs_object_type[0] != kernel::KernelObjectType::TUPLE) {
    auto tuple_abs = cnode->abstract()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_abs);
    auto items = tuple_abs->elements();
    size_t output_index = 0;
    for (const auto &item : items) {
      if (output_index >= kernel_outputs_object_type.size()) {
        MS_LOG(DEBUG) << "index is large equal than list size: " << output_index << " vs "
                      << kernel_outputs_object_type.size();
        return false;
      }
      if (!MatchObjectType(kernel::TypeIdToKernelObjectType(AnfAlgo::GetAbstractObjectType(item)),
                           kernel_outputs_object_type[output_index])) {
        return false;
      }
      ++output_index;
    }
    new_output_object_types = {kernel::KernelObjectType::TUPLE_UNFOLD};
  } else {
    auto output_num = AnfAlgo::GetOutputElementNum(cnode);
    if (output_num > 0) {
      if (!MatchObjectType(kernel::TypeIdToKernelObjectType(node_output_object_type), kernel_outputs_object_type[0])) {
        return false;
      }
      new_output_object_types.push_back(kernel_outputs_object_type[0]);
    }
  }

  kernel_build_info->SetInputsKernelObjectType(new_input_object_types);
  kernel_build_info->SetOutputsKernelObjectType(new_output_object_types);

  return true;
}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> FilteredKernelInfoByObjectType(
  const CNodePtr &cnode, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> result;
  for (const auto &kernel_build_info : kernel_info_list) {
    MS_EXCEPTION_IF_NULL(kernel_build_info);
    auto new_kernel_build_info = std::make_shared<kernel::KernelBuildInfo>(*kernel_build_info);
    if (!MatchObjectType(cnode, new_kernel_build_info)) {
      continue;
    }
    result.push_back(new_kernel_build_info);
  }
  return result;
}

void SetCastAndWeightFormat(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (!common::AnfAlgo::HasNodeAttr(kAttrPynativeNextIndex, kernel_node) ||
      !common::AnfAlgo::HasNodeAttr(kAttrPynativeNextOpName, kernel_node)) {
    MS_LOG(EXCEPTION) << "The node [" << kernel_node->DebugString() << "] attr of " << kAttrPynativeNextIndex << " or "
                      << kAttrPynativeNextOpName << " has not been set yet!" << trace::DumpSourceLines(kernel_node);
  }
  auto next_index = common::AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPynativeNextIndex);
  auto next_op_name = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrPynativeNextOpName);
  auto iter = kNextOpFormatList.find(next_op_name);
  if (iter == kNextOpFormatList.end()) {
    MS_LOG(INFO) << "The op name " << next_op_name << "has not been set in the next op map ";
    return;
  }
  if (iter->second.size() < next_index) {
    MS_LOG(EXCEPTION) << "Next input index " << next_index << "is out of range in the next op map max size is "
                      << iter->second.size() << trace::DumpSourceLines(kernel_node);
  }
  if (common::AnfAlgo::GetCNodeName(kernel_node) != prim::kPrimCast->name()) {
    MS_LOG(INFO) << "Only supported to change the node Cast's build info!!!";
    return;
  }
  common::AnfAlgo::SetNodeAttr(kOriSelectFormat, MakeValue(AnfAlgo::GetInputFormat(kernel_node, 0)), kernel_node);
  auto format = iter->second[next_index];
  auto info_builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(kernel_node));
  MS_EXCEPTION_IF_NULL(info_builder);
  info_builder->SetInputsFormat({format});
  info_builder->SetOutputsFormat({format});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), kernel_node.get());
}

void SetWeightFormat(const AnfNodePtr &real_input_node, std::vector<string> output_format, const CNodePtr &kernel_node,
                     size_t input_index, bool force_fresh = false) {
  MS_EXCEPTION_IF_NULL(real_input_node);
  if (real_input_node->isa<CNode>()) {
    return;
  }

  if (AnfAlgo::OutputAddrExist(real_input_node, 0) &&
      AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) != kTypeUnknown) {
    return;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool disable_convert = real_input_node->isa<Parameter>() || real_input_node->isa<ValueNode>();
  // In PyNative mode, the weight data will be copied to the device in the first step,
  // and there will be no HostToDeviceCopy in the follow-up. If host format conversion is disabled,
  // the TransData operator will be executed in each subsequent step, resulting in poor performance.
  if (disable_convert && (context_ptr->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) ||
                          context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode)) {
    disable_convert = trans::kFormatWithTransFunc.find(output_format[0]) == trans::kFormatWithTransFunc.end();
  }
  // if not find in host convert format map means the host has not registered the convert function of this format
  if (output_format[0] != kOpFormat_DEFAULT && disable_convert) {
    output_format = {AnfAlgo::GetOutputFormat(real_input_node, 0)};
  }
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  // we set special device info of a input tensor.
  auto op_info = kernel::tbe::TbeDynamicShapeUtil::FindOp(common::AnfAlgo::GetCNodeName(kernel_node), kernel_node);
  if (op_info != nullptr) {
    force_fresh = op_info->is_ref() || force_fresh;
  }
  auto selected_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  // tensor id -> type id
  static std::unordered_map<std::string, TypeId> format_type;
  if (AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) == kTypeUnknown || force_fresh) {
    if (IsValueNode<tensor::Tensor>(real_input_node)) {
      auto host_tensor_ptr = GetValueNode<tensor::TensorPtr>(real_input_node);
      MS_EXCEPTION_IF_NULL(host_tensor_ptr);
      std::vector<string> format = {host_tensor_ptr->device_info().host_format_};
      output_format = format[0] == kOpFormat_DEFAULT ? output_format : format;
      builder->SetOutputsFormat(output_format);
      auto iter = format_type.find(host_tensor_ptr->id());
      if (iter != format_type.end()) {
        std::vector<TypeId> output_type = {iter->second};
        builder->SetOutputsDeviceType(output_type);
        AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
      } else {
        std::vector<TypeId> output_type = {selected_kernel_info->GetInputDeviceType(input_index)};
        builder->SetOutputsDeviceType(output_type);
        AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
        format_type[host_tensor_ptr->id()] = output_type[0];
      }
    } else {
      builder->SetOutputsFormat(output_format);
      std::vector<TypeId> output_type = {common::AnfAlgo::GetOutputInferDataType(real_input_node, 0)};
      builder->SetOutputsDeviceType(output_type);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
    }
  }
}

bool RefreshCastAndParamWeightFormat(const AnfNodePtr &input_node, const string &format) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return false;
  }
  if (!input_node->isa<CNode>()) {
    return false;
  }
  auto cast_node = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cast_node);
  if (common::AnfAlgo::GetCNodeName(cast_node) != prim::kPrimCast->name()) {
    return true;
  }
  if (AnfAlgo::IsFeatureMapOutput(cast_node)) {
    return true;
  }
  if (format == kOpFormat_FRACTAL_ZN_RNN || format == kOpFormat_ND_RNN_BIAS) {
    return true;
  }
  auto info_builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(input_node));
  MS_EXCEPTION_IF_NULL(info_builder);
  info_builder->SetInputsFormat({format});
  info_builder->SetOutputsFormat({format});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), cast_node.get());
  auto cast_input_node = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cast_node, 0), 0);
  SetWeightFormat(cast_input_node.first, {format}, cast_node, 0, true);
  return true;
}

TypeId GetInputDeviceType(const CNodePtr &kernel_node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  TypeId type = kTypeUnknown;
  auto input_node = common::AnfAlgo::GetPrevNodeOutput(kernel_node, input_idx).first;
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(input_node->kernel_info());
  if (kernel_info != nullptr && kernel_info->select_kernel_build_info() != nullptr) {
    type = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_idx);
  } else {
    type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
  }
  return type;
}

TypeId GetInputDeviceType(const KernelWithIndex &input_node_with_index) {
  TypeId type = kTypeUnknown;
  auto input_node = input_node_with_index.first;
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(input_node->kernel_info());
  if (kernel_info != nullptr && kernel_info->select_kernel_build_info() != nullptr) {
    type = AnfAlgo::GetOutputDeviceDataType(input_node_with_index.first, input_node_with_index.second);
  } else {
    type = common::AnfAlgo::GetOutputInferDataType(input_node_with_index.first, input_node_with_index.second);
  }
  return type;
}

string InferOutputFormat(const CNodePtr &kernel_node, const std::vector<std::string> &inputs_format) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  // Infer output format from inputs format.
  std::unordered_map<std::string, int> all_input_formats;
  for (const auto &format : inputs_format) {
    all_input_formats[format]++;
  }

  string output_infer_format;
  int max_format_counts = 0;
  for (const auto &it : all_input_formats) {
    if (it.second > max_format_counts) {
      max_format_counts = it.second;
      output_infer_format = it.first;
    }
  }
  if (output_infer_format.empty()) {
    output_infer_format = GetPriorityMatchFormat(kernel_node);
  }
  return output_infer_format;
}

KernelSelectStatus SelectCustomKernelInfo(const CNodePtr &kernel_node, KernelType *kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_type);
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  // Custom op's kernel type can be one of [TBE_KERNEL, AKG_KERNEL] on Ascend
  auto func_type = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrFuncType);
  if (func_type == kCustomTypeTbe) {
    *kernel_type = KernelType::TBE_KERNEL;
  } else if (IsOneOfCustomAkgType(func_type)) {
    *kernel_type = KernelType::AKG_KERNEL;
  } else if (func_type == kCustomTypeAICPU) {
    *kernel_type = KernelType::AICPU_KERNEL;
  } else if (func_type == kCustomTypeAOT) {
    *kernel_type = KernelType::BISHENG_KERNEL;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported func type for Custom operator on Ascend, it should be 'tbe', 'ir_builder', "
                      << "'tvm_compute' or 'hybrid', but got [" << func_type << "]"
                      << " for Custom operator [" << op_name << "]";
  }
  static const std::map<KernelType, kernel::OpImplyType> kKernelImplyTypeMap{
    {KernelType::TBE_KERNEL, kernel::OpImplyType::kImplyTBE},
    {KernelType::AKG_KERNEL, kernel::OpImplyType::kImplyAKG},
    {KernelType::AICPU_KERNEL, kernel::OpImplyType::kImplyAICPU},
    {KernelType::BISHENG_KERNEL, kernel::OpImplyType::kImplyBISHENG}};
  auto it = kKernelImplyTypeMap.find(*kernel_type);
  kernel::OpImplyType imply_type = kernel::OpImplyType::kImplyAKG;
  if (it != kKernelImplyTypeMap.end()) {
    imply_type = it->second;
  }
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, imply_type);
  // Only process Custom op that does not has reg info
  if (op_info_ptr != nullptr) {
    // For BiSheng Ascend operator, continue the process even if the input info is missed.
    if (!(imply_type == kernel::OpImplyType::kImplyBISHENG && op_info_ptr->inputs_ptr().empty())) {
      return kNoMatched;
    }
  }
  // If Custom op has not set reg info, then infer info from inputs
  MS_LOG(WARNING) << "Not find operator information for Custom op[" << op_name << "]. "
                  << "Infer operator information from inputs. For more details, "
                  << "please refer to 'mindspore.ops.Custom' at https://www.mindspore.cn.";
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetKernelType(*kernel_type);
  builder->SetProcessor(kernel::Processor::AICORE);
  builder->SetFusionType(kPatternOpaque);
  builder->SetOpPattern(kernel::OpPattern::kCommonPattern);
  // set inputs info
  std::vector<TypeId> inputs_device_type;
  std::vector<std::string> inputs_format;
  std::vector<kernel::KernelObjectType> inputs_kernel_object_type;
  std::unordered_set<string> all_input_formats;

  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(kernel_node, i);
    MS_EXCEPTION_IF_NULL(input_node);
    if (common::AnfAlgo::IsTupleOutput(input_node)) {
      std::vector<KernelWithIndex> inputs_with_index = common::AnfAlgo::GetRealPrevNodesOutput(kernel_node, i);
      for (auto &j : inputs_with_index) {
        auto type = GetInputDeviceType(j);
        inputs_device_type.emplace_back(type);
        auto format = AnfAlgo::GetOutputFormat(j.first, j.second);
        inputs_format.emplace_back(format);
        all_input_formats.insert(format);
      }
      inputs_kernel_object_type.emplace_back(kernel::KernelObjectType::TUPLE_UNFOLD);
    } else {
      auto type = GetInputDeviceType(kernel_node, i);
      inputs_device_type.emplace_back(type);
      auto format = AnfAlgo::GetPrevNodeOutputFormat(kernel_node, i);
      inputs_format.emplace_back(format);
      all_input_formats.insert(format);
      inputs_kernel_object_type.emplace_back(kernel::KernelObjectType::TENSOR);
    }
  }

  if (all_input_formats.size() > 1) {
    MS_LOG(WARNING) << op_name << " has different input formats, the number of input formats is "
                    << all_input_formats.size();
  }
  builder->SetInputsDeviceType(inputs_device_type);
  builder->SetInputsFormat(inputs_format);
  builder->SetInputsKernelObjectType(inputs_kernel_object_type);
  // set outputs info
  std::vector<TypeId> outputs_device_type;
  std::vector<std::string> outputs_format;
  auto output_infer_format = InferOutputFormat(kernel_node, inputs_format);
  MS_LOG(INFO) << "Outputs of " << op_name << " will use same inferred format: " << output_infer_format;
  size_t output_num = AnfAlgo::GetOutputElementNum(kernel_node);
  for (size_t i = 0; i < output_num; ++i) {
    outputs_device_type.push_back(common::AnfAlgo::GetOutputInferDataType(kernel_node, i));
    outputs_format.push_back(output_infer_format);
  }
  builder->SetOutputsDeviceType(outputs_device_type);
  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsKernelObjectType(
    std::vector<kernel::KernelObjectType>(outputs_format.size(), kernel::KernelObjectType::TENSOR));
  // Set kernel build info to node
  auto build_info = builder->Build();
  MS_LOG(INFO) << "Current node: " << kernel_node->fullname_with_scope() << " selected: " << build_info;
  AnfAlgo::SetSelectKernelBuildInfo(build_info, kernel_node.get());
  SetTensorDeviceInfo(kernel_node);
  return kStatusAllMatched;
}

void FillNoneInKernelInfo(const CNodePtr &kernel_node, std::vector<kernel::KernelBuildInfoPtr> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  // Only process Custom op
  if (!IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    return;
  }
  for (auto &idx : *kernel_info_list) {
    auto build_info = idx;
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(build_info);
    // Fill inputs info. If type or format is None, infer it from inputs
    std::vector<TypeId> inputs_device_type;
    std::vector<std::string> inputs_format;
    for (size_t i = 0; i < build_info->GetInputNum(); ++i) {
      auto type = build_info->GetInputDeviceType(i);
      if (type == TypeId::kMetaTypeNone) {
        type = GetInputDeviceType(kernel_node, i);
      }
      inputs_device_type.push_back(type);
      auto format = build_info->GetInputFormat(i);
      if (format.empty()) {
        format = AnfAlgo::GetPrevNodeOutputFormat(kernel_node, i);
      }
      inputs_format.push_back(format);
    }
    builder->SetInputsDeviceType(inputs_device_type);
    builder->SetInputsFormat(inputs_format);
    // Fill outputs info. If type is None, infer it from abstract, if format is None, infer it from inputs format
    std::vector<TypeId> outputs_device_type;
    std::vector<std::string> outputs_format;
    auto output_infer_format = InferOutputFormat(kernel_node, inputs_format);
    for (size_t i = 0; i < build_info->GetOutputNum(); ++i) {
      auto type = build_info->GetOutputDeviceType(i);
      if (type == TypeId::kMetaTypeNone) {
        type = common::AnfAlgo::GetOutputInferDataType(kernel_node, i);
      }
      outputs_device_type.push_back(type);
      auto format = build_info->GetOutputFormat(i);
      if (format.empty()) {
        format = output_infer_format;
      }
      outputs_format.push_back(format);
    }
    builder->SetOutputsDeviceType(outputs_device_type);
    builder->SetOutputsFormat(outputs_format);
    idx = builder->Build();
  }
}

void ResetPreFixedFormat(const CNodePtr &kernel_node, kernel::KernelBuildInfoPtr *selected_kernel_info) {
  if (!common::AnfAlgo::HasNodeAttr(kAttrFixedInputFormat, kernel_node) ||
      !common::AnfAlgo::HasNodeAttr(kAttrFixedOutputFormat, kernel_node)) {
    return;
  }

  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(*selected_kernel_info);
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetInputsFormat(common::AnfAlgo::GetNodeAttr<std::vector<string>>(kernel_node, kAttrFixedInputFormat));
  builder->SetOutputsFormat(common::AnfAlgo::GetNodeAttr<std::vector<string>>(kernel_node, kAttrFixedOutputFormat));
  *selected_kernel_info = builder->Build();
  MS_LOG(INFO) << "Current node: " << kernel_node->fullname_with_scope()
               << " selected kernel build info after reset fixed format: " << (*selected_kernel_info)->ToString();
  common::AnfAlgo::EraseNodeAttr(kAttrFixedInputFormat, kernel_node);
  common::AnfAlgo::EraseNodeAttr(kAttrFixedOutputFormat, kernel_node);
}
}  // namespace

void RefreshInputParameter(const CNodePtr &kernel_node, const AnfNodePtr &input_kernel_node,
                           const std::string &input_format, size_t input_index) {
  auto input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_kernel_node, 0);
  MS_EXCEPTION_IF_NULL(input_with_index.first);
  auto real_input_node = input_with_index.first;
  MS_EXCEPTION_IF_NULL(real_input_node);
  if (RefreshCastAndParamWeightFormat(real_input_node, input_format)) {
    return;
  }
  if (real_input_node->isa<Parameter>() && !common::AnfAlgo::IsParameterWeight(real_input_node->cast<ParameterPtr>())) {
    return;
  }

  std::vector<std::string> output_format = {input_format};
  SetWeightFormat(real_input_node, output_format, kernel_node, input_index);
}

void SetTensorDeviceInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto selected_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t real_input_num = 0;
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_object_type = selected_kernel_info->GetInputKernelObjectType(input_index);
    if (input_object_type == kernel::KernelObjectType::TUPLE_UNFOLD) {
      std::vector<KernelWithIndex> kernels_with_index =
        common::AnfAlgo::GetRealPrevNodesOutput(kernel_node, input_index);
      for (auto &i : kernels_with_index) {
        RefreshInputParameter(kernel_node, i.first, selected_kernel_info->GetInputFormat(real_input_num),
                              real_input_num);
        ++real_input_num;
      }
    } else {
      auto input_kernel_node = common::AnfAlgo::GetInputNode(kernel_node, input_index);
      MS_EXCEPTION_IF_NULL(input_kernel_node);
      RefreshInputParameter(kernel_node, input_kernel_node, selected_kernel_info->GetInputFormat(real_input_num),
                            real_input_num);
      ++real_input_num;
    }
  }
}

bool CheckFp16(const CNodePtr &cnode, bool check_input) {
  MS_EXCEPTION_IF_NULL(cnode);
  // check output
  for (size_t output_index = 0; output_index < AnfUtils::GetOutputTensorNum(cnode); ++output_index) {
    if (common::AnfAlgo::GetOutputInferDataType(cnode, output_index) == kNumberTypeFloat16) {
      return true;
    }
  }

  // check input
  if (check_input) {
    for (size_t input_index = 0; input_index < common::AnfAlgo::GetInputTensorNum(cnode); ++input_index) {
      std::vector<TypeId> inputs_type = common::AnfAlgo::GetRealPrevNodesOutputInferDataType(cnode, input_index);
      if (std::any_of(inputs_type.cbegin(), inputs_type.cend(),
                      [](const TypeId &type_id) { return type_id == kNumberTypeFloat16; })) {
        return true;
      }
    }
  }

  return false;
}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> ApplyForceFP32Strategy(
  const CNodePtr &cnode, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list,
  bool *raise_fp32) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> result;
  static std::string precision_mode;
  if (precision_mode.empty()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    precision_mode = ms_context->get_param<std::string>(MS_CTX_PRECISION_MODE);
  }
  if (precision_mode != kForceFp32Strategy) {
    return result;
  }

  MS_LOG(DEBUG) << "Apply force_fp32 stratepy for node " << cnode->fullname_with_scope();
  const OriginTypeConverter fp32_converter = [](const TypeId &origin_type) {
    if (origin_type == kNumberTypeFloat16 || origin_type == kNumberTypeFloat) {
      return kNumberTypeFloat32;
    }
    return origin_type;
  };

  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> res;
  if (IsCubeKernel(cnode)) {
    // for cube op, select fp16-in & fp32-out first if origin type is fp16
    res = FilteredKernelInfoByDtype(cnode, kernel_info_list, kDefaultOriginTypeConverter, fp32_converter);
    if (!res.empty() && CheckFp16(cnode, false)) {
      *raise_fp32 = true;
    }
  } else {
    // for vector op, select fp32-in & fp32-out first
    res = FilteredKernelInfoByDtype(cnode, kernel_info_list, fp32_converter, fp32_converter);
    if (!res.empty() && CheckFp16(cnode, true)) {
      *raise_fp32 = true;
    }
  }
  return res;
}

KernelSelectStatus SetMatchedKernelInfo(const CNodePtr &kernel_node,
                                        const std::vector<kernel::KernelBuildInfoPtr> &kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  KernelSelectStatus select_status = kNoMatched;
  if (kernel_info_list.empty()) {
    return kNoMatched;
  }
  bool precision_reduce = false;
  kernel::KernelBuildInfoPtr selected_kernel_info = nullptr;
  // Matched kernel info
  // Apply force_fp32 strategy if it is set in context
  bool raise_fp32 = false;
  auto filtered_kernel_info_list = ApplyForceFP32Strategy(kernel_node, kernel_info_list, &raise_fp32);
  if (!filtered_kernel_info_list.empty()) {
    select_status = raise_fp32 ? kStatusRaisePrecision : kStatusAllMatched;
  } else {
    // Filter kernel info matched with me inferred type
    filtered_kernel_info_list = FilteredKernelInfoByDtype(kernel_node, kernel_info_list);
    if (filtered_kernel_info_list.empty()) {
      // selected kernel info using raised precision or reduce precision
      filtered_kernel_info_list =
        FilterRaisedOrReducePrecisionMatchedKernelInfo(kernel_node, kernel_info_list, &precision_reduce);
      if (filtered_kernel_info_list.empty()) {
        return kNoMatched;
      }
      select_status = precision_reduce ? kStatusReducePrecision : kStatusRaisePrecision;
    } else {
      select_status = kStatusAllMatched;
    }
  }

  // filter object_type and adjust tuple_unfold condition
  MS_LOG(DEBUG) << "Node " << kernel_node->fullname_with_scope() << "'s kernel info list size is "
                << filtered_kernel_info_list.size() << " before object type matching";
  filtered_kernel_info_list = FilteredKernelInfoByObjectType(kernel_node, filtered_kernel_info_list);
  MS_LOG(DEBUG) << "Node " << kernel_node->fullname_with_scope() << "'s kernel info list size is "
                << filtered_kernel_info_list.size() << " after object type matching";
  if (filtered_kernel_info_list.empty()) {
    return kNoMatched;
  }

  selected_kernel_info = ChooseMatchedKernelInfo(kernel_node, filtered_kernel_info_list);
  if (select_status == kStatusReducePrecision || select_status == kStatusRaisePrecision) {
    MS_LOG(INFO) << PrintRaiseOrReducePrecisionSelectedInfo(kernel_node, selected_kernel_info, select_status);
  }

  // Set kernel build info to node
  MS_LOG(DEBUG) << "Current node: " << kernel_node->fullname_with_scope()
                << " selected: " << selected_kernel_info->ToString();
  ResetPreFixedFormat(kernel_node, &selected_kernel_info);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info, kernel_node.get());
  // Set format and data type for input tensor.
  if (common::AnfAlgo::HasNodeAttr(kAttrPynativeNextOpName, kernel_node)) {
    SetCastAndWeightFormat(kernel_node);
  }
  SetTensorDeviceInfo(kernel_node);
  return select_status;
}

std::string KernelInfoCandidateList(const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &ai_core,
                                    const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &ai_cpu) {
  std::ostringstream buffer;
  buffer << "\nAI CORE:\n";
  if (!ai_core.empty()) {
    for (const auto &i : ai_core) {
      buffer << i->ToString();
      buffer << "\n";
    }
  } else {
    buffer << "{}\n";
  }
  buffer << "AI CPU:\n";
  if (!ai_cpu.empty()) {
    for (const auto &i : ai_cpu) {
      buffer << i->ToString();
      buffer << "\n";
    }
    buffer << "\n";
  } else {
    buffer << "{}\n";
  }
  return buffer.str();
}

std::pair<std::string, ExceptionType> CollectNotMatchMessage(
  const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &ai_core,
  const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &ai_cpu, const std::ostringstream &aicore_info,
  const std::ostringstream &aicpu_info, const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto full_name = common::AnfAlgo::GetCNodeName(kernel_node);
  std::stringstream ss;
  ExceptionType etype;
  if (ai_core.empty() && ai_cpu.empty()) {
    ss << "Can not find any available kernel info for: " << full_name
       << ". Maybe the operator is not supported on Ascend platform." << trace::DumpSourceLines(kernel_node);
    etype = NotSupportError;
  } else {
    auto candidates = KernelInfoCandidateList(ai_core, ai_cpu);
    ss << "Can not select a valid kernel info for [" << full_name
       << "] in AI CORE or AI CPU kernel info candidates list.#umsg#Kernel Info Candidates List:#umsg#" << candidates
       << "Please check the given data type or shape:"
       << "\nAI CORE: " << aicore_info.str() << "\nAI CPU: " << aicpu_info.str()
       << "\nFor more details, please refer to 'Kernel Select Failed' at "
          "https://www.mindspore.cn"
       << trace::DumpSourceLines(kernel_node);
    etype = TypeError;
  }
  return std::make_pair(ss.str(), etype);
}

void SetRaiseOrReduceFlag(const CNodePtr &kernel_node, KernelSelectStatus status) {
  if (status == kStatusRaisePrecision) {
    common::AnfAlgo::SetNodeAttr(kAttrPrecisionFlag, MakeValue("raise"), kernel_node);
  } else if (status == kStatusReducePrecision) {
    common::AnfAlgo::SetNodeAttr(kAttrPrecisionFlag, MakeValue("reduce"), kernel_node);
  }
}

void UpdateInputForHighPrecisionOp(const CNodePtr &kernel_node, const KernelBuildInfoBuilderPtr &builder) {
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kHighPrecisionOp.count(op_name) == 0) {
    return;
  }
  auto input_dtypes = AnfAlgo::GetAllInputDeviceTypes(kernel_node);
  auto output_dtypes = AnfAlgo::GetAllOutputDeviceTypes(kernel_node);
  auto has_fp32 = std::any_of(output_dtypes.begin(), output_dtypes.end(),
                              [](TypeId type) { return type == TypeId::kNumberTypeFloat32; });
  if (has_fp32) {
    std::vector<TypeId> new_input_types;
    for (auto type : input_dtypes) {
      if (type == TypeId::kNumberTypeFloat16) {
        new_input_types.push_back(TypeId::kNumberTypeFloat32);
        common::AnfAlgo::SetNodeAttr(kAttrAclHighPrecision, MakeValue(true), kernel_node);
      } else {
        new_input_types.push_back(type);
      }
    }
    builder->SetInputsDeviceType(new_input_types);
    MS_LOG(INFO) << "Update data type for " << kernel_node->fullname_with_scope() << " from " << input_dtypes << " to "
                 << new_input_types;
  }
}

void RestoreCastOpFormat(const CNodePtr &kernel_node, const KernelBuildInfoBuilderPtr &builder) {
  if (common::AnfAlgo::GetCNodeName(kernel_node) != prim::kPrimCast->name() ||
      !common::AnfAlgo::HasNodeAttr(kOriSelectFormat, kernel_node)) {
    return;
  }
  auto ori_format = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kOriSelectFormat);
  builder->SetInputsFormat({ori_format});
  builder->SetOutputsFormat({ori_format});
  MS_LOG(INFO) << "Restore the format of Cast op: " << kernel_node->fullname_with_scope() << " to " << ori_format;
  common::AnfAlgo::EraseNodeAttr(kOriSelectFormat, kernel_node);
}

void UpdateOutputForBNTrainingUpdateOp(const CNodePtr &kernel_node, const KernelBuildInfoBuilderPtr &builder) {
  if (common::AnfAlgo::GetCNodeName(kernel_node) != prim::kPrimBNTrainingUpdate->name()) {
    return;
  }
  builder->SetOutputFormat(kOpFormat_NCHW, 0);
  MS_LOG(INFO) << "set the first output's format of BNTrainingUpdate op: " << kernel_node->fullname_with_scope()
               << " to " << kOpFormat_NCHW;
}

void SetAclKernelInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (!common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, kernel_node)) {
    MS_LOG(DEBUG) << "No kAttrMutableKernel found, cannot set ACL_KERNEL for " << kernel_node->DebugString();
    return;
  }

  KernelType kernel_type = AnfAlgo::GetKernelType(kernel_node);
  if (kernel_type != AICPU_KERNEL && kernel_type != TBE_KERNEL) {
    MS_LOG(INFO) << "Current node doesn't support acl kernel launch! Node info:" << kernel_node->DebugString();
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (mode != kPynativeMode || device_target != kAscendDevice) {
    MS_LOG(INFO) << "Current mode or device doesn't support acl kernel launch! Node info:"
                 << kernel_node->DebugString();
    return;
  }

  if (common::AnfAlgo::IsGraphKernel(kernel_node) || IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    MS_LOG(INFO) << "Current node is graph kernel or custom io! Node info:" << kernel_node->DebugString();
    return;
  }

  // Update node's kernel type to acl.
  auto new_builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(kernel_node));
  MS_EXCEPTION_IF_NULL(new_builder);
  new_builder->SetKernelType(ACL_KERNEL);
  MS_LOG(INFO) << "SUCCESS SET ACL KERNEL FOR" << kernel_node->DebugString();

  // For high precision op
  UpdateInputForHighPrecisionOp(kernel_node, new_builder);
  // Restore the device format of cast op to original value if it has been modified according to conv2d
  RestoreCastOpFormat(kernel_node, new_builder);
  // The reshape type of first output of BNTrainingUpdate is NCH
  // Set the first output's format of BNTrainingUpdate to NCHW to avoid discontinuity between device shapes
  UpdateOutputForBNTrainingUpdateOp(kernel_node, new_builder);

  AnfAlgo::SetSelectKernelBuildInfo(new_builder->Build(), kernel_node.get());
}

void SetDynamicInputSizeAttrBeforeKernelSelect(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall) ||
      common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimPartial)) {
    return;
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode)) {
    return;
  }
  std::vector<int64_t> dyn_input_sizes;
  size_t input_num = cnode->inputs().size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode, i);
    (void)dyn_input_sizes.emplace_back(kernel::CalOutputTupleSize(input_node));
  }
  if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), cnode);
  }
}

void RefreshDynamicInputSizeAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode)) {
    MS_LOG(DEBUG) << "Node has not set kAttrDynInputSizes yet, node: " << cnode->fullname_with_scope();
    return;
  }
  auto dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrDynInputSizes);
  auto input_obj_types = AnfAlgo::GetInputKernelObjectTypes(cnode);
  auto node_inputs_object_type = kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllInputObjectType(cnode));
  // For the condition node is tuple & kernel is tensor
  if (node_inputs_object_type.size() == input_obj_types.size()) {
    for (size_t i = 0; i < input_obj_types.size(); ++i) {
      if (input_obj_types[i] == kernel::KernelObjectType::TENSOR &&
          node_inputs_object_type[i] == kernel::KernelObjectType::TUPLE) {
        dyn_input_sizes[i] = -1;
      }
    }
  }
  // For the condition kernel is tuple
  for (size_t i = 0; i < input_obj_types.size(); ++i) {
    if (input_obj_types[i] == kernel::KernelObjectType::TUPLE) {
      dyn_input_sizes[i] = -1;
    }
  }

  if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), cnode);
  } else {
    common::AnfAlgo::EraseNodeAttr(kAttrDynInputSizes, cnode);
  }
}

std::tuple<KernelSelectStatus, std::string, ExceptionType> SelectKernelInfoWithMsg(const CNodePtr &kernel_node,
                                                                                   KernelType kernel_type) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> aicpu_kernel_info_list;
  std::ostringstream aicore_in_out_info;
  std::ostringstream aicpu_in_out_info;
  std::tuple<KernelSelectStatus, std::string, ExceptionType> result =
    std::make_tuple(kStatusAllMatched, "", NoExceptionType);
  MS_EXCEPTION_IF_NULL(kernel_node);
  SetDynamicInputSizeAttrBeforeKernelSelect(kernel_node);
  if (common::AnfAlgo::IsGraphKernel(kernel_node)) {
    auto func_graph = GetValueNode<FuncGraphPtr>(kernel_node->input(kAnfPrimitiveIndex));
    MS_EXCEPTION_IF_NULL(func_graph);
    SelectGraphKernelInfo(kernel_node, func_graph);
    RefreshDynamicInputSizeAttr(kernel_node);
    return result;
  }
  if (IsPrimitiveCNode(kernel_node, prim::kPrimCallInline)) {
    opt::SelectCallInlineKernelInfo(kernel_node);
    SetTensorDeviceInfo(kernel_node);
    RefreshDynamicInputSizeAttr(kernel_node);
    return result;
  }
  if (common::AnfAlgo::HasNodeAttr(ops::kBatchRank, kernel_node)) {
    std::stringstream ss;
    ss << common::AnfAlgo::GetCNodeName(kernel_node)
       << " does not support 'batch_rank' on Ascend, which means that 'vmap' cannot support "
       << common::AnfAlgo::GetCNodeName(kernel_node) << " on Ascend currently.";
    return std::make_tuple(kNoMatched, ss.str(), NotSupportError);
  }

  if (IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    auto select_status = SelectCustomKernelInfo(kernel_node, &kernel_type);
    if (select_status == kStatusAllMatched) {
      RefreshDynamicInputSizeAttr(kernel_node);
      return result;
    }
  }

  kernel::KernelQuery(kernel_node, &kernel_info_list, kernel_type);
  FillNoneInKernelInfo(kernel_node, &kernel_info_list);
  auto select_status = SetMatchedKernelInfo(kernel_node, kernel_info_list);
  if (IsPrimitiveCNode(kernel_node, prim::kPrimLabelSwitch)) {
    auto selected_kernel_info = ChooseMatchedKernelInfo(kernel_node, kernel_info_list);
    AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info, kernel_node.get());
    // Set format and data type for input tensor.
    SetTensorDeviceInfo(kernel_node);
    select_status = kStatusAllMatched;
  }
  if (select_status != kNoMatched) {
    // If match the conditions of acl, current op run on acl mode.
    SetAclKernelInfo(kernel_node);
  } else {
    // If node can't find valid ai_core kernel info, re-find in ai_cpu kernel info
    GatherInputAndOutputInferType(aicore_in_out_info, kernel_node);
    MS_LOG(DEBUG) << "The node [" << kernel_node->fullname_with_scope()
                  << "] cannot find valid TBE kernel info, try to get ai_cpu kernel info";
    if (common::AnfAlgo::HasNodeAttr(kAttrMeOpName, kernel_node)) {
      std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
      auto me_op_name = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrMeOpName);
      auto primitive = GetCNodePrimitive(kernel_node);
      MS_EXCEPTION_IF_NULL(primitive);
      primitive->set_name(me_op_name);
      // reset full scope name
      kernel_node->set_fullname_with_scope("");
      MS_LOG(INFO) << "Rename op type from " << op_name << " to " << me_op_name << " for op "
                   << kernel_node->fullname_with_scope() << " in aicpu kernel select.";
    }

    kernel::ConvertAttrAndInputBeforeAicpuKernelSelect(kernel_node);
    FallbackOps(kernel_node);
    kernel::AICPUQuery(kernel_node, &aicpu_kernel_info_list);
    select_status = SetMatchedKernelInfo(kernel_node, aicpu_kernel_info_list);
    if (select_status != kNoMatched) {
      common::AnfAlgo::SetNodeAttr(kAttrIsAiCpuKernel, MakeValue(true), kernel_node);
    }
  }
  // The kernel info can not find in ai_cpu kernel lists and ai_core kernel lists
  if (select_status == kNoMatched) {
    GatherInputAndOutputInferType(aicpu_in_out_info, kernel_node);
    std::get<0>(result) = select_status;
    auto [msg, etype] = CollectNotMatchMessage(kernel_info_list, aicpu_kernel_info_list, aicore_in_out_info,
                                               aicpu_in_out_info, kernel_node);
    constexpr int one = 1;
    constexpr int two = 2;
    std::get<one>(result) = msg;
    std::get<two>(result) = etype;
    return result;
  }
  RefreshDynamicInputSizeAttr(kernel_node);
  SetRaiseOrReduceFlag(kernel_node, select_status);
  std::get<0>(result) = select_status;
  return result;
}

KernelSelectStatus SelectKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto [status, msg, etype] = SelectKernelInfoWithMsg(kernel_node);
  if (!msg.empty()) {
    MS_EXCEPTION(etype) << msg;
  }
  return status;
}

void SetAscendKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto kernel_build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_info);

  if (common::AnfAlgo::IsGraphKernel(kernel_node)) {
    return;
  }

  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetOriginDataFormat(kernel_build_info->GetOriginDataFormat());
  builder->SetInputsFormat(kernel_build_info->GetAllInputFormats());
  builder->SetInputsDeviceType(kernel_build_info->GetAllInputDeviceTypes());
  builder->SetOutputsFormat(kernel_build_info->GetAllOutputFormats());
  builder->SetOutputsDeviceType(kernel_build_info->GetAllOutputDeviceTypes());
  builder->SetOpPattern(kernel_build_info->op_pattern());
  builder->SetFusionType(kernel_build_info->fusion_type());

  auto new_kernel_type = kernel_type;
  auto new_processor = kernel_build_info->processor();
  if (kernel_type == UNKNOWN_KERNEL_TYPE) {
    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> aicpu_kernel_info_list;
    kernel::KernelQuery(kernel_node, &kernel_info_list, kernel_type);
    auto select_status = SetMatchedKernelInfo(kernel_node, kernel_info_list);
    if (select_status != kNoMatched) {
      new_kernel_type = TBE_KERNEL;
      new_processor = kernel::Processor::AICORE;
      MS_LOG(INFO) << kernel_node->fullname_with_scope() << " uses TBE_KERNEL";
    } else {
      kernel::AICPUQuery(kernel_node, &aicpu_kernel_info_list);
      select_status = SetMatchedKernelInfo(kernel_node, aicpu_kernel_info_list);
      if (select_status != kNoMatched) {
        new_kernel_type = AICPU_KERNEL;
        new_processor = kernel::Processor::AICPU;
        MS_LOG(INFO) << kernel_node->fullname_with_scope() << " uses AICPU_KERNEL";
      }
    }
  }
  if (new_kernel_type == UNKNOWN_KERNEL_TYPE) {
    new_kernel_type = AKG_KERNEL;
    new_processor = kernel::Processor::AICORE;
    MS_LOG(INFO) << kernel_node->fullname_with_scope() << " uses AKG_KERNEL";
  }
  builder->SetKernelType(new_kernel_type);
  builder->SetProcessor(new_processor);
  kernel_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
}

// After operator selection in graph optimization, new nodes will be added, select kernel info for those nodes
// check whether the node has completed the operator selection. If not, the operator
// selection needs to be performed to set kernel info.
void SelectKernelInfoAfterKernelSelect(const std::vector<CNodePtr> &nodes) {
  profiler::CollectHostInfo("Ascend", "Operator Compilation", "SelectKernelInfo", 0, 0, 0);
  // Check whether the node has completed kernel selection.
  for (const auto &node : nodes) {
    auto kernel_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    if (kernel_info != nullptr && kernel_info->valid()) {
      continue;
    }

    // Kernel selection process.
    auto [status, msg, etype] = SelectKernelInfoWithMsg(node);
    if (status == device::ascend::kNoMatched) {
      auto graph = AnfAlgo::FetchKernelGraph(node.get());
      std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
      HandleKernelSelectFailure(graph, node, failure_info);
    }
  }
  profiler::CollectHostInfo("Ascend", "Operator Compilation", "SelectKernelInfo", 0, 0, 1);
}

std::string TryBackoffCpu(const KernelGraphPtr &graph, const CNodePtr &node,
                          const std::pair<std::string, ExceptionType> &failure_info) {
  // The Pynative_mode and task_sink does not support the backoff ability.
  if (!AnfAlgo::IsEnableKernelSelectBackoff(graph)) {
    return failure_info.first;
  }
  // Ref op does not support the backoff ability.
  auto op_info = kernel::tbe::TbeDynamicShapeUtil::FindOp(common::AnfAlgo::GetCNodeName(node), node);
  if (op_info != nullptr && op_info->is_ref()) {
    return failure_info.first;
  }

  if (graph->is_graph_run_mode()) {
    return failure_info.first +
           "\nThe operator is not supported in task sink mode. You can try to export "
           "environment variable GRAPH_OP_RUN to 1 to execute this operator.";
  }

  MS_LOG(INFO) << "Try to use backoff CPU kernel, node:" << node->fullname_with_scope();
  // Erease  kAttrDynInputSizes before cpu kernel select, since cpu may expand it according to kAttrDynInputSizes
  // and make wrong choose, for example, the TupleToTensor op
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, node)) {
    common::AnfAlgo::EraseNodeAttr(kAttrDynInputSizes, node);
  }
  auto [cpu_msg, cpu_etype] = device::cpu::SetKernelInfoWithMsg(node);
  if (cpu_msg.empty()) {
    SetTensorDeviceInfo(node);
    AnfAlgo::SetKernelSelectBackoffInfo(node, failure_info);
  } else {
    std::ostringstream oss;
    oss << "Ascend operator selection failed info: " << failure_info.first
        << "\nCPU operator selection failed type: " << cpu_etype << ". failed info: " << cpu_msg;
    return oss.str();
  }
  return "";
}

void HandleKernelSelectFailure(const KernelGraphPtr &graph, const CNodePtr &node,
                               const std::pair<std::string, ExceptionType> &failure_info) {
  auto msg = TryBackoffCpu(graph, node, failure_info);
  if (!msg.empty()) {
    MS_EXCEPTION(failure_info.second) << msg;
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
