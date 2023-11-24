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

#include "plugin/device/ascend/hal/device/dump/kernel_dumper.h"
#include <algorithm>
#include <utility>
#include "google/protobuf/util/json_util.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif
#include "acl/acl_rt.h"
#include "ops/ascend_op_name.h"
#include "include/common/utils/anfalgo.h"
#include "graph/def_types.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "runtime/kernel.h"
#include "plugin/device/ascend/hal/device/ge_types_convert.h"
#include "proto/op_mapping_info.pb.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/dump/dumper_base.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace {
static constexpr uint64_t kOpDebugMemorySize = 2048;
const size_t kDebugP2pSize = 8UL;
constexpr size_t kHcomIndex = 4;
}  // namespace
DUMPER_REG(kAscendDevice, KernelDumper);
std::mutex KernelDumper::dumper_mutex_;
std::map<rtStream_t, std::unique_ptr<OpDebugTask>> KernelDumper::op_debug_tasks;
std::map<uint32_t, bool> KernelDumper::is_data_map;
std::map<std::string, std::string> KernelDumper::stream_task_graphs;

OpDebugTask::~OpDebugTask() {
  if (op_debug_addr != nullptr) {
    (void)aclrtFree(op_debug_addr);
    op_debug_addr = nullptr;
  }
  if (new_op_debug_addr != nullptr) {
    (void)aclrtFree(new_op_debug_addr);
    new_op_debug_addr = nullptr;
  }
}

KernelDumper::~KernelDumper() {
  if (proto_dev_mem_ != nullptr) {
    (void)aclrtFree(proto_dev_mem_);
    proto_dev_mem_ = nullptr;
  }
  if (proto_size_dev_mem_ != nullptr) {
    (void)aclrtFree(proto_size_dev_mem_);
    proto_size_dev_mem_ = nullptr;
  }
  if (p2p_debug_addr_ != nullptr) {
    (void)aclrtFree(p2p_debug_addr_);
    p2p_debug_addr_ = nullptr;
  }
  if (dev_load_mem_ != nullptr) {
    (void)aclrtFree(dev_load_mem_);
    dev_load_mem_ = nullptr;
  }
}

void KernelDumper::OpLoadDumpInfo(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto stream = AscendStreamMng::GetInstance().GetStream(AnfAlgo::GetStreamId(kernel));
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  if (DumpJsonParser::GetInstance().op_debug_mode() > 0) {
    auto rt_ret = aclrtSynchronizeStreamWithTimeout(stream, -1);
    if (rt_ret != ACL_ERROR_RT_AICORE_OVER_FLOW) {
      return;
    }
  }

  if (!KernelNeedDump(kernel)) {
    return;
  }
  if (common::AnfAlgo::IsNonTaskOp(kernel)) {
    MS_LOG(WARNING) << "[KernelDumper] kernel [" << kernel->UniqueName() << "] is a non-task node, skip dump.";
    return;
  }

  std::lock_guard<std::mutex> lock(dump_mutex_);
  aicpu::dump::OpMappingInfo dump_info;
  SetOpMappingInfo(NOT_NULL(&dump_info), kernel);

  DumpJsonParser::GetInstance().MatchKernel(kernel->fullname_with_scope());
  aicpu::dump::Task task;
  ConstructDumpTask(NOT_NULL(kernel), NOT_NULL(&task));
  MS_EXCEPTION_IF_NULL(dump_info.mutable_task());
  dump_info.mutable_task()->Add(std::move(task));
  ExecutorDumpOp(dump_info, stream);
  // graph id may changed in Unload
  graph_id_ = AnfAlgo::GetGraphId(kernel.get());
  std::string stream_task_id = std::to_string(stream_id_) + std::to_string(task_id_);
  KernelDumper::stream_task_graphs.emplace(stream_task_id, kernel->fullname_with_scope());
  MS_LOG(INFO) << "[KernelDumper] Get runtime info graph_id:" << graph_id_ << " stream_id:" << stream_id_
               << " task_id:" << task_id_ << " fullname:" << kernel->fullname_with_scope();
}

void KernelDumper::SetOpMappingInfo(NotNull<aicpu::dump::OpMappingInfo *> dump_info, const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  dump_info->set_dump_path(dump_path_);
  dump_info->set_model_name(net_name_);
  dump_info->set_dump_step(iteration_);
  FuncGraphPtr f_graph = kernel->func_graph();
  auto kernel_graph_ = f_graph->cast<KernelGraphPtr>();
  auto root_graph_id = kernel_graph_->root_graph_id();
  dump_info->set_model_id(root_graph_id);
  dump_info->set_flag(kAicpuLoadFlag);

  auto input_ctrl_tensors = kernel_graph_->device_loop_control_tensors();
  SetdeviceLoopcontrolTensors(input_ctrl_tensors, dump_info);
}

void KernelDumper::SetOpMappingInfo(NotNull<aicpu::dump::OpMappingInfo *> dump_info,
                                    const std::shared_ptr<HcclTaskInfo> &task_info) {
  MS_EXCEPTION_IF_NULL(task_info);
  dump_info->set_dump_path(dump_path_);
  dump_info->set_model_name(net_name_);
  dump_info->set_dump_step(iteration_);
  dump_info->set_model_id(task_info->graph_id());
  dump_info->set_flag(kAicpuLoadFlag);

  auto input_ctrl_tensors = task_info->device_loop_control_tensors();
  SetdeviceLoopcontrolTensors(input_ctrl_tensors, dump_info);
}

void KernelDumper::Init() {
  if (already_print_) {
    return;
  }
  if (initialed_) {
    MS_LOG(INFO) << "[KernelDumper] already initialized, no need to do it again.";
    already_print_ = true;
    return;
  }
  initialed_ = true;
  op_debug_mode_ = DumpJsonParser::GetInstance().op_debug_mode();
  net_name_ = DumpJsonParser::GetInstance().net_name();
  iteration_ = DumpJsonParser::GetInstance().iteration_string();
  dump_path_ = DumpJsonParser::GetInstance().path();
  if (iteration_ == "all") {
    iteration_ = "0-" + std::to_string(ULONG_MAX);
  }
  uint32_t rank_id = 0;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    // get actual rank id if it's distribution training case.
    if (!CommManager::GetInstance().GetRankID(kHcclWorldGroup, &rank_id)) {
      MS_LOG(INFO) << "Failed to get rank id.";
    }
  }
  dump_path_ = dump_path_ + "/rank_" + std::to_string(rank_id) + "/";
  MS_LOG(INFO) << "[KernelDumper] dump_path: " << dump_path_;
  uint32_t op_debug_mode = DumpJsonParser::GetInstance().op_debug_mode();
  auto iter = kOverflowModeStr.find(op_debug_mode);
  if (iter == kOverflowModeStr.end()) {
    MS_LOG(EXCEPTION) << "[KernelDumper] Invalid op debug mode " << op_debug_mode;
  }
  if (op_debug_mode != kNoOverflow) {
    is_op_debug_ = true;
  }
}

void KernelDumper::DumpHcclOutput(const std::shared_ptr<HcclTaskInfo> &task_info, const rtStream_t &stream) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    MS_LOG(INFO) << "Async dump is not enabled, no need to insert dump op for hccl task.";
    return;
  }
  if (!DumpJsonParser::GetInstance().OutputNeedDump()) {
    MS_LOG(INFO) << "Skip dump output";
    return;
  }
  uint32_t op_debug_mode = DumpJsonParser::GetInstance().op_debug_mode();
  if (op_debug_mode != kNoOverflow) {
    MS_LOG(WARNING) << "HCCL operator is not supported overflow detection!";
    return;
  }
  MS_EXCEPTION_IF_NULL(task_info);
  auto ori_name = task_info->op_name();
  auto idx = ori_name.rfind("_");
  auto op_name = ori_name.substr(0, idx);
  if (!DumpJsonParser::GetInstance().NeedDump(op_name)) {
    MS_LOG(INFO) << "OP: " << op_name << " is not in kernels, no need to dump.";
    return;
  }

  Init();
  aicpu::dump::OpMappingInfo dump_info;
  SetOpMappingInfo(NOT_NULL(&dump_info), task_info);

  aicpu::dump::Task dump_task_obj;
  aicpu::dump::Task *dump_task = &dump_task_obj;
  dump_task->set_end_graph(false);
  uint32_t task_id_rt = 0;
  uint32_t stream_id_rt = 0;
  rtError_t rt_ret = rtGetTaskIdAndStreamID(&task_id_rt, &stream_id_rt);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] rtGetTaskIdAndStreamID failed, rt_ret = " << rt_ret;
    return;
  }
  MS_LOG(INFO) << "[KernelDumper] Get from rtGetTaskIdAndStreamID, task_id:" << task_id_rt
               << " stream_id:" << stream_id_rt;
  dump_task->set_task_id(task_id_rt);
  dump_task->set_stream_id(stream_id_rt);
  MS_EXCEPTION_IF_NULL(dump_task->mutable_op());
  dump_task->mutable_op()->set_op_name(op_name);
  auto op_type = task_info->hccl_type().substr(kHcomIndex);
  dump_task->mutable_op()->set_op_type(op_type);

  DumpKernelOutput(task_info, NOT_NULL(dump_task));
  MS_EXCEPTION_IF_NULL(dump_info.mutable_task());
  dump_info.mutable_task()->Add(std::move(dump_task_obj));
  ExecutorDumpOp(dump_info, stream);
}

void KernelDumper::ExecutorDumpOp(const aicpu::dump::OpMappingInfo &op_mapping_info, const rtStream_t &stream_) {
  MS_EXCEPTION_IF_NULL(stream_);
  std::string proto_msg;
  const size_t proto_size = op_mapping_info.ByteSizeLong();
  const bool ret = op_mapping_info.SerializeToString(&proto_msg);
  if ((!ret) || (proto_size == 0U)) {
    MS_LOG(ERROR) << "[KernelDumper] Protobuf Failed, proto_size is: " << proto_size;
    return;
  }
  std::string proto_json;
  (void)google::protobuf::util::MessageToJsonString(op_mapping_info, &proto_json);
  rtError_t rt_ret = aclrtMalloc(&proto_dev_mem_, proto_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] Call rt api rtMalloc failed, ret = " << rt_ret;
    return;
  }

  rt_ret = aclrtMemcpy(proto_dev_mem_, proto_size, proto_msg.c_str(), proto_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] Call aclrtMemcpy failed, ret = " << rt_ret;
    return;
  }

  rt_ret = aclrtMalloc(&proto_size_dev_mem_, sizeof(size_t), ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] Call rt api rtMalloc failed, ret = " << rt_ret;
    return;
  }
  rt_ret = aclrtMemcpy(proto_size_dev_mem_, sizeof(size_t), &proto_size, sizeof(size_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] Call aclrtMemcpy failed, ret = " << rt_ret;
    return;
  }

  constexpr uint32_t io_addr_num = 2U;
  constexpr uint32_t args_size = sizeof(aicpu::AicpuParamHead) + (io_addr_num * sizeof(uint64_t));
  uint8_t args[args_size] = {};
  size_t args_pos = 0U;
  aicpu::AicpuParamHead &param_head = *(static_cast<aicpu::AicpuParamHead *>(static_cast<void *>(&args[args_pos])));
  args_pos += sizeof(aicpu::AicpuParamHead);
  param_head.length = args_size;
  param_head.ioAddrNum = io_addr_num;
  *(static_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) = ::ge::PtrToValue(proto_dev_mem_);
  args_pos += sizeof(uint64_t);
  *(static_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) = ::ge::PtrToValue(proto_size_dev_mem_);
  rt_ret = rtCpuKernelLaunch(nullptr, kDumpKernelsDumpOp,
                             1U,  // blockDim default 1
                             &args[0], args_size,
                             nullptr,  // no need smDesc
                             stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] Call rt api rtCpuKernelLaunch Failed, rt_ret = " << rt_ret;
    return;
  }
  auto rt_sync = aclrtSynchronizeStreamWithTimeout(stream_, -1);
  if (rt_sync != ACL_ERROR_NONE && rt_sync != ACL_ERROR_RT_AICORE_OVER_FLOW) {
    MS_LOG(WARNING) << "[KernelDumper] Call rt api aclrtSynchronizeStreamWithTimeout Failed, rt_sync = " << rt_sync;
  }
}

void KernelDumper::ConstructDumpTask(NotNull<const CNodePtr &> kernel, NotNull<aicpu::dump::Task *> dump_task) {
  dump_task->set_end_graph(false);
  rtError_t rt_ret = rtGetTaskIdAndStreamID(&task_id_, &stream_id_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] rtGetTaskIdAndStreamID failed, rt_ret = " << rt_ret;
    return;
  }
  dump_task->set_task_id(task_id_);
  dump_task->set_stream_id(stream_id_);
  MS_EXCEPTION_IF_NULL(dump_task->mutable_op());
  dump_task->mutable_op()->set_op_name(kernel->fullname_with_scope());
  dump_task->mutable_op()->set_op_type(common::AnfAlgo::GetCNodeName(kernel.get()));

#ifndef ENABLE_SECURITY
  DumpKernelOutput(kernel, dump_task);
  DumpKernelInput(kernel, dump_task);
#endif
}

void KernelDumper::DumpKernelOutput(const CNodePtr &kernel, NotNull<aicpu::dump::Task *> task) const {
  if (!DumpJsonParser::GetInstance().OutputNeedDump()) {
    MS_LOG(INFO) << "Skip dump output";
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  if (HasAbstractMonad(kernel)) {
    MS_LOG(WARNING) << "Skip Monad node output:" << kernel->fullname_with_scope();
    return;
  }
  auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
  for (size_t i = 0; i < output_size; ++i) {
    auto data_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
    auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel, i);
    auto output_origin_shape = common::AnfAlgo::GetOutputInferShape(kernel, i);

    aicpu::dump::Output output;
    output.set_data_type(static_cast<int>(GeTypesConvert::GetGeDataType(data_type)));
    output.set_format(static_cast<int>(GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    SetDumpShape(output_shape, NOT_NULL(output.mutable_shape()));
    SetDumpShape(output_origin_shape, NOT_NULL(output.mutable_origin_shape()));

    output.set_original_output_format(
      static_cast<int>(GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    // device address data size
    auto address = AnfAlgo::GetOutputAddr(kernel, i);
    output.set_size(address->GetSize());
    output.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address->GetPtr())));
    MS_EXCEPTION_IF_NULL(task->mutable_output());
    task->mutable_output()->Add(std::move(output));
  }
}

void KernelDumper::DumpKernelOutput(const std::shared_ptr<HcclTaskInfo> &task_info, NotNull<aicpu::dump::Task *> task) {
  MS_EXCEPTION_IF_NULL(task_info);
  auto address = task_info->get_output_addr_list();
  for (size_t i = 0; i < task_info->output_num(); ++i) {
    aicpu::dump::Output output;
    auto hccl_dtype = static_cast<HcclDataType>(task_info->data_type());
    auto ge_dtype = device::ascend::GeTypesConvert::TransHcclDataTypeToGeDataType(hccl_dtype);
    output.set_data_type(static_cast<int>(ge_dtype));
    MS_LOG(INFO) << "Output data_type: " << output.data_type();
    auto output_shape = task_info->hccl_kernel_output_shape_list()[i];
    auto origin_shape = task_info->hccl_host_output_shape_list()[i];
    SetDumpShape(output_shape, NOT_NULL(output.mutable_shape()));
    SetDumpShape(origin_shape, NOT_NULL(output.mutable_origin_shape()));

    auto output_format = task_info->data_format()[i];
    output.set_format(
      static_cast<int>(device::ascend::GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    output.set_original_output_format(
      static_cast<int>(device::ascend::GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    MS_EXCEPTION_IF_NULL(address[i]);
    output.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address[i])));
    MS_LOG(INFO) << "Output address: " << output.address();
    size_t output_size = task_info->output_size_list()[i];
    output.set_size(output_size);
    MS_LOG(INFO) << "Output size: " << output_size;
    MS_EXCEPTION_IF_NULL(task->mutable_output());
    task->mutable_output()->Add(std::move(output));
  }
}

void KernelDumper::DumpKernelInput(const CNodePtr &kernel, NotNull<aicpu::dump::Task *> task) const {
  if (!DumpJsonParser::GetInstance().InputNeedDump()) {
    MS_LOG(INFO) << "Skip dump input";
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  if (common::AnfAlgo::IsNodeInputContainMonad(kernel)) {
    MS_LOG(WARNING) << "Skip Monad node:" << kernel->fullname_with_scope();
    return;
  }
  auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);

  for (size_t i = 0; i < input_size; ++i) {
    auto real_index = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, i);
    if (common::AnfAlgo::IsNoneInput(kernel, real_index)) {
      continue;
    }
    aicpu::dump::Input input;
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, real_index);
    auto input_node = input_node_with_index.first;
    auto input_index = input_node_with_index.second;
    std::string output_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(input_node, input_index);
    if (output_type == kTypeUnknown) {
      MS_LOG(WARNING) << "[KernelDumper] It is not suggested to use a lonely weight parameter as the output of graph";
      output_type = common::AnfAlgo::GetOutputInferDataType(input_node, input_index);
    }
    auto output_shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    auto output_origin_shape = common::AnfAlgo::GetOutputInferShape(input_node, input_index);

    input.set_data_type(static_cast<int>(GeTypesConvert::GetGeDataType(output_type)));
    input.set_format(static_cast<int>(GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    SetDumpShape(output_shape, NOT_NULL(input.mutable_shape()));
    SetDumpShape(output_origin_shape, NOT_NULL(input.mutable_origin_shape()));

    // device  address data size
    auto address = AnfAlgo::GetPrevNodeOutputAddr(kernel, real_index);
    if (address == nullptr) {
      MS_LOG(INFO) << "[KernelDumper] The output address prev of node: " << kernel->fullname_with_scope()
                   << " is null.";
      return;
    }
    input.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address->GetPtr())));
    input.set_size(address->GetSize());
    MS_EXCEPTION_IF_NULL(task->mutable_input());
    task->mutable_input()->Add(std::move(input));
  }
}

std::string KernelDumper::StripUniqueId(const std::string node_name) const {
  size_t last_underscore = node_name.find_last_of('_');
  std::string stripped_node_name;
  if (last_underscore == string::npos) {
    MS_LOG(ERROR) << "Could not strip unique ID from " << node_name;
    stripped_node_name = node_name;
  } else {
    stripped_node_name = node_name.substr(0, last_underscore);
  }
  return stripped_node_name;
}

void KernelDumper::SetOpMappingInfoRegister(NotNull<aicpu::dump::OpMappingInfo *> dump_info, const CNodePtr &kernel) {
  dump_info->set_dump_path(dump_path_);
  dump_info->set_model_name(overflow_dump_filename);
  dump_info->set_dump_step(iteration_);
  dump_info->set_flag(kAicpuLoadFlag);

  MS_EXCEPTION_IF_NULL(kernel);
  FuncGraphPtr f_graph = kernel->func_graph();
  auto kernel_graph_ = f_graph->cast<KernelGraphPtr>();

  auto input_ctrl_tensors = kernel_graph_->device_loop_control_tensors();
  SetdeviceLoopcontrolTensors(input_ctrl_tensors, dump_info);
}

#ifndef ENABLE_SECURITY
void KernelDumper::MallocP2PDebugMem(const void *const op_debug_addr) {
  const uint64_t debug_addrs_tmp = ::ge::PtrToValue(op_debug_addr);
  int64_t value = 0;
  rtError_t rt_ret = rtGetRtCapability(FEATURE_TYPE_MEMORY, static_cast<int32_t>(MEMORY_INFO_TS_LIMITED), &value);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[KernelDumper] Call rt api rtGetRtCapability failed, ret = " << rt_ret;
  }
  auto memory_type = (value == static_cast<int64_t>(RT_CAPABILITY_SUPPORT)) ? RT_MEMORY_TS : RT_MEMORY_HBM;
  rt_ret = rtMalloc(&p2p_debug_addr_, kDebugP2pSize, memory_type, 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] Call rtMalloc failed, ret = " << rt_ret;
    return;
  }
  rt_ret =
    aclrtMemcpy(p2p_debug_addr_, sizeof(uint64_t), &debug_addrs_tmp, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "[KernelDumper] Call aclrtMemcpy failed, ret = " << rt_ret;
  }
}

void KernelDumper::OpDebugRegisterForStream(const CNodePtr &kernel) {
  uint32_t op_debug_mode = DumpJsonParser::GetInstance().op_debug_mode();
  auto iter = kOverflowModeStr.find(op_debug_mode);
  if (iter == kOverflowModeStr.end()) {
    MS_LOG(EXCEPTION) << "Invalid op debug mode " << op_debug_mode;
  }
  if (op_debug_mode == kNoOverflow) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  std::lock_guard<std::mutex> lock(dumper_mutex_);
  auto stream = AscendStreamMng::GetInstance().GetStream(AnfAlgo::GetStreamId(kernel));
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  if (KernelDumper::op_debug_tasks.find(stream) != KernelDumper::op_debug_tasks.end()) {
    return;
  } else {
    std::string stream_id = std::to_string(AnfAlgo::GetStreamId(kernel));
    KernelDumper::stream_task_graphs.emplace(stream_id, "KernelDumper");
    auto graph_id = AnfAlgo::GetGraphId(kernel.get());
    if (KernelDumper::is_data_map.find(graph_id) != KernelDumper::is_data_map.end()) {
      return;
    }
    bool is_data_map_ = false;
    KernelGraphPtr kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(kernel->func_graph());
    const auto kernels = kernel_graph->execution_order();
    is_data_map_ = std::any_of(kernels.cbegin(), kernels.cend(), [](const auto &kernel) {
      return kernel->fullname_with_scope().find("InitDataSetQueue") != std::string::npos;
    });
    if (is_data_map_) {
      return;
    }
    KernelDumper::is_data_map.emplace(graph_id, is_data_map_);
    auto &op_debug_task = KernelDumper::op_debug_tasks[stream];
    op_debug_task = std::make_unique<OpDebugTask>();

    int64_t value = 0;
    rtError_t rt_ret = rtGetRtCapability(FEATURE_TYPE_MEMORY, static_cast<int32_t>(MEMORY_INFO_TS_LIMITED), &value);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "[KernelDumper] Call rt api rtGetRtCapability failed, ret = " << rt_ret;
    }
    auto memory_type = (value == static_cast<int64_t>(RT_CAPABILITY_SUPPORT)) ? RT_MEMORY_TS : RT_MEMORY_HBM;
    rt_ret = rtMalloc(&op_debug_task->op_debug_addr, kOpDebugMemorySize, memory_type, 0);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "[KernelDumper] Call rt api rtMalloc failed, ret = " << rt_ret;
    }
    rt_ret = rtDebugRegisterForStream(stream, op_debug_mode, op_debug_task->op_debug_addr,
                                      &op_debug_task->debug_stream_id, &op_debug_task->debug_task_id);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "[KernelDumper] Call rtDebugRegisterForStream failed, ret = " << rt_ret;
    }
    MallocP2PDebugMem(op_debug_task->op_debug_addr);
  }

  aicpu::dump::OpMappingInfo dump_info;
  SetOpMappingInfoRegister(NOT_NULL(&dump_info), kernel);
  SetOpDebugMappingInfo(NOT_NULL(&dump_info), KernelDumper::op_debug_tasks[stream]->debug_task_id,
                        KernelDumper::op_debug_tasks[stream]->debug_stream_id, p2p_debug_addr_);
  RtLoadDumpData(dump_info, &dev_load_mem_);
}

void KernelDumper::OpDebugUnregisterForStream() {
  for (auto iter = KernelDumper::op_debug_tasks.begin(); iter != KernelDumper::op_debug_tasks.end(); iter++) {
    rtError_t rt_ret = rtDebugUnRegisterForStream(iter->first);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "[KernelDumper] Call rtDebugUnRegisterForStream failed, ret = " << rt_ret;
    }
  }
  KernelDumper::op_debug_tasks.clear();
  KernelDumper::stream_task_graphs.clear();
  OverflowDumper::Clear();
}
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
