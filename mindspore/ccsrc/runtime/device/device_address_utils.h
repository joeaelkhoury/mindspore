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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_COMMON_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_COMMON_UTILS_H_

#include <vector>
#include <string>
#include "runtime/hardware/device_context.h"
#include "runtime/pynative/op_compiler.h"

namespace mindspore {
using device::DeviceContext;
namespace runtime {
// Extract the methods related to DeviceAddress in GraphCompiler to the DeviceAddressUtils class.
class BACKEND_EXPORT DeviceAddressUtils {
 public:
  static void CreateParameterDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static std::vector<device::DeviceAddressPtr> CreateDeviceAddressForTensorValue(const DeviceContext *device_context,
                                                                                 const ValuePtr &node_value,
                                                                                 size_t output_idx,
                                                                                 const ValueNodePtr &value_node);
  static void CreateValueNodeDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static void CreateKernelOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph,
                                              bool is_gradient_out);
  static vector<device::DeviceAddressPtr> CreateGraphOutputDeviceAddress(const OpCompilerInfoPtr &op_compiler_info,
                                                                         const abstract::AbstractBasePtr &out_abstract);
  static void CreateKernelWorkspaceDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  static void CreateDeviceAddressByMapTensorNode(const DeviceContext *device_context, const AnfNodePtr &node,
                                                 size_t index);
  static void UpdateDeviceAddressForInplaceNode(const KernelGraphPtr &graph);
  static void UpdateDeviceAddress(const session::AnfWithOutIndex &cur_pair,
                                  const session::AnfWithOutIndex &origin_pair);
  static void UpdateDeviceAddressForRefNode(const KernelGraphPtr &graph);
  static device::DeviceAddressPtr CloneEmptyDeviceAddress(const device::DeviceAddressPtr &old_device_address,
                                                          const DeviceContext *device_context);
  static void CreateGraphOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);

  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const tensor::TensorPtr &tensor, const std::string &input_name);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const std::optional<tensor::TensorPtr> &val,
                                                     const std::string &input_name);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, const ScalarPtr &scalar_value,
                                                     const std::string &input_name);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context,
                                                     const StringImmPtr &string_imm, const std::string &input_name);
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, const TypePtr &type,
                                                     const std::string &input_name);

  template <typename T>
  static device::DeviceAddressPtr CreateInputAddress(const DeviceContext *device_context, const T &t,
                                                     const std::string &input_name) {
    return nullptr;
  }

  static void CopyConstantToDevice(const device::DeviceContext *device_context,
                                   const device::DeviceAddressPtr &device_address, const std::string &op_name);
  static std::vector<device::DeviceAddressPtr> CreateInputTensorAddress(const DeviceContext *device_context,
                                                                        const std::vector<tensor::TensorPtr> &tensors,
                                                                        const std::string &input_name,
                                                                        bool need_ret_address = false);

  static void CreateOutputTensorAddress(const DeviceContext *device_context, const tensor::TensorPtr &tensor,
                                        const std::string &output_name, const std::string &format = "");

  static device::DeviceAddressPtr CreateOutputTensorAddress(const DeviceContext *device_context,
                                                            const tensor::TensorPtr &tensor,
                                                            const pynative::DeviceAddressPromisePtr &promise,
                                                            const std::string &output_name,
                                                            const std::string &format = "");

  static device::DeviceAddressPtr CreateWorkspaceAddress(const DeviceContext *device_context,
                                                         const size_t &workspace_size);
};
device::DeviceAddressPtr GetInputAddressForRef(const AnfNodePtr &node, const OpCompilerInfoPtr &op_compiler_info);
device::DeviceAddressPtr GetOutputAddressForRef(const AnfNodePtr &node, const OpCompilerInfoPtr &op_compiler_info,
                                                size_t output_index);
size_t GetTensorDeviceSize(const DeviceContext *device_context, const AnfNodePtr &node, const ShapeVector &shape,
                           std::string format, TypeId dtype, size_t output_index);
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_COMMON_UTILS_H_
