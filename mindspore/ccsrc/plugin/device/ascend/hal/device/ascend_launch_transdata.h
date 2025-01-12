/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LAUNCH_TRANSDATA_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LAUNCH_TRANSDATA_H_

#include <utility>
#include <vector>
#include <memory>
#include <string>
#include "plugin/device/ascend/hal/device/ascend_launch_kernel.h"

namespace mindspore::device::ascend {
class AscendLaunchTransData : public AscendLaunchKernel {
 public:
  AscendLaunchTransData(void *stream, TypeId dtype, size_t total_size, std::string src_format, std::string dst_format,
                        ShapeVector host_shape, int64_t groups)
      : AscendLaunchKernel(stream),
        dtype_(dtype),
        total_size_(total_size),
        transdata_graph_(nullptr),
        input_addr_(nullptr),
        src_format_(std::move(src_format)),
        dst_format_(std::move(dst_format)),
        shape_(std::move(host_shape)),
        groups_(groups),
        kernel_type_(KernelType::UNKNOWN_KERNEL_TYPE) {}

  ~AscendLaunchTransData() override = default;

  void SetInputAddr(uint8_t *input_addr) override { input_addr_ = input_addr; }
  void FreeDeviceMem(void *addr) override;
  size_t AlignSizeForLaunchKernel(size_t size) override;
  uint8_t *AllocDeviceMem(size_t size) override;
  void KernelSelect(const std::shared_ptr<session::KernelGraph> &kernel_graph) override;
  void KernelBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) override;

  void LaunchOpKernel() override;
  void FreeLaunchDeviceMem() override;
  void FreeWorkspaceDeviceMem();

 protected:
  TypeId dtype_;
  size_t total_size_;
  std::shared_ptr<session::KernelGraph> transdata_graph_;
  uint8_t *input_addr_;
  std::string src_format_;
  std::string dst_format_;
  ShapeVector shape_;
  int64_t groups_;

 private:
  std::shared_ptr<session::KernelGraph> ObtainTransDataKernelGraph();
  void ConstructKernelGraphAndSetAttr();
  enum KernelType kernel_type_;
};
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LAUNCH_TRANSDATA_H_
