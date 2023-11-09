//
// Created by jojo on 2023/10/31.
//

#include "plugin/device/ascend/kernel/pyboost/customize/bias_add.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "transform/acl_ir/acl_helper.h"
#include "kernel/pyboost/py_boost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
bool BiasAddAscendLaunch(const PrimitivePtr &primitive, const tensor::TensorPtr &input_x, const tensor::TensorPtr &bias,
                         const tensor::TensorPtr &output) {
  GilReleaseWithCheck release_gil;

  auto device_context = PyBoostUtils::GetDeviceContext(kAscendDevice);

  auto input_x_address =
    runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, input_x, "input_x", true);
  auto bias_address = runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, bias, "bias", true);
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, output, "output");
  auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(output->device_address());
  MS_EXCEPTION_IF_NULL(input_x_address);
  MS_EXCEPTION_IF_NULL(bias_address);
  MS_EXCEPTION_IF_NULL(output_address);

  std::vector<std::string> input_device_formats = {input_x_address->format(), bias_address->format()};
  auto output_device_formats = {output_address->format()};
  auto input_device_types = {input_x_address->type_id(), bias_address->type_id()};
  auto output_device_types = {output_address->type_id()};

  kernel::AclKernelModPtr bias_add_kernel = std::make_shared<kernel::AclKernelMod>();
  bias_add_kernel->SetPrimitive(primitive);
  bias_add_kernel->CreateAclConverter();
  bias_add_kernel->SetDeviceInfo(input_device_formats, output_device_formats, input_device_types, output_device_types);
  ShapeVector input_x_shape = input_x->shape();
  ShapeVector bias_shape = bias->shape();
  const std::string &format = transform::AclHelper::GetFormatFromAttr(primitive);
  bias_add_kernel->PackageInput(kIndex0, format, &input_x_shape);
  bias_add_kernel->PackageInput(kIndex1, format, &bias_shape);
  bias_add_kernel->PackageOutput(kIndex0, output->shape());

  auto x_addr = std::make_shared<kernel::Address>(input_x_address->GetMutablePtr(), input_x_address->GetSize());
  auto b_addr = std::make_shared<kernel::Address>(bias_address->GetMutablePtr(), bias_address->GetSize());
  auto out_addr = std::make_shared<kernel::Address>(output_address->GetMutablePtr(), output_address->GetSize());

  auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  MS_LOG(DEBUG) << "Begin launch kernel: BasiAdd";
  auto ret = bias_add_kernel->Launch({x_addr, b_addr}, std::vector<AddressPtr>{}, {out_addr}, stream_ptr);
  MS_LOG(DEBUG) << "End launch kernel: BasiAdd";
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: BasiAdd";
  }
  MS_LOG(DEBUG) << "End";

  return true;
}

tensor::TensorPtr BiasAddAscendCall(const PrimitivePtr &primitive, const tensor::TensorPtr &input_x,
                                    const tensor::TensorPtr &bias, const tensor::TensorPtr &output) {
  BiasAddAscendLaunch(primitive, input_x, bias, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
