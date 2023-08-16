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
#include "include/api/model.h"
#include "include/api/model_group.h"
#include "include/api/model_parallel_runner.h"
#include "src/common/log_adapter.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

namespace mindspore::lite {
namespace py = pybind11;
using MSTensorPtr = std::shared_ptr<MSTensor>;

std::vector<MSTensorPtr> MSTensorToMSTensorPtr(const std::vector<MSTensor> &tensors) {
  std::vector<MSTensorPtr> tensors_ptr;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(tensors_ptr),
                 [](auto &item) { return std::make_shared<MSTensor>(item); });
  return tensors_ptr;
}

std::vector<MSTensor> MSTensorPtrToMSTensor(const std::vector<MSTensorPtr> &tensors_ptr) {
  std::vector<MSTensor> tensors;
  for (auto &item : tensors_ptr) {
    if (item == nullptr) {
      MS_LOG(ERROR) << "Tensor object cannot be nullptr";
      return {};
    }
    tensors.push_back(*item);
  }
  return tensors;
}

std::vector<MSTensorPtr> PyModelPredict(Model *model, const std::vector<MSTensorPtr> &inputs_ptr,
                                        const std::vector<MSTensorPtr> &outputs_ptr) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  std::vector<MSTensor> outputs;
  if (!outputs_ptr.empty()) {
    outputs = MSTensorPtrToMSTensor(outputs_ptr);
  }
  if (!model->Predict(inputs, &outputs).IsOk()) {
    return {};
  }
  return MSTensorToMSTensorPtr(outputs);
}

Status PyModelResize(Model *model, const std::vector<MSTensorPtr> &inputs_ptr,
                     const std::vector<std::vector<int64_t>> &new_shapes) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return kLiteError;
  }
  auto inputs = MSTensorPtrToMSTensor(inputs_ptr);
  return model->Resize(inputs, new_shapes);
}

Status PyModelUpdateConfig(Model *model, const std::string &key, const std::map<std::string, std::string> &value) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return kLiteError;
  }
  for (auto &item : value) {
    if (model->UpdateConfig(key, item).IsError()) {
      MS_LOG(ERROR) << "Update config failed, section: " << key << ", config name: " << item.first
                    << ", config value: " << item.second;
      return kLiteError;
    }
  }
  return kSuccess;
}

std::vector<MSTensorPtr> PyModelGetInputs(Model *model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(model->GetInputs());
}

std::vector<MSTensorPtr> PyModelGetOutputs(Model *model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(model->GetOutputs());
}

void ModelPyBind(const py::module &m) {
  (void)py::enum_<ModelType>(m, "ModelType")
    .value("kMindIR", ModelType::kMindIR)
    .value("kMindIR_Lite", ModelType::kMindIR_Lite);

  (void)py::enum_<StatusCode>(m, "StatusCode")
    .value("kSuccess", StatusCode::kSuccess)
    .value("kLiteError", StatusCode::kLiteError)
    .value("kLiteNullptr", StatusCode::kLiteNullptr)
    .value("kLiteParamInvalid", StatusCode::kLiteParamInvalid)
    .value("kLiteNoChange", StatusCode::kLiteNoChange)
    .value("kLiteSuccessExit", StatusCode::kLiteSuccessExit)
    .value("kLiteMemoryFailed", StatusCode::kLiteMemoryFailed)
    .value("kLiteNotSupport", StatusCode::kLiteNotSupport)
    .value("kLiteThreadPoolError", StatusCode::kLiteThreadPoolError)
    .value("kLiteUninitializedObj", StatusCode::kLiteUninitializedObj)
    .value("kLiteFileError", StatusCode::kLiteFileError)
    .value("kLiteServiceDeny", StatusCode::kLiteServiceDeny)
    .value("kLiteOutOfTensorRange", StatusCode::kLiteOutOfTensorRange)
    .value("kLiteInputTensorError", StatusCode::kLiteInputTensorError)
    .value("kLiteReentrantError", StatusCode::kLiteReentrantError)
    .value("kLiteGraphFileError", StatusCode::kLiteGraphFileError)
    .value("kLiteNotFindOp", StatusCode::kLiteNotFindOp)
    .value("kLiteInvalidOpName", StatusCode::kLiteInvalidOpName)
    .value("kLiteInvalidOpAttr", StatusCode::kLiteInvalidOpAttr)
    .value("kLiteOpExecuteFailure", StatusCode::kLiteOpExecuteFailure)
    .value("kLiteFormatError", StatusCode::kLiteFormatError)
    .value("kLiteInferError", StatusCode::kLiteInferError)
    .value("kLiteInferInvalid", StatusCode::kLiteInferInvalid)
    .value("kLiteInputParamInvalid", StatusCode::kLiteInputParamInvalid);

  (void)py::class_<Status, std::shared_ptr<Status>>(m, "Status")
    .def(py::init<>())
    .def("ToString", &Status::ToString)
    .def("IsOk", &Status::IsOk)
    .def("IsError", &Status::IsError);

  (void)py::class_<Model, std::shared_ptr<Model>>(m, "ModelBind")
    .def(py::init<>())
    .def("build_from_buff",
         py::overload_cast<const void *, size_t, ModelType, const std::shared_ptr<Context> &>(&Model::Build),
         py::call_guard<py::gil_scoped_release>())
    .def("build_from_file",
         py::overload_cast<const std::string &, ModelType, const std::shared_ptr<Context> &>(&Model::Build),
         py::call_guard<py::gil_scoped_release>())
    .def("build_from_buff_with_decrypt",
         py::overload_cast<const void *, size_t, ModelType, const std::shared_ptr<Context> &, const Key &,
                           const std::string &, const std::string &>(&Model::Build))
    .def("build_from_file_with_decrypt",
         py::overload_cast<const std::string &, ModelType, const std::shared_ptr<Context> &, const Key &,
                           const std::string &, const std::string &>(&Model::Build))
    .def("load_config", py::overload_cast<const std::string &>(&Model::LoadConfig))
    .def("update_config", &PyModelUpdateConfig)
    .def("resize", &PyModelResize)
    .def("predict", &PyModelPredict, py::call_guard<py::gil_scoped_release>())
    .def("get_inputs", &PyModelGetInputs)
    .def("get_outputs", &PyModelGetOutputs)
    .def("get_input_by_tensor_name",
         [](Model &model, const std::string &tensor_name) { return model.GetInputByTensorName(tensor_name); })
    .def("get_output_by_tensor_name",
         [](Model &model, const std::string &tensor_name) { return model.GetOutputByTensorName(tensor_name); });
}

#ifdef PARALLEL_INFERENCE
std::vector<MSTensorPtr> PyModelParallelRunnerPredict(ModelParallelRunner *model,
                                                      const std::vector<MSTensorPtr> &inputs_ptr,
                                                      const MSKernelCallBack &before = nullptr,
                                                      const MSKernelCallBack &after = nullptr) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  std::vector<MSTensor> inputs = MSTensorPtrToMSTensor(inputs_ptr);
  std::vector<MSTensor> outputs;
  if (!model->Predict(inputs, &outputs, before, after).IsOk()) {
    return {};
  }
  return MSTensorToMSTensorPtr(outputs);
}
std::vector<MSTensorPtr> PyModelParallelRunnerGetInputs(ModelParallelRunner *model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(model->GetInputs());
}

std::vector<MSTensorPtr> PyModelParallelRunnerGetOutputs(ModelParallelRunner *model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "Model object cannot be nullptr";
    return {};
  }
  return MSTensorToMSTensorPtr(model->GetOutputs());
}
#endif

void ModelParallelRunnerPyBind(const py::module &m) {
#ifdef PARALLEL_INFERENCE
  (void)py::class_<RunnerConfig, std::shared_ptr<RunnerConfig>>(m, "RunnerConfigBind")
    .def(py::init<>())
    .def("set_config_info", py::overload_cast<const std::string &, const std::map<std::string, std::string> &>(
                              &RunnerConfig::SetConfigInfo))
    .def("get_config_info", &RunnerConfig::GetConfigInfo)
    .def("set_config_path", py::overload_cast<const std::string &>(&RunnerConfig::SetConfigPath))
    .def("get_config_path", &RunnerConfig::GetConfigPath)
    .def("set_workers_num", &RunnerConfig::SetWorkersNum)
    .def("get_workers_num", &RunnerConfig::GetWorkersNum)
    .def("set_context", &RunnerConfig::SetContext)
    .def("get_context", &RunnerConfig::GetContext)
    .def("set_device_ids", &RunnerConfig::SetDeviceIds)
    .def("get_device_ids", &RunnerConfig::GetDeviceIds)
    .def("get_context_info",
         [](RunnerConfig &runner_config) {
           const auto &context = runner_config.GetContext();
           std::string result = "thread num: " + std::to_string(context->GetThreadNum()) +
                                ", bind mode: " + std::to_string(context->GetThreadAffinityMode());
           return result;
         })
    .def("get_config_info_string", [](RunnerConfig &runner_config) {
      std::string result = "";
      const auto &config_info = runner_config.GetConfigInfo();
      for (auto &section : config_info) {
        result += section.first + ": ";
        for (auto &config : section.second) {
          auto temp = config.first + " " + config.second + "\n";
          result += temp;
        }
      }
      return result;
    });

  (void)py::class_<ModelParallelRunner, std::shared_ptr<ModelParallelRunner>>(m, "ModelParallelRunnerBind")
    .def(py::init<>())
    .def("init",
         py::overload_cast<const std::string &, const std::shared_ptr<RunnerConfig> &>(&ModelParallelRunner::Init),
         py::call_guard<py::gil_scoped_release>())
    .def("get_inputs", &PyModelParallelRunnerGetInputs)
    .def("get_outputs", &PyModelParallelRunnerGetOutputs)
    .def("predict", &PyModelParallelRunnerPredict, py::call_guard<py::gil_scoped_release>());
#endif
}

Status PyModelGroupAddModelByObject(ModelGroup *model_group, const std::vector<Model *> &models_ptr) {
  if (model_group == nullptr) {
    MS_LOG(ERROR) << "Model group object cannot be nullptr";
    return {};
  }
  std::vector<Model> models;
  for (auto model_ptr : models_ptr) {
    if (model_ptr == nullptr) {
      MS_LOG(ERROR) << "Model object cannot be nullptr";
      return {};
    }
    models.push_back(*model_ptr);
  }
  return model_group->AddModel(models);
}

void ModelGroupPyBind(const py::module &m) {
  (void)py::enum_<ModelGroupFlag>(m, "ModelGroupFlag")
    .value("kShareWeight", ModelGroupFlag::kShareWeight)
    .value("kShareWorkspace", ModelGroupFlag::kShareWorkspace);

  (void)py::class_<ModelGroup, std::shared_ptr<ModelGroup>>(m, "ModelGroupBind")
    .def(py::init<ModelGroupFlag>())
    .def("add_model", py::overload_cast<const std::vector<std::string> &>(&ModelGroup::AddModel))
    .def("add_model_by_object", &PyModelGroupAddModelByObject)
    .def("cal_max_size_of_workspace",
         py::overload_cast<ModelType, const std::shared_ptr<Context> &>(&ModelGroup::CalMaxSizeOfWorkspace));
}
}  // namespace mindspore::lite
