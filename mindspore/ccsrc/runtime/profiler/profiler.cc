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

#include "runtime/profiler/profiler.h"
#include <functional>
#include <iomanip>
#include "utils/file_utils.h"
#include "include/common/debug/common.h"

namespace mindspore {
namespace runtime {
static const int kPrecisionDigits = 2;

// The string of json file.
static const char kJsonName[] = "name";
static const char kJsonPh[] = "ph";
static const char kJsonPid[] = "pid";
static const char kJsonTid[] = "tid";
static const char kJsonTs[] = "ts";
static const char kJsonDur[] = "dur";
static const char kJsonPhX[] = "X";

// The env of runtime profiler.
static const char kEnableRuntimeProfiler[] = "MS_ENABLE_RUNTIME_PROFILER";
static const char kRuntimeProfilerTopNum[] = "MS_ENABLE_PROFILER_TOP_NUM";

// Save file name.
static const char kJsonFileName[] = "RuntimeProfilerJson";
static const char kSummaryInfoFileName[] = "RuntimeProfilerSummary";
static const char kDetailInfoFileName[] = "RuntimeProfilerDetail";

static const std::map<ProfilerModule, std::string> kProfilerModuleString = {
  {ProfilerModule::kPython, "Python"},
  {ProfilerModule::kRuntime, "RuntimeFramework"},
  {ProfilerModule::kPynative, "PynativeFramework"},
  {ProfilerModule::kKernel, "Kernel"},
  {ProfilerModule::kOther, "Other"},
};

static const std::map<ProfilerEvent, std::string> kProfilerEventString = {
  {ProfilerEvent::kDefault, "Default"},
  {ProfilerEvent::kKernelInfer, "KernelInfer"},
  {ProfilerEvent::kKernelResize, "KernelResize"},
  {ProfilerEvent::kKernelLaunch, "KernelLaunch"},
  {ProfilerEvent::kKernelUpdate, "KernelUpdate"},
  {ProfilerEvent::kGraphLaunch, "GraphLaunch"},
  {ProfilerEvent::kInputProcess, "InputProcess"},
  {ProfilerEvent::kOutputProcess, "OutputProcess"},
  {ProfilerEvent::kWaitTaskFinish, "WaitTaskFinish"},
  {ProfilerEvent::kPreLaunch, "PreLaunch"},
  {ProfilerEvent::kPostLaunch, "PostLaunch"},
  {ProfilerEvent::kSendOutput, "SendOutput"},
  {ProfilerEvent::kMemoryAlloc, "MemoryAlloc"},
  {ProfilerEvent::kMemoryFree, "MemoryFree"},
  {ProfilerEvent::kCopyData, "CopyData"},
  {ProfilerEvent::kStreamSync, "StreamSync"},
  // Inner event.
  {ProfilerEvent::kKernelInferInner, "KernelInferInner"},
  {ProfilerEvent::kKernelInferDataSync, "KernelInferDataSync"},
};

namespace {
std::string GetTidString(const std::thread::id &tid) {
  std::stringstream ss;
  ss << tid;
  return ss.str();
}

std::string GetRealPathName(const std::string &name) {
  auto path_name = GetSaveGraphsPathName(name);
  auto real_path = mindspore::Common::CreatePrefixPath(path_name);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path: " << path_name;
    return ("./" + name);
  }
  return real_path.value();
}
}  // namespace

ProfilerRecorder::ProfilerRecorder(ProfilerModule module, ProfilerEvent event, const std::string &op_name,
                                   bool is_inner_event) {
  if (!ProfilerAnalyzer::GetInstance().profiler_enable()) {
    return;
  }
  module_ = module;
  event_ = event;
  op_name_ = ProfilerAnalyzer::GetInstance().GetBriefName(op_name);
  start_time_ = ProfilerAnalyzer::GetInstance().GetTimeStamp();
  is_inner_event_ = is_inner_event;
}

ProfilerRecorder::~ProfilerRecorder() {
  if (!ProfilerAnalyzer::GetInstance().profiler_enable()) {
    return;
  }
  ProfilerAnalyzer::GetInstance().RecordData(std::make_shared<ProfilerData>(
    module_, event_, op_name_, is_inner_event_, start_time_, ProfilerAnalyzer::GetInstance().GetTimeStamp()));
}

ProfilerAnalyzer &ProfilerAnalyzer::GetInstance() noexcept {
  static ProfilerAnalyzer instance{};
  return instance;
}

void ProfilerAnalyzer::Initialize() {
  if (init_) {
    return;
  }
  init_ = true;

  if (common::GetEnv(kEnableRuntimeProfiler) != "1") {
    return;
  }
  profiler_enable_ = true;
  auto top_num_env = common::GetEnv(kRuntimeProfilerTopNum);
  if (top_num_env != std::string()) {
    show_top_num_ = stoi(top_num_env);
  }

  auto now_time = std::to_string(GetTimeStamp());
  json_file_name_ = GetRealPathName(kJsonFileName + now_time + ".json");
  summary_info_file_name_ = GetRealPathName(kSummaryInfoFileName + now_time + ".csv");
  detail_info_file_name_ = GetRealPathName(kDetailInfoFileName + now_time + ".csv");
}

void ProfilerAnalyzer::Clear() {
  if (!profiler_enable_) {
    return;
  }

  // Dump json data.
  DumpJsonData();

  // Reset the saved data.
  json_infos_.clear();
  data_.clear();
  module_infos_.clear();
}

uint64_t ProfilerAnalyzer::GetTimeStamp() {
  auto now_time = std::chrono::steady_clock::now();
  int64_t us_time_stamp = std::chrono::duration_cast<std::chrono::microseconds>(now_time.time_since_epoch()).count();
  return static_cast<uint64_t>(us_time_stamp);
}

// For example: ScopeName(XX/XX/ReLU-op1) --> BriefName(ReLU)
std::string ProfilerAnalyzer::GetBriefName(const std::string &scope_name) {
  auto first_index = scope_name.rfind('/');
  auto second_index = scope_name.rfind("-op");
  if ((first_index != std::string::npos) && (second_index != std::string::npos) &&
      (first_index + 1 < scope_name.size()) && (first_index + 1 < second_index)) {
    return scope_name.substr(first_index + 1, second_index - first_index - 1);
  }
  return scope_name;
}

void ProfilerAnalyzer::RecordData(const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  std::unique_lock<std::mutex> lock(data_mutex_);
  (void)data_.emplace_back(data);
}

void ProfilerAnalyzer::StartStep() {
  Initialize();
  if (!profiler_enable_) {
    return;
  }
  ++step_;

  std::unique_lock<std::mutex> lock(data_mutex_);
  // Reset the saved data.
  step_total_time_ = 0;
  data_.clear();
  module_infos_.clear();
}

void ProfilerAnalyzer::EndStep() {
  if (!profiler_enable_) {
    return;
  }
  std::unique_lock<std::mutex> lock(data_mutex_);
  if (data_.empty()) {
    return;
  }

  // Process data.
  for (auto &data : data_) {
    MS_EXCEPTION_IF_NULL(data);
    SaveJsonData(data);
    AnalyzeSummaryData(data);
  }

  // Dump data.
  DumpDetailData();
  DumpSummaryData();

  // Reset the saved data.
  step_total_time_ = 0;
  data_.clear();
  module_infos_.clear();
}

void ProfilerAnalyzer::SaveJsonData(const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  nlohmann::json json_data;
  json_data[kJsonName] =
    kProfilerModuleString.at(data->module_) + "::" + kProfilerEventString.at(data->event_) + "::" + data->op_name_;
  json_data[kJsonPh] = kJsonPhX;
  json_data[kJsonPid] = std::to_string(data->pid_);
  json_data[kJsonTid] = GetTidString(data->tid_);
  json_data[kJsonTs] = data->start_time_;
  json_data[kJsonDur] = data->dur_time_;

  json_infos_.emplace_back(json_data);
}

void ProfilerAnalyzer::AnalyzeSummaryData(const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(data);

  if (module_infos_.count(data->module_) > 0) {
    auto &module_info = module_infos_[data->module_];
    MS_EXCEPTION_IF_NULL(module_info);
    MS_EXCEPTION_IF_NULL(module_info->module_statistics_info_);
    if (!data->is_inner_event_) {
      module_info->module_statistics_info_->AccumulateTime(data->dur_time_);
      step_total_time_ += data->dur_time_;
    }
    return AnalyzeEventSummaryData(&module_info->event_infos_, data);
  }

  auto module_info = std::make_shared<ProfilerModuleInfo>();
  module_info->module_statistics_info_ =
    std::make_shared<ProfilerStatisticsInfo>(kProfilerModuleString.at(data->module_));
  if (!data->is_inner_event_) {
    module_info->module_statistics_info_->AccumulateTime(data->dur_time_);
    step_total_time_ += data->dur_time_;
  }
  (void)module_infos_.emplace(data->module_, module_info);
  AnalyzeEventSummaryData(&module_info->event_infos_, data);
}

void ProfilerAnalyzer::AnalyzeEventSummaryData(std::map<ProfilerEvent, ProfilerEventInfoPtr> *const event_infos,
                                               const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(event_infos);
  MS_EXCEPTION_IF_NULL(data);
  if (event_infos->count(data->event_) > 0) {
    auto &event_info = (*event_infos)[data->event_];
    MS_EXCEPTION_IF_NULL(event_info);
    MS_EXCEPTION_IF_NULL(event_info->event_statistics_info_);
    event_info->event_statistics_info_->AccumulateTime(data->dur_time_);
    return AnalyzeOpSummaryData(&event_info->op_infos_, data);
  }

  auto event_info = std::make_shared<ProfilerEventInfo>();
  event_info->event_statistics_info_ =
    std::make_shared<ProfilerStatisticsInfo>(kProfilerEventString.at(data->event_), data->is_inner_event_);
  event_info->event_statistics_info_->AccumulateTime(data->dur_time_);
  (void)event_infos->emplace(data->event_, event_info);
  AnalyzeOpSummaryData(&event_info->op_infos_, data);
}

void ProfilerAnalyzer::AnalyzeOpSummaryData(mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> *const op_infos,
                                            const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(op_infos);
  MS_EXCEPTION_IF_NULL(data);
  if (op_infos->count(data->op_name_) > 0) {
    auto &op_info = (*op_infos)[data->op_name_];
    MS_EXCEPTION_IF_NULL(op_info);
    return op_info->AccumulateTime(data->dur_time_);
  }

  auto op_info = std::make_shared<ProfilerStatisticsInfo>(data->op_name_, data->is_inner_event_);
  op_info->AccumulateTime(data->dur_time_);
  (void)op_infos->emplace(data->op_name_, op_info);
}

void ProfilerAnalyzer::DumpJsonData() {
  ChangeFileMode(json_file_name_, S_IWUSR);
  std::ofstream ofs(json_file_name_, std::ofstream::app);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << json_file_name_ << "] failed!";
    return;
  }
  ofs << json_infos_.dump();
  ChangeFileMode(json_file_name_, S_IRUSR);
}

void ProfilerAnalyzer::DumpDetailData() {
  ChangeFileMode(detail_info_file_name_, S_IWUSR);
  std::ofstream ofs(detail_info_file_name_, std::ofstream::app);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << detail_info_file_name_ << "] failed!";
    return;
  }

  ofs << "[Step:" << step_ << " time:" << step_total_time_ << "us]\n";
  for (auto &data : data_) {
    MS_EXCEPTION_IF_NULL(data);
    ofs << "module:" << kProfilerModuleString.at(data->module_) << ", event:" << kProfilerEventString.at(data->event_)
        << ", op:" << data->op_name_ << ", start_time:" << data->start_time_ << ", end_time:" << data->end_time_
        << ", dur_time:," << data->dur_time_ << ",us, tid:" << GetTidString(data->tid_) << ", pid:" << data->pid_
        << "\n";
  }
  ofs << "\n";

  ChangeFileMode(detail_info_file_name_, S_IRUSR);
}

void ProfilerAnalyzer::DumpSummaryData() {
  // Fill the summary info.
  std::stringstream string_stream;
  string_stream << "[Step:" << step_ << " time:" << step_total_time_ << "us]\n";
  DumpModuleSummaryData(string_stream);
  std::cout << string_stream.str() << std::endl;

  ChangeFileMode(summary_info_file_name_, S_IWUSR);
  std::ofstream ofs(summary_info_file_name_, std::ofstream::app);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << summary_info_file_name_ << "] failed!";
    return;
  }
  ofs << string_stream.str();
  ChangeFileMode(summary_info_file_name_, S_IRUSR);
}

void ProfilerAnalyzer::DumpModuleSummaryData(std::stringstream &string_stream) {
  // Order module info by total time.
  std::multimap<uint64_t, ProfilerModuleInfo *, std::greater_equal<uint64_t>> order_module_infos;
  for (auto &module_info : module_infos_) {
    MS_EXCEPTION_IF_NULL(module_info.second);
    auto &module_statistics_info = module_info.second->module_statistics_info_;
    MS_EXCEPTION_IF_NULL(module_statistics_info);
    module_statistics_info->Average();
    module_statistics_info->CalculatePercent(step_total_time_);
    (void)order_module_infos.emplace(module_statistics_info->total_time_, module_info.second.get());
  }

  for (auto &order_module_info : order_module_infos) {
    auto &module_statistics_info = order_module_info.second->module_statistics_info_;
    string_stream << "Module:" << module_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", total_time:" << module_statistics_info->total_time_
                  << "us, average_time:" << module_statistics_info->average_time_
                  << "us, total_count:" << module_statistics_info->count_
                  << ", percent:" << module_statistics_info->percent_ << "%\n";
    DumpEventSummaryData(order_module_info.second->event_infos_, string_stream);
  }

  string_stream << "\n";
}

void ProfilerAnalyzer::DumpEventSummaryData(const std::map<ProfilerEvent, ProfilerEventInfoPtr> &event_infos,
                                            std::stringstream &string_stream) {
  // Order event info by total time.
  std::multimap<uint64_t, ProfilerEventInfo *, std::greater_equal<uint64_t>> order_event_infos;
  std::multimap<uint64_t, ProfilerEventInfo *, std::greater_equal<uint64_t>> order_inner_event_infos;
  for (auto &event_info : event_infos) {
    MS_EXCEPTION_IF_NULL(event_info.second);
    auto &event_statistics_info = event_info.second->event_statistics_info_;
    MS_EXCEPTION_IF_NULL(event_statistics_info);
    event_statistics_info->Average();
    event_statistics_info->CalculatePercent(step_total_time_);
    if (event_statistics_info->is_inner_info_) {
      (void)order_inner_event_infos.emplace(event_statistics_info->total_time_, event_info.second.get());
    } else {
      (void)order_event_infos.emplace(event_statistics_info->total_time_, event_info.second.get());
    }
  }

  for (auto &order_event_info : order_event_infos) {
    auto &event_statistics_info = order_event_info.second->event_statistics_info_;
    string_stream << "  Event:" << event_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", total_time:" << event_statistics_info->total_time_
                  << "us, average_time:" << event_statistics_info->average_time_
                  << "us, total_count:" << event_statistics_info->count_
                  << ", percent:" << event_statistics_info->percent_ << "%\n";
    DumpOpSummaryData(order_event_info.second->op_infos_, string_stream);
  }

  // Inner event.
  for (auto &order_inner_event_info : order_inner_event_infos) {
    auto &event_statistics_info = order_inner_event_info.second->event_statistics_info_;
    string_stream << "  EventInner:" << event_statistics_info->name_ << std::fixed
                  << std::setprecision(kPrecisionDigits) << ", total_time:" << event_statistics_info->total_time_
                  << "us, average_time:" << event_statistics_info->average_time_
                  << "us, total_count:" << event_statistics_info->count_ << "\n";
    DumpOpSummaryData(order_inner_event_info.second->op_infos_, string_stream);
  }

  string_stream << "\n";
}

void ProfilerAnalyzer::DumpOpSummaryData(const mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> &op_infos,
                                         std::stringstream &string_stream) {
  if (show_top_num_ == 0) {
    return;
  }

  // Order op info by total time and average time.
  std::multimap<uint64_t, ProfilerStatisticsInfo *, std::greater_equal<uint64_t>> total_time_order_op_infos;
  std::multimap<double, ProfilerStatisticsInfo *, std::greater_equal<double>> average_time_order_op_infos;
  for (auto &op_info : op_infos) {
    auto &op_statistics_info = op_info.second;
    MS_EXCEPTION_IF_NULL(op_statistics_info);
    op_statistics_info->Average();
    op_statistics_info->CalculatePercent(step_total_time_);
    (void)total_time_order_op_infos.emplace(op_statistics_info->total_time_, op_statistics_info.get());
    (void)average_time_order_op_infos.emplace(op_statistics_info->average_time_, op_statistics_info.get());
  }

  // Show the op info by the total time top num.
  string_stream << "    Total time top " << show_top_num_ << " op:\n";
  int show_num = 0;
  for (auto &order_op_info : total_time_order_op_infos) {
    if (++show_num > show_top_num_) {
      break;
    }
    auto &op_statistics_info = order_op_info.second;
    string_stream << "      Op:" << op_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", total_time:" << op_statistics_info->total_time_
                  << "us, average_time:" << op_statistics_info->average_time_
                  << "us, total_count:" << op_statistics_info->count_;
    if (op_statistics_info->is_inner_info_) {
      string_stream << "\n";
    } else {
      string_stream << ", percent:" << op_statistics_info->percent_ << "%\n";
    }
  }

  // Show the op info by the average time top num.
  string_stream << "    Average time top " << show_top_num_ << " op:\n";
  show_num = 0;
  for (auto &order_op_info : average_time_order_op_infos) {
    if (++show_num > show_top_num_) {
      break;
    }
    auto &op_statistics_info = order_op_info.second;
    string_stream << "      Op:" << op_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", average_time:" << op_statistics_info->average_time_
                  << "us, total_time:" << op_statistics_info->total_time_
                  << "us, total_count:" << op_statistics_info->count_;
    if (op_statistics_info->is_inner_info_) {
      string_stream << "\n";
    } else {
      string_stream << ", percent:" << op_statistics_info->percent_ << "%\n";
    }
  }

  string_stream << "\n";
}
}  // namespace runtime
}  // namespace mindspore
