/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_SIGNAL_UTIL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_SIGNAL_UTIL_H_

#include <csignal>

namespace mindspore {
class SignalGuard {
 public:
  SignalGuard();
  ~SignalGuard();

 private:
  void RegisterHandlers();
  static void IntHandler(int sig_num, siginfo_t *sig_info, void *context);

  void (*old_handler)(int, siginfo_t *, void *) = nullptr;
  struct sigaction int_action;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_SIGNAL_UTIL_H_
