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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_ASYNC_QUEUE_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_ASYNC_QUEUE_H_

#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <condition_variable>
#include <utility>

#include "include/backend/visible.h"
#include "runtime/pynative/async/task.h"

namespace mindspore {
namespace pynative {
enum kThreadWaitLevel : int {
  kLevelUnknown = 0,
  kLevelPython,
  kLevelGrad,
  kLevelFrontend,
  kLevelBackend,
  kLevelDevice,
};
// Create a new thread to execute the tasks in the queue sequentially.
class BACKEND_EXPORT AsyncQueue {
 public:
  explicit AsyncQueue(std::string name, kThreadWaitLevel wait_level)
      : name_(std::move(name)), wait_level_(wait_level) {}
  virtual ~AsyncQueue();

  // Add task to the end of the queue.
  void Push(const std::shared_ptr<AsyncTask> &task);

  // Wait for all async task finish executing.
  void Wait();

  // Check if the queue is empty.
  bool Empty();

  // clear tasks of queue, and wait last task.
  void Clear();

  // When an exception occurs, the state needs to be reset.
  void Reset();

  // Thread join before the process exit.
  void WorkerJoin();

  // Reinit resources after fork occurs.
  void ChildAfterFork();

  bool TaskInQueue(uint32_t task_id);

 protected:
  void WorkerLoop();
  void SetThreadName() const;

  std::unique_ptr<std::thread> worker_{nullptr};
  std::mutex task_mutex_;
  std::unique_ptr<std::condition_variable> task_cond_var_{nullptr};
  std::string name_;
  kThreadWaitLevel wait_level_;
  inline static std::unordered_map<std::thread::id, kThreadWaitLevel> thread_id_to_wait_level_;
  inline static std::mutex level_mutex_;

 private:
  void ClearTaskWithException();

  std::queue<std::shared_ptr<AsyncTask>> tasks_queque_;
  std::set<uint32_t> task_in_queue_;
};
using AsyncQueuePtr = std::shared_ptr<AsyncQueue>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_ASYNC_QUEUE_H_
