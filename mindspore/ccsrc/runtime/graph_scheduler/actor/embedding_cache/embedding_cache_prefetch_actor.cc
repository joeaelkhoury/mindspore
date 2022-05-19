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

#include "runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace runtime {
using kernel::Address;
using kernel::AddressPtrList;
using mindspore::session::KernelGraph;

// One and two dimensional shape placeholder.
const ShapeVector kOneDimensionalShape = {1};
const ShapeVector kTwoDimensionalShape = {1, 1};

// Maximum number of threads for concurrent accelerated cache processing.
constexpr size_t kMaxThreadNum = 16;
// Maximum number of feature ids processed per thread.
constexpr size_t kMaxIdsPerThread = 10000;

namespace {
ParameterPtr NewParameter(const KernelGraphPtr &graph, TypePtr type, const ShapeVector &shape) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(type);

  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  auto abstract = std::make_shared<abstract::AbstractTensor>(type, shape);
  param->set_abstract(abstract);

  auto mutable_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(mutable_inputs);
  mutable_inputs->push_back(param);

  return param;
}

bool InferOpShape(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  opt::dynamic_shape::InferOp(kernel);
  auto args = kernel::GetArgsFromCNode(kernel);
  MS_EXCEPTION_IF_NULL(args);

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (!kernel_mod->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map)) {
    MS_LOG(ERROR) << "Kernel " << kernel->fullname_with_scope() << " resize failed.";
    return false;
  }
  return true;
}
}  // namespace

void EmbeddingCachePrefetchActor::Initialize() {
  MS_EXCEPTION_IF_NULL(device_context_);
  if (!device_context_->CreateStream(&stream_id_)) {
    MS_LOG(EXCEPTION) << "Create stream failed.";
  }
}

void EmbeddingCachePrefetchActor::Finalize() {
  embedding_cache_lookup_node_ = nullptr;
  embedding_cache_update_node_ = nullptr;
}

void EmbeddingCachePrefetchActor::BuildEmbeddingCacheLookupKernel() {
  auto graph = std::make_shared<KernelGraph>();

  // 1. Create parameter nodes which are inputs of embedding cache look up kernel(operator name: 'Gather').
  ParameterPtr input_param = NewParameter(graph, kFloat32, kTwoDimensionalShape);
  ParameterPtr input_indices = NewParameter(graph, kInt32, kOneDimensionalShape);

  // 2. Create a CNode for operator Gather.
  PrimitivePtr emb_lookup_primitive = std::make_shared<Primitive>(kGatherV2OpName);
  emb_lookup_primitive->set_attr(kAttrAxis, MakeValue<int64_t>(0));
  emb_lookup_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));
  emb_lookup_primitive->set_attr(kAttrOutputIsDynamicShape, MakeValue(true));
  emb_lookup_primitive->set_attr(kAttrStream, MakeValue(stream_id_));

  std::vector<AnfNodePtr> emb_lookup_input_nodes{NewValueNode(emb_lookup_primitive), input_param, input_indices};
  embedding_cache_lookup_node_ = graph->NewCNode(emb_lookup_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_lookup_node_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimensionalShape);
  embedding_cache_lookup_node_->set_abstract(abstract);

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->CreateKernel({embedding_cache_lookup_node_});
}

void EmbeddingCachePrefetchActor::BuildEmbeddingCacheUpdateKernel() {
  auto graph = std::make_shared<KernelGraph>();

  // 1. Create parameter nodes which are inputs of embedding cache update kernel(operator name: 'ScatterUpdate').
  ParameterPtr input_param = NewParameter(graph, kFloat32, kTwoDimensionalShape);
  ParameterPtr input_indices = NewParameter(graph, kInt32, kOneDimensionalShape);
  ParameterPtr update_values = NewParameter(graph, kFloat32, kTwoDimensionalShape);

  // 2. Create a CNode for operator ScatterUpdate.
  PrimitivePtr embedding_cache_update_primitive = std::make_shared<Primitive>(kScatterUpdateOpName);
  embedding_cache_update_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));
  embedding_cache_update_primitive->set_attr(kAttrStream, MakeValue(stream_id_));

  std::vector<AnfNodePtr> embedding_cache_update_input_nodes{NewValueNode(embedding_cache_update_primitive),
                                                             input_param, input_indices, update_values};
  embedding_cache_update_node_ = graph->NewCNode(embedding_cache_update_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_update_node_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimensionalShape);
  embedding_cache_update_node_->set_abstract(abstract);

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->CreateKernel({embedding_cache_update_node_});
}

bool EmbeddingCachePrefetchActor::LookupDeviceCache(void *indices, void *embedding_cache, size_t indices_num,
                                                    size_t cache_size, size_t embedding_size, void *outputs) {
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(embedding_cache_lookup_node_);

  // 1. Update parameter nodes' shape.
  auto input_param_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, 0);
  MS_ERROR_IF_NULL(input_param_node);
  const ShapeVector input_param_shape = {SizeToLong(cache_size), SizeToLong(embedding_size)};
  auto input_param_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_param_shape);
  input_param_node->set_abstract(input_param_abstract);

  auto input_indices_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, 1);
  MS_ERROR_IF_NULL(input_indices_node);
  const ShapeVector input_indices_shape = {SizeToLong(indices_num)};
  auto input_indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, input_indices_shape);
  input_indices_node->set_abstract(input_indices_abstract);

  // 2. Infer shape for embedding cache look up kernel(operator name: 'Gather') which is dynamic shape kernel.
  if (!InferOpShape(embedding_cache_lookup_node_)) {
    MS_LOG(ERROR) << "Infer operator shape failed, op name: " << embedding_cache_lookup_node_->fullname_with_scope();
    return false;
  }

  // 3. Do embedding cache look up on device.
  AddressPtrList kernel_inputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float)),
    std::make_shared<Address>(indices, indices_num * sizeof(int))};
  AddressPtrList kernel_outputs = {std::make_shared<Address>(outputs, indices_num * embedding_size * sizeof(float))};

  MS_ERROR_IF_NULL(device_context_);
  auto ret = device_context_->LaunchKernel(embedding_cache_lookup_node_, kernel_inputs, {}, kernel_outputs);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel: " << embedding_cache_lookup_node_->fullname_with_scope() << " failed.";
    return false;
  }
  return true;
}

bool EmbeddingCachePrefetchActor::UpdateDeviceCache(void *indices, void *update_value, size_t indices_num,
                                                    size_t cache_size, size_t embedding_size, void *embedding_cache) {
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(update_value);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(embedding_cache_update_node_);

  // 1. Update parameter nodes' shape.
  auto input_param_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, 0);
  MS_ERROR_IF_NULL(input_param_node);
  const ShapeVector input_param_shape = {SizeToLong(cache_size), SizeToLong(embedding_size)};
  auto input_param_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_param_shape);
  input_param_node->set_abstract(input_param_abstract);

  auto input_indices_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, 1);
  MS_ERROR_IF_NULL(input_indices_node);
  const ShapeVector input_indices_shape = {SizeToLong(indices_num)};
  auto input_indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, input_indices_shape);
  input_indices_node->set_abstract(input_indices_abstract);

  auto update_values_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, 2);
  MS_ERROR_IF_NULL(update_values_node);
  const ShapeVector update_values_shape = {SizeToLong(indices_num), SizeToLong(embedding_size)};
  auto update_values_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, update_values_shape);
  update_values_node->set_abstract(update_values_abstract);

  // 2. Infer shape for embedding cache update kernel(operator name: 'ScatterUpdate') which is dynamic shape kernel.
  if (!InferOpShape(embedding_cache_update_node_)) {
    MS_LOG(ERROR) << "Infer operator shape failed, op name: " << embedding_cache_update_node_->fullname_with_scope();
    return false;
  }

  // 3. Do update cache on device.
  AddressPtrList kernel_inputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float)),
    std::make_shared<Address>(indices, indices_num * sizeof(int)),
    std::make_shared<Address>(update_value, indices_num * embedding_size * sizeof(float))};
  AddressPtrList kernel_outputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float))};

  MS_ERROR_IF_NULL(device_context_);
  auto ret = device_context_->LaunchKernel(embedding_cache_update_node_, kernel_inputs, {}, kernel_outputs);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel: " << embedding_cache_update_node_->fullname_with_scope() << " failed.";
    return false;
  }
  return true;
}

void EmbeddingCachePrefetchActor::Run() {
  // Note:Need to wait data channel ready.

  MS_LOG(INFO) << "Begin prefetching cache.";
  while (running_) {
    if (!PrefetchCache()) {
      running_ = false;
    }
  }
  MS_LOG(INFO) << "End prefetching cache.";
}

bool EmbeddingCachePrefetchActor::PrefetchCache() {
  // 1. Acquire batch ids
  void *data = nullptr;
  RETURN_IF_FALSE_WITH_LOG(PsDataPrefetch::GetInstance().QueryData(channel_name_, &data), "Query input data failed.");

  if (data == nullptr) {
    MS_LOG(INFO) << "There is no input data of dataset channel name:" << channel_name_;
    std::unique_lock<std::mutex> locker(data_mutex_);
    const int64_t longest_time_to_wait = 100;
    (void)data_parser_.wait_for(locker, std::chrono::milliseconds(longest_time_to_wait));
    return true;
  }

  RETURN_IF_FALSE_WITH_LOG(IncreaseStep(), "Increase step failed.");
  auto data_size = PsDataPrefetch::GetInstance().data_size(channel_name_);
  if (data_size == 0) {
    MS_LOG(ERROR) << "The data size of batch ids can not be zero.";
    return false;
  }
  auto batch_ids = reinterpret_cast<int *>(data);
  auto batch_ids_num = data_size / sizeof(int);
  std::unique_ptr<int[]> hash_index = std::make_unique<int[]>(batch_ids_num);
  auto ret = memset_s(&statistics_info_, sizeof(statistics_info_), 0, sizeof(statistics_info_));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memset for cache statistics info failed, errno[" << ret << "]";
    return false;
  }

  // 2. Count cache miss ids.
  RETURN_IF_FALSE_WITH_LOG(CountCacheMissIds(batch_ids, batch_ids_num, hash_index.get()),
                           "Count cache miss ids failed.");

  if ((device_cache_need_wait_graph_ || host_cache_need_wait_graph_) && (!WaitGraphRun())) {
    MS_LOG(ERROR) << "Cache prefetching waits graph finish failed.";
    return false;
  }

  // 3. If the device cache does not reach 100% hit rate, the cache needs to be updated.
  RETURN_IF_FALSE_WITH_LOG(UpdateCache(), "Update local cache failed.");

  // 4. Replace the batch_ids by hash index for GetNext operator to get hash index as input.
  size_t dest_len = data_size;
  ret = memcpy_s(data, dest_len, hash_index.get(), data_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy hash index failed, errno[" << ret << "]";
    return false;
  }
  RETURN_IF_FALSE_WITH_LOG(PsDataPrefetch::GetInstance().FinalizeData(channel_name_), "Finalize data failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::IncreaseStep() {
  if (data_step_ >= UINT64_MAX) {
    MS_LOG(ERROR) << "The data step (" << data_step_ << ") will exceed the maximum value of uint64_t.";
    return false;
  }
  data_step_++;
  set_current_graph_step();
  if (graph_running_step_ > data_step_) {
    MS_LOG(ERROR) << "The graph running step (" << graph_running_step_ << ") exceed the data step (" << data_step_
                  << ").";
    return false;
  }
  return true;
}

bool EmbeddingCachePrefetchActor::CountCacheMissIds(const int *batch_ids, const size_t batch_ids_num, int *hash_index) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);

  statistics_info_.batch_id_count_ = batch_ids_num;
  std::unique_ptr<bool[]> in_device = std::make_unique<bool[]>(batch_ids_num);
  std::unique_ptr<bool[]> out_range = std::make_unique<bool[]>(batch_ids_num);
  auto ret = memset_s(in_device.get(), batch_ids_num * sizeof(bool), 0, batch_ids_num * sizeof(bool));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
    return false;
  }
  ret = memset_s(out_range.get(), batch_ids_num * sizeof(bool), 0, batch_ids_num * sizeof(bool));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
    return false;
  }

  // 1. Analyze the hit/miss info of the local host cache and device cache.
  RETURN_IF_FALSE_WITH_LOG(
    CheckCacheHitOrOutRange(batch_ids, batch_ids_num, hash_index, in_device.get(), out_range.get()),
    "Check cache hit or out range failed.");
  RETURN_IF_FALSE_WITH_LOG(ResetEmbeddingHashMap(), "Reset embedding hash map failed.");

  // 2.calculate the swapping and mapping(feature id to cache index) information of the missing feature id that needs to
  // be inserted into the cache.
  for (size_t i = 0; i < batch_ids_num; i++) {
    if (in_device[i] || out_range[i]) {
      continue;
    }
    bool need_swap_host_to_device = true;
    bool need_swap_device_to_host = true;
    int index = INVALID_INDEX_VALUE;
    RETURN_IF_FALSE_WITH_LOG(
      ParseDeviceData(batch_ids[i], &need_swap_device_to_host, &need_swap_host_to_device, &index),
      "Parse device cache data failed.");
    hash_index[i] = index + local_device_cache_bounds_.first;
    if (need_swap_host_to_device) {
      RETURN_IF_FALSE_WITH_LOG(ParseHostDataHostToDevice(batch_ids[i]),
                               "Parse local host cache data(swap local host cache to device) failed.");
    }
    if (need_swap_device_to_host) {
      RETURN_IF_FALSE_WITH_LOG(ParseHostDataDeviceToHost(),
                               "Parse local host cache data(swap device cache to local host) failed.");
    }
  }
  return true;
}

bool EmbeddingCachePrefetchActor::ParseDeviceData(size_t id, bool *need_swap_device_to_host,
                                                  bool *need_swap_host_to_device, int *hash_index) {
  MS_ERROR_IF_NULL(need_swap_device_to_host);
  MS_ERROR_IF_NULL(need_swap_host_to_device);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);

  int index = INVALID_INDEX_VALUE;
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(id);
  if (iter != hash_id_to_index.end()) {
    *need_swap_device_to_host = false;
    *need_swap_host_to_device = false;
    index = iter->second;
    if (device_hash_map->hash_step(index) != data_step_) {
      statistics_info_.hash_hit_count_++;
      device_hash_map->set_hash_step(index, data_step_);
    }
  } else {
    int *device_to_host_index = embedding_device_cache_->device_to_host_index.get();
    int *device_to_host_ids = embedding_device_cache_->device_to_host_ids.get();
    int *host_to_device_index = embedding_device_cache_->host_to_device_index.get();
    int *host_to_device_ids = embedding_device_cache_->host_to_device_ids.get();
    MS_ERROR_IF_NULL(host_to_device_index);
    MS_ERROR_IF_NULL(host_to_device_ids);
    auto tmp_device_to_host_size = statistics_info_.device_to_host_size_;
    while (true) {
      // Calculate the mapping of id to index.
      index = device_hash_map->ParseData(id, device_to_host_index, device_to_host_ids, data_step_, graph_running_step_,
                                         &(statistics_info_.device_to_host_size_), &device_cache_need_wait_graph_);
      if (index == INVALID_INDEX_VALUE) {
        if (!WaitGraphRun()) {
          return false;
        }
        continue;
      }
      host_to_device_index[statistics_info_.host_to_device_size_] = index;
      host_to_device_ids[statistics_info_.host_to_device_size_] = id;
      statistics_info_.host_to_device_size_++;
      *need_swap_device_to_host = statistics_info_.device_to_host_size_ > tmp_device_to_host_size;
      break;
    }
  }

  *hash_index = index;
  return true;
}

bool EmbeddingCachePrefetchActor::ParseHostDataHostToDevice(size_t id) {
  MS_ERROR_IF_NULL(embedding_host_cache_);
  int *host_to_device_index = embedding_host_cache_->host_to_device_index.get();
  MS_ERROR_IF_NULL(host_to_device_index);
  auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);

  const auto &hash_id_to_index = host_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(id);
  if (iter != hash_id_to_index.end()) {
    auto index = iter->second;
    if (host_hash_map->hash_step(index) != data_step_) {
      host_hash_map->set_hash_step(index, data_step_);
    }
    host_to_device_index[statistics_info_.host_to_device_size_ - 1] = index;
  } else {
    int *host_to_server_index = embedding_host_cache_->host_to_server_index.get();
    int *host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
    int *server_to_host_index = embedding_host_cache_->server_to_host_index.get();
    int *server_to_host_ids = embedding_host_cache_->server_to_host_ids.get();
    MS_ERROR_IF_NULL(server_to_host_index);
    MS_ERROR_IF_NULL(server_to_host_ids);
    while (true) {
      // Calculate the mapping of id to index.
      auto index =
        host_hash_map->ParseData(id, host_to_server_index, host_to_server_ids, data_step_, graph_running_step_,
                                 &statistics_info_.host_to_server_size_, &host_cache_need_wait_graph_);
      if (index == INVALID_INDEX_VALUE) {
        RETURN_IF_FALSE_WITH_LOG(WaitGraphRun(), "Wait graph run failed.");
        continue;
      }
      host_to_device_index[statistics_info_.host_to_device_size_ - 1] = index;
      server_to_host_index[statistics_info_.server_to_host_size_] = index;
      server_to_host_ids[statistics_info_.server_to_host_size_++] = id;
      break;
    }
  }

  return true;
}

bool EmbeddingCachePrefetchActor::ParseHostDataDeviceToHost() {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);
  int *device_to_host_ids = embedding_device_cache_->device_to_host_ids.get();
  int *device_to_host_index = embedding_host_cache_->device_to_host_index.get();
  MS_ERROR_IF_NULL(device_to_host_ids);
  MS_ERROR_IF_NULL(device_to_host_index);

  auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);
  int swap_device_to_host_id = device_to_host_ids[statistics_info_.device_to_host_size_ - 1];
  const auto &hash_id_to_index = host_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(swap_device_to_host_id);
  if (iter != hash_id_to_index.end()) {
    auto index = iter->second;
    if (host_hash_map->hash_step(index) != data_step_) {
      host_hash_map->set_hash_step(index, data_step_);
    }
    device_to_host_index[statistics_info_.device_to_host_size_ - 1] = index;
  } else {
    int *host_to_server_index = embedding_host_cache_->host_to_server_index.get();
    int *host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
    while (true) {
      // Calculate the mapping of id to index.
      auto index = host_hash_map->ParseData(swap_device_to_host_id, host_to_server_index, host_to_server_ids,
                                            data_step_, graph_running_step_, &statistics_info_.host_to_server_size_,
                                            &host_cache_need_wait_graph_);
      if (index == INVALID_INDEX_VALUE) {
        RETURN_IF_FALSE_WITH_LOG(WaitGraphRun(), "Wait graph run");
        continue;
      }
      device_to_host_index[statistics_info_.device_to_host_size_ - 1] = index;
      break;
    }
  }

  return true;
}

bool EmbeddingCachePrefetchActor::CheckCacheHitOrOutRangeFunc(const int *batch_ids, const size_t batch_ids_num,
                                                              int *hash_index, bool *in_device, bool *out_range,
                                                              size_t *hash_hit_count) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(out_range);
  MS_ERROR_IF_NULL(hash_hit_count);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();

  for (size_t i = 0; i < batch_ids_num; ++i) {
    if (batch_ids[i] < local_embedding_slice_bounds_.first) {
      hash_index[i] = batch_ids[i] - local_embedding_slice_bounds_.first + local_device_cache_bounds_.first;
      out_range[i] = true;
      continue;
    }
    if (batch_ids[i] >= local_embedding_slice_bounds_.second) {
      hash_index[i] = batch_ids[i] + local_device_cache_bounds_.second;
      out_range[i] = true;
      continue;
    }
    auto iter = hash_id_to_index.find(batch_ids[i]);
    if (iter != hash_id_to_index.end()) {
      hash_index[i] = iter->second + local_device_cache_bounds_.first;
      if (device_hash_map->hash_step(iter->second) != data_step_) {
        ++(*hash_hit_count);
        device_hash_map->set_hash_step(iter->second, data_step_);
      }
      in_device[i] = true;
    }
  }
  return true;
}

bool EmbeddingCachePrefetchActor::CheckCacheHitOrOutRange(const int *batch_ids, const size_t batch_ids_num,
                                                          int *hash_index, bool *in_device, bool *out_range) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(out_range);

  size_t thread_num = batch_ids_num / kMaxIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t hash_hit_count[kMaxThreadNum] = {0};
  size_t i = 0;
  size_t offset = 0;

  for (; i < thread_num; ++i) {
    if (offset >= batch_ids_num) {
      break;
    }
    size_t proc_len = batch_ids_num / thread_num + (i < (batch_ids_num % thread_num) ? 1 : 0);
    threads[i] = std::thread(&EmbeddingCachePrefetchActor::CheckCacheHitOrOutRangeFunc, this, batch_ids + offset,
                             proc_len, hash_index + offset, in_device + offset, out_range + offset, hash_hit_count + i);
    offset += proc_len;
  }
  if (offset != batch_ids_num) {
    MS_LOG(WARNING) << "Check id in device inadequate, total:" << batch_ids_num << " checked:" << offset;
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  for (size_t j = 0; j < i; j++) {
    statistics_info_.hash_hit_count_ += hash_hit_count[j];
  }
  return true;
}

bool EmbeddingCachePrefetchActor::ResetEmbeddingHashMap() {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  const auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  MS_ERROR_IF_NULL(embedding_host_cache_);
  const auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);
  device_hash_map->Reset();
  host_hash_map->Reset();
  device_cache_need_wait_graph_ = false;
  host_cache_need_wait_graph_ = false;
  return true;
}

bool EmbeddingCachePrefetchActor::WaitGraphRun() {
  MS_LOG(INFO) << "Hash table has no space to insert new data and retries within 2 minutes.";
  std::unique_lock<std::mutex> locker(data_mutex_);
  const int64_t longest_time_to_wait = 120;
  if (!data_parser_.wait_for(locker, std::chrono::seconds(longest_time_to_wait),
                             [this] { return graph_step_ > graph_running_step_; })) {
    MS_LOG(ERROR) << "Data parse timeout, suggest to enlarge the vocab cache size(graph step:" << graph_step_
                  << ", graph running step:" << graph_running_step_ << ").";
    return false;
  }
  set_current_graph_step();
  return true;
}

bool EmbeddingCachePrefetchActor::UpdateCache() {
  for (const auto &item : hash_tables_) {
    auto hash_info = item.second;
    RETURN_IF_FALSE_WITH_LOG(PushCacheFromLocalHostToRemote(hash_info), "Push cache from local host to remote failed.");
    RETURN_IF_FALSE_WITH_LOG(PushCacheFromDeviceToLocalHost(hash_info), "Push cache from device to local host failed.");
    RETURN_IF_FALSE_WITH_LOG(PullCacheFromRemoteToLocalHost(hash_info), "Pull cache from remote to local host failed.");
    RETURN_IF_FALSE_WITH_LOG(PullCacheFromLocalHostToDevice(hash_info), "Pull cache from local host to device failed.");
  }
  return true;
}

bool EmbeddingCachePrefetchActor::PushCacheFromLocalHostToRemote(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_.host_to_server_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_host_cache_);
  auto host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
  MS_ERROR_IF_NULL(host_to_server_ids);
  auto host_to_server_index = embedding_host_cache_->host_to_server_index.get();
  MS_ERROR_IF_NULL(host_to_server_index);

  std::vector<float> swap_out_data;
  auto embedding_size = hash_info.embedding_size;
  swap_out_data.resize(swap_indices_size * embedding_size);
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());

  RETURN_IF_FALSE_WITH_LOG(LookupLocalHostCache(embedding_size, swap_indices_size, host_hash_table_addr,
                                                host_to_server_index, swap_out_data.data()),
                           "Lookup local host cache failed.");
  RETURN_IF_FALSE_WITH_LOG(PushEmbeddingsToRemote(host_to_server_ids, swap_indices_size, swap_out_data.data(),
                                                  swap_out_data.size() * sizeof(float)),
                           "Push embeddings to remote failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_.device_to_host_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_device_cache_->cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);

  auto device_cache_device_to_host_index = embedding_device_cache_->device_to_host_index.get();
  auto host_cache_device_to_host_index = embedding_host_cache_->device_to_host_index.get();
  MS_ERROR_IF_NULL(device_cache_device_to_host_index);
  MS_ERROR_IF_NULL(host_cache_device_to_host_index);
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto embedding_size = hash_info.embedding_size;
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);

  RETURN_IF_FALSE_WITH_LOG(
    LookupDeviceCache(embedding_device_cache_->hash_swap_index_addr_, hash_table_addr, swap_indices_size,
                      cache_vocab_size, embedding_size, embedding_device_cache_->hash_swap_value_addr_),
    "Lookup device cache failed.");
  MS_ERROR_IF_NULL(device_context_);
  RETURN_IF_FALSE_WITH_LOG(device_context_->SyncStream(stream_id_), "Synchronize stream failed.");
  RETURN_IF_FALSE_WITH_LOG(
    InsertLocalHostCache(embedding_size, IntToSize(swap_indices_size), host_cache_device_to_host_index,
                         swap_out_data.get(), host_hash_table_addr),
    "Insert local host cache failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::PullCacheFromRemoteToLocalHost(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_.server_to_host_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_host_cache_);
  auto server_to_host_ids = embedding_host_cache_->server_to_host_ids.get();
  MS_ERROR_IF_NULL(server_to_host_ids);
  auto server_to_host_index = embedding_host_cache_->server_to_host_index.get();
  MS_ERROR_IF_NULL(server_to_host_index);

  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  MS_ERROR_IF_NULL(host_hash_table_addr);
  auto embedding_size = hash_info.embedding_size;
  std::vector<float> lookup_result(swap_indices_size * embedding_size, 0);

  RETURN_IF_FALSE_WITH_LOG(PullEembeddingsFromRemote(server_to_host_ids, swap_indices_size, &lookup_result),
                           "Pull embedding from remote failed.");
  RETURN_IF_FALSE_WITH_LOG(InsertLocalHostCache(embedding_size, IntToSize(swap_indices_size), server_to_host_index,
                                                lookup_result.data(), host_hash_table_addr),
                           "Insert local host cache failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_.host_to_device_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_device_cache_->cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);

  auto host_cache_host_to_device_index = embedding_host_cache_->host_to_device_index.get();
  auto device_cache_host_to_device_index = embedding_device_cache_->host_to_device_index.get();
  MS_ERROR_IF_NULL(host_cache_host_to_device_index);
  MS_ERROR_IF_NULL(device_cache_host_to_device_index);

  auto embedding_size = hash_info.embedding_size;
  MS_ERROR_IF_NULL(hash_info.device_address.addr);
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  MS_ERROR_IF_NULL(hash_info.host_address);
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);
  RETURN_IF_FALSE_WITH_LOG(LookupLocalHostCache(embedding_size, swap_indices_size, host_hash_table_addr,
                                                host_cache_host_to_device_index, swap_out_data.get()),
                           "Lookup local host cache failed.");

  RETURN_IF_FALSE_WITH_LOG(
    UpdateDeviceCache(embedding_device_cache_->hash_swap_index_addr_, embedding_device_cache_->hash_swap_value_addr_,
                      swap_indices_size, cache_vocab_size, embedding_size, hash_table_addr),
    "Update device embedding cache failed.");
  MS_ERROR_IF_NULL(device_context_);
  RETURN_IF_FALSE_WITH_LOG(device_context_->SyncStream(stream_id_), "Synchronize stream failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::InsertLocalHostCache(size_t embedding_size, size_t insert_indices_size,
                                                       const int *insert_indices, const float *insert_data,
                                                       float *hash_table_addr) {
  MS_ERROR_IF_NULL(insert_indices);
  MS_ERROR_IF_NULL(insert_data);
  MS_ERROR_IF_NULL(hash_table_addr);

  size_t first_dim_size = local_host_cache_size_;
  size_t thread_num = insert_indices_size / kMaxIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t proc_len = (insert_indices_size + thread_num - 1) / thread_num;
  size_t i = 0;
  size_t offset = 0;

  auto insert_cache_func = [this](size_t insert_indices_size, size_t embedding_size, size_t first_dim_size,
                                  const int *insert_indices, const float *insert_data, float *hash_table_addr) {
    auto type_size = sizeof(float);
    size_t copy_len = embedding_size * type_size;
    size_t dest_len = copy_len;
    for (size_t i = 0; i < insert_indices_size; ++i) {
      int index = insert_indices[i];
      if (index >= 0 && index < SizeToInt(first_dim_size)) {
        auto ret =
          memcpy_s(hash_table_addr + index * embedding_size, dest_len, insert_data + i * embedding_size, copy_len);
        if (ret != EOK) {
          MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
          running_ = false;
          return;
        }
      }
    }
  };

  for (; i < thread_num; i++) {
    if (offset >= insert_indices_size) {
      break;
    }
    threads[i] = std::thread(insert_cache_func, proc_len, embedding_size, first_dim_size, insert_indices + offset,
                             insert_data + offset * embedding_size, hash_table_addr);
    offset += proc_len;
    if (offset + proc_len > insert_indices_size) {
      proc_len = insert_indices_size - offset;
    }
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  return running_;
}

void EmbeddingCachePrefetchActor::LookupEmbeddingTable(size_t indices_num, size_t embedding_size, size_t first_dim_size,
                                                       const float *input_addr, const int *indices_addr,
                                                       float *output_addr) {
  MS_ERROR_IF_NULL_WO_RET_VAL(input_addr);
  MS_ERROR_IF_NULL_WO_RET_VAL(indices_addr);
  MS_ERROR_IF_NULL_WO_RET_VAL(output_addr);

  auto type_size = sizeof(float);
  size_t lens = embedding_size * type_size;
  for (size_t i = 0; i < indices_num; ++i) {
    int index = indices_addr[i];
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = index * embedding_size;
      auto ret = memcpy_s(output_addr, (indices_num - i) * lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
        running_ = false;
        return;
      }
    } else {
      auto ret = memset_s(output_addr, (indices_num - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
        running_ = false;
        return;
      }
    }
    output_addr += embedding_size;
  }
}

bool EmbeddingCachePrefetchActor::LookupLocalHostCache(size_t embedding_size, size_t indices_num,
                                                       const float *hash_table_addr, const int *indices_addr,
                                                       float *output_addr) {
  MS_ERROR_IF_NULL(hash_table_addr);
  MS_ERROR_IF_NULL(indices_addr);
  MS_ERROR_IF_NULL(output_addr);

  size_t first_dim_size = local_host_cache_size_;
  size_t thread_num = indices_num / kMaxIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t proc_len = (indices_num + thread_num - 1) / thread_num;
  size_t i = 0;
  size_t offset = 0;

  for (; i < thread_num; i++) {
    if (offset >= indices_num) {
      break;
    }
    threads[i] =
      std::thread(&EmbeddingCachePrefetchActor::LookupEmbeddingTable, this, proc_len, embedding_size, first_dim_size,
                  hash_table_addr, indices_addr + offset, output_addr + offset * embedding_size);
    offset += proc_len;
    if (offset + proc_len > indices_num) {
      proc_len = indices_num - offset;
    }
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  return running_;
}

bool EmbeddingCachePrefetchActor::PullEembeddingsFromRemote(const int *ids, size_t ids_num,
                                                            std::vector<float> *outputs) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(outputs);

  std::vector<std::vector<int>> slice_ids_list(server_num_);
  // 1. Partition ids by remote embedding slice bound and get unique ids.
  RETURN_IF_FALSE_WITH_LOG(PartitionIds(ids, ids_num, &slice_ids_list), "Partition ids failed.");

  for (size_t i = 0; i < server_num_; i++) {
    auto &slice_ids = slice_ids_list[i];
    if (slice_ids.empty()) {
      continue;
    }

    // 2. Send unique ids to remote to do embedding lookup.
    RETURN_IF_FALSE_WITH_LOG(SendToRemote(i, slice_ids.data(), slice_ids.size() * sizeof(int)),
                             "Send ids to server failed.");
  }

  std::vector<std::vector<float>> slice_embeddings_list(server_num_);
  for (size_t i = 0; i < server_num_; i++) {
    if (slice_ids_list[i].empty()) {
      continue;
    }

    // 3. Wait embeddings result.
    auto &slice_embeddings = slice_embeddings_list[i];
    RETURN_IF_FALSE_WITH_LOG(WaitRespFromRemote(i, &slice_embeddings), "Wait response from server failed.");
  }

  // 4. Retrieve embeddings by input ids order.
  RETURN_IF_FALSE_WITH_LOG(RetrieveEmbeddings(ids, ids_num, slice_ids_list, slice_embeddings_list, outputs),
                           "Retrieve embeddings failed.");

  return true;
}

bool EmbeddingCachePrefetchActor::PushEmbeddingsToRemote(const int *ids, size_t ids_num, const float *embeddings,
                                                         size_t embeddings_len) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(embeddings);

  std::vector<std::vector<int>> slice_ids_list(server_num_);
  std::vector<std::vector<float>> slice_embeddings_list(server_num_);
  // 1. Partition ids end embeddings by remote embedding slice bound.
  RETURN_IF_FALSE_WITH_LOG(
    PartitionIdsAndEmbeddings(ids, ids_num, embeddings, embeddings_len, &slice_ids_list, &slice_embeddings_list),
    "Partition ids and embeddings failed.");

  for (size_t i = 0; i < server_num_; i++) {
    auto &slice_ids = slice_ids_list[i];
    if (slice_ids.empty()) {
      continue;
    }

    // 2. Send embeddings to remote.
    auto &slice_embeddings = slice_embeddings_list[i];
    RETURN_IF_FALSE_WITH_LOG(SendToRemote(i, slice_ids.data(), slice_ids.size() * sizeof(int), slice_embeddings.data(),
                                          slice_embeddings.size() * sizeof(float)),
                             "Send ids and embeddings to server failed.");
  }

  return true;
}

void EmbeddingCachePrefetchActor::GetRemoteEmbeddingSliceBound() {
  server_num_ = PSContext::instance()->server_num();
  if (server_num_ == 0) {
    MS_LOG(EXCEPTION) << "The server num is 0";
  }
  size_t average_slice_size = vocab_size_ / server_num_;
  std::vector<size_t> remote_embedding_slice_sizes = std::vector<size_t>(server_num_, average_slice_size);
  size_t rest_vocab_size = vocab_size_ % server_num_;
  for (size_t i = 0; i < rest_vocab_size; i++) {
    remote_embedding_slice_sizes[i] += 1;
  }

  size_t begin;
  size_t end;
  for (size_t i = 0; i < server_num_; i++) {
    if (i == 0) {
      begin = 0;
      end = remote_embedding_slice_sizes[0] - 1;
    } else {
      begin = remote_embedding_slice_bounds_[i - 1].second + 1;
      end = begin + remote_embedding_slice_sizes[i] - 1;
    }
    remote_embedding_slice_bounds_.emplace_back(begin, end);
  }
}

bool EmbeddingCachePrefetchActor::PartitionIds(const int *ids, size_t ids_num,
                                               std::vector<std::vector<int>> *slice_ids_list) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(slice_ids_list);

  for (size_t i = 0; i < slice_ids_list->size(); i++) {
    int begin = SizeToInt(remote_embedding_slice_bounds_[i].first);
    int end = SizeToInt(remote_embedding_slice_bounds_[i].second);

    mindspore::HashSet<int> unique_ids;
    (void)std::for_each(ids, ids + ids_num, [&](int id) {
      if (id >= begin && id <= end) {
        (void)unique_ids.insert(id);
      }
    });

    std::vector<int> &slice_ids = slice_ids_list->at(i);
    (void)std::for_each(unique_ids.begin(), unique_ids.end(), [&](int id) { slice_ids.push_back(id); });
  }

  return true;
}

bool EmbeddingCachePrefetchActor::PartitionIdsAndEmbeddings(const int *ids, size_t ids_num, const float *embeddings,
                                                            size_t embeddings_len,
                                                            std::vector<std::vector<int>> *slice_ids_list,
                                                            std::vector<std::vector<float>> *slice_embeddings_list) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(embeddings);
  MS_ERROR_IF_NULL(slice_ids_list);
  MS_ERROR_IF_NULL(slice_embeddings_list);

  if (ids_num == 0) {
    MS_LOG(WARNING) << "The ids number is 0";
    return true;
  }

  size_t embedding_dim = (embeddings_len / ids_num) / sizeof(float);
  size_t partition_num = slice_ids_list->size();
  for (size_t i = 0; i < partition_num; i++) {
    int begin = SizeToInt(remote_embedding_slice_bounds_[i].first);
    int end = SizeToInt(remote_embedding_slice_bounds_[i].second);

    std::vector<int> &slice_ids = slice_ids_list->at(i);
    std::vector<float> &slice_embeddings = slice_embeddings_list->at(i);
    for (size_t j = 0; j < ids_num; j++) {
      if (ids[j] >= begin && ids[j] <= end) {
        slice_ids.push_back(ids[j]);
        slice_embeddings.insert(slice_embeddings.end(), embeddings + (j * embedding_dim),
                                embeddings + (j * embedding_dim) + embedding_dim);
      }
    }
  }
  return true;
}

bool EmbeddingCachePrefetchActor::SendToRemote(size_t server_rank_id, const void *keys, size_t keys_len,
                                               const void *values, size_t values_len) {
  // Note: Need to implement the method via send actor.
  return true;
}

bool EmbeddingCachePrefetchActor::WaitRespFromRemote(size_t server_rank_id, std::vector<float> *outputs) {
  // Note: Need to implement the method via recv actor.
  return true;
}

bool EmbeddingCachePrefetchActor::RetrieveEmbeddings(const int *ids, size_t ids_num,
                                                     const std::vector<std::vector<int>> &slice_ids_list,
                                                     const std::vector<std::vector<float>> &slice_embeddings_list,
                                                     std::vector<float> *outputs) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(outputs);

  if (ids_num == 0) {
    MS_LOG(WARNING) << "The ids number is 0";
    return true;
  }

  // Merge all slice ids and embedding data address into ids_to_addrs map.
  mindspore::HashMap<int, const float *> ids_to_addrs;
  size_t embedding_dim = outputs->size() / ids_num;
  size_t offset = 0;
  for (size_t i = 0; i < slice_ids_list.size(); i++) {
    const std::vector<int> &slice_ids = slice_ids_list[i];
    if (slice_ids.empty()) {
      continue;
    }
    const std::vector<float> &slice_embeddings = slice_embeddings_list[i];
    const float *embeddings_data = slice_embeddings.data();
    for (size_t j = 0; j < slice_ids.size(); j++) {
      (void)ids_to_addrs.emplace(slice_ids[j], embeddings_data + offset);
      offset += embedding_dim;
    }
    offset = 0;
  }

  float *outputs_data = outputs->data();
  size_t dst_size = embedding_dim * sizeof(float);
  size_t src_size = dst_size;
  offset = 0;

  // Retrieve embeddings by input ids order.
  for (size_t i = 0; i < ids_num; i++) {
    auto id = ids[i];
    auto iter = ids_to_addrs.find(id);
    if (iter == ids_to_addrs.end()) {
      MS_LOG(WARNING) << "Can not find id[" << id << "]";
      continue;
    }

    auto ret = memcpy_s(outputs_data + offset, dst_size, iter->second, src_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
      return false;
    }
    offset += embedding_dim;
  }
  return true;
}
}  // namespace runtime
}  // namespace mindspore
