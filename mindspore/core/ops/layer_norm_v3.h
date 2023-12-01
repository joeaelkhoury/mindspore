/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_LAYER_NORM_V3_H_
#define MINDSPORE_CORE_OPS_LAYER_NORM_V3_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLayerNormV3 = "LayerNormV3";
/// \brief Applies the Layer Normalization to the input tensor.
/// Refer to Python API @ref mindspore.ops.LayerNormV3 for more details.
class MIND_API LayerNormV3 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LayerNormV3);
  /// \brief Constructor.
  LayerNormV3() : BaseOperator(kNameLayerNormV3) {}
  explicit LayerNormV3(const std::string k_name) : BaseOperator(k_name) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.LayerNormV3 for the inputs.
  void Init(const int64_t begin_norm_axis = 1, const int64_t begin_params_axis = 1, const float epsilon = 1e-7);
  /// \brief Set begin_norm_axis.
  void set_begin_norm_axis(const int64_t begin_norm_axis);
  /// \brief Set begin_params_axis.
  void set_begin_params_axis(const int64_t begin_params_axis);
  /// \brief Set epsilon.
  void set_epsilon(const float epsilon);
  /// \brief Get begin_norm_axis.
  ///
  /// \return begin_norm_axis.
  int64_t get_begin_norm_axis() const;
  /// \brief Get begin_params_axis.
  ///
  /// \return begin_params_axis.
  int64_t get_begin_params_axis() const;
  /// \brief Get epsilon.
  ///
  /// \return epsilon.
  float get_epsilon() const;
};

using PrimLayerNormV3Ptr = std::shared_ptr<LayerNormV3>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LAYER_NORM_H_
