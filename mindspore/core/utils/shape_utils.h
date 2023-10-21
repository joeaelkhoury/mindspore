/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_SHAPE_UTILS_INFO_H_
#define MINDSPORE_SHAPE_UTILS_INFO_H_

#include <algorithm>
#include <vector>
#include <string>
#include "abstract/dshape.h"
#include "utils/log_adapter.h"

namespace mindspore {
inline std::string ShapeVectorToString(const ShapeVector &shape) {
  std::string str_shape = "";
  for (auto &item : shape) {
    str_shape += std::to_string(item) + ", ";
  }
  str_shape = str_shape.length() >= 2 ? str_shape.substr(0, str_shape.length() - 2) : str_shape;
  return str_shape;
}

inline size_t SizeOf(const ShapeVector &shape) {
  size_t data_size = 1;
  for (auto dim : shape) {
    if (dim <= 0) {
      // For dynamic shape which has negative dimensions, data size should be zero.
      return 0;
    }
    if (SIZE_MAX / dim < data_size) {
      MS_EXCEPTION(ValueError) << "The product value of shape (" << ShapeVectorToString(shape)
                               << ") exceeds the maximum value of size_t: " << SIZE_MAX;
    }
    data_size *= static_cast<size_t>(dim);
  }
  return data_size;
}

inline bool IsDynamicRank(const ShapeVector &shape) {
  if ((shape.size() == abstract::Shape::kDynamicRankLen) && (shape[0] == abstract::Shape::kShapeRankAny)) {
    return true;
  }
  if (std::any_of(shape.cbegin(), shape.cend(),
                  [](ShapeValueDType s) { return s == abstract::Shape::kShapeRankAny; })) {
    MS_EXCEPTION(ValueError) << "Shape should have only one -2 or no -2 at all but got (" << ShapeVectorToString(shape)
                             << ").";
  }
  return false;
}

inline bool IsDynamicShape(const ShapeVector &shape) {
  return std::any_of(shape.cbegin(), shape.cend(),
                     [](ShapeValueDType s) { return s == abstract::Shape::kShapeDimAny; });
}

inline bool IsDynamic(const ShapeVector &shape) {
  if (std::any_of(shape.begin(), shape.end(), [](ShapeValueDType s) { return s < abstract::Shape::kShapeRankAny; })) {
    MS_EXCEPTION(ValueError) << "Shape should not have values less than -2 but got (" << ShapeVectorToString(shape)
                             << ").";
  }
  return IsDynamicRank(shape) || IsDynamicShape(shape);
}

inline bool IsShapeEmpty(const ShapeVector &shape) {
  constexpr size_t kOne = 1;
  constexpr size_t kZero = 0;
  return shape.size() == kOne && shape[0] == kZero;
}

template <typename T>
std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  ss << "[";
  auto size = values.size();
  for (size_t i = 0; i < size; ++i) {
    ss << values[i];
    if (i != size - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}
}  // namespace mindspore

#endif  // MINDSPORE_SHAPE_UTILS_INFO_H_
