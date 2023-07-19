/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_COMPARISON_OP_NAME_H_
#define MINDSPORE_CORE_BASE_COMPARISON_OP_NAME_H_

namespace mindspore {
// Comparisons
constexpr auto kScalarEqOpName = "scalar_eq";
constexpr auto kScalarLtOpName = "scalar_lt";
constexpr auto kScalarGtOpName = "scalar_gt";
constexpr auto kScalarLeOpName = "scalar_le";
constexpr auto kScalarGeOpName = "scalar_ge";
constexpr auto kScalarBoolOpName = "ScalarBool";
constexpr auto kBoolNotOpName = "bool_not";
constexpr auto kNotEqualOpName = "NotEqual";
constexpr auto kLogicalXorOpName = "LogicalXor";
constexpr auto kEqualOpName = "Equal";
constexpr auto kGreaterEqualOpName = "GreaterEqual";
constexpr auto kGreaterOpName = "Greater";
constexpr auto kLessEqualOpName = "LessEqual";
constexpr auto kLessOpName = "Less";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_COMPARISON_OP_NAME_H_